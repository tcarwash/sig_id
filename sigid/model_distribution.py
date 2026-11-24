"""Portable, validated Signal ID model release bundles and first-run installer.

No model bytes, training examples, local paths or private capture IDs are
committed to Git. Publication requires a separate reviewed release gate.
Downloading is explicit unless SIG_ID_AUTO_INSTALL_MODELS=1 is configured.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile

BUNDLE_SCHEMA = "sig-id-model-bundle-v1"
OFFICIAL_REPO = "tcarwash/sig_id"
ASSET_NAME = "sig-id-models.zip"
ARCHIVE_LIMIT = 32 * 1024 * 1024
FILE_LIMIT = 16 * 1024 * 1024
TAG_RE = re.compile(r"models-v[0-9]{4}\.[0-9]{2}\.[0-9]{2}(?:\.[0-9]+)?\Z")
SAFE_FILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
REQUIRED_FILES = {"classifier.json", "cv-signal.pt", "cv-signal.pt.json"}
# Do not distribute reviewer capture UUIDs, private dataset paths, real
# recording manifests or source hashes from original model metadata.
CLASSIFIER_PUBLIC_KEYS = {
    "abstention_threshold", "classes", "classifier_family_by_method",
    "classifier_window_seconds", "experts", "families",
    "family_abstention_threshold", "family_by_method", "family_calibration",
    "family_model", "family_thresholds", "feature_count",
    "feature_frequency_scope", "feature_version", "method_names",
    "mode_calibration", "mode_classes_by_family", "mode_details",
    "mode_thresholds", "model_sha256", "model_type", "registry_is_training_taxonomy",
    "sample_rate_hz", "schema_version", "unknown_rejection_validated",
}
DETECTOR_PUBLIC_KEYS = {
    "schema_version", "model_file", "model_sha256", "model_size_bytes",
    "preprocessing_version", "image_shape", "model_image_size",
    "confidence_threshold", "confidence_threshold_source",
    "class_agnostic", "classes", "task", "scope",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(data: dict) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _load_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _component_files(meta: dict) -> set[str]:
    if meta.get("model_type") != "hierarchical_family_experts" or meta.get("schema_version") != 1:
        raise ValueError("only hierarchical classifier releases are supported")
    components = [meta.get("family_model"), *(meta.get("experts") or {}).values()]
    if not components or any(not isinstance(c, dict) for c in components):
        raise ValueError("invalid classifier component index")
    names: set[str] = set()
    for component in components:
        name = component.get("file")
        digest = component.get("sha256")
        if not isinstance(name, str) or not SAFE_FILE_RE.fullmatch(name) or not name.endswith(".npz"):
            raise ValueError("unsafe classifier component filename")
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("classifier component missing SHA-256")
        names.add(name)
    return names


def _validate_public_metadata(classifier: dict, detector: dict) -> None:
    from .cv_detector import PREPROCESSING_VERSION
    if detector.get("schema_version") != 1 or detector.get("model_file") != "cv-signal.pt":
        raise ValueError("unsupported detector metadata")
    if detector.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("detector preprocessing version incompatible with this application")
    if classifier.get("model_type") != "hierarchical_family_experts":
        raise ValueError("incompatible classifier format")
    _component_files(classifier)


def _checkpoint_private_path_markers(data: bytes) -> list[str]:
    """Conservatively flag local path markers in model checkpoint metadata.

    PyTorch weights can contain pickle metadata with original local training
    paths even after public sidecar JSON has been sanitized. Never unpickle
    untrusted checkpoint bytes during this privacy check.
    """
    patterns = (b"/home/", b"/Users/", b"/storage5/", b"\\Users\\", b"\\Documents\\")
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            contents = b"".join(z.read(entry) for entry in z.namelist()
                                if entry.endswith("data.pkl") and z.getinfo(entry).file_size <= 2 * 1024 * 1024)
    else:
        contents = data[:FILE_LIMIT]
    return [pattern.decode("ascii", errors="replace") for pattern in patterns if pattern in contents]


def _approval_status(classifier: dict, detector: dict, report: dict | None) -> dict:
    if report is None:
        return {"status": "development", "note": "Not eligible for publication or automatic install"}
    if not report.get("licensing_reviewed") or not report.get("corresponding_source_available"):
        raise ValueError("release approval requires verified Ultralytics redistribution/source licensing")
    if report.get("classifier_model_sha256") != classifier.get("model_sha256"):
        raise ValueError("approval report classifier hash mismatch")
    if report.get("detector_model_sha256") != detector.get("model_sha256"):
        raise ValueError("approval report detector hash mismatch")
    recall = report.get("live_cv_recall_iou50")
    precision = report.get("live_cv_precision_iou50")
    fp = report.get("false_positives_per_hour")
    if (not isinstance(recall, (float, int)) or recall < 0.50 or recall > 1
            or not isinstance(precision, (float, int)) or precision < 0.70 or precision > 1
            or not isinstance(fp, (float, int)) or fp < 0 or fp > 10):
        raise ValueError("release fails real held-out recall/precision/FP-per-hour gate")
    family = classifier.get("family_calibration", {})
    if family.get("status", "").startswith("reject_all") or classifier.get("family_abstention_threshold", 0) > 1:
        raise ValueError("classifier calibration rejects every label; cannot publish")
    if any(info.get("status", "").startswith("reject_all") for info in classifier.get("mode_calibration", {}).values()):
        raise ValueError("mode experts still have reject-all calibration")
    if not report.get("independent_real_holdout") or not report.get("evaluator"):
        raise ValueError("release requires independent real holdout provenance and evaluator")
    return {
        "status": "approved",
        "independent_real_holdout": str(report["independent_real_holdout"])[:256],
        "live_cv_recall_iou50": float(recall),
        "live_cv_precision_iou50": float(precision),
        "false_positives_per_hour": float(fp),
        "licensing_reviewed": True,
    }


def build_bundle(
    release_tag: str, classifier_path: str | Path, detector_path: str | Path,
    output_path: str | Path, *, approval_report: str | Path | None = None,
) -> dict:
    if not TAG_RE.fullmatch(release_tag):
        raise ValueError("release tag must be models-vYYYY.MM.DD[.N]")
    classifier_path = Path(classifier_path).resolve(strict=True)
    detector_path = Path(detector_path).resolve(strict=True)
    if classifier_path.name != "classifier.json" or detector_path.name != "cv-signal.pt":
        raise ValueError("expected classifier.json and cv-signal.pt")
    classifier = _load_json(classifier_path)
    detector = _load_json(detector_path.with_suffix(".pt.json"))
    _validate_public_metadata(classifier, detector)
    detector_bytes = detector_path.read_bytes()
    if _sha(detector_bytes) != detector["model_sha256"]:
        raise ValueError("detector file differs from model metadata hash")
    private_markers = _checkpoint_private_path_markers(detector_bytes)
    if approval_report is not None and private_markers:
        raise ValueError("public detector checkpoint contains local training paths; sanitize and reevaluate it before publication")

    from .classifier import load_model
    load_model(classifier_path)  # Verifies original component hashes, arrays, and feature version.
    verdict = _approval_status(
        classifier, detector,
        _load_json(approval_report) if approval_report is not None else None,
    )
    public_classifier = {key: value for key, value in classifier.items() if key in CLASSIFIER_PUBLIC_KEYS}
    public_detector = {key: value for key, value in detector.items() if key in DETECTOR_PUBLIC_KEYS}
    files: dict[str, bytes] = {
        "classifier.json": _json_bytes(public_classifier),
        "cv-signal.pt": detector_bytes,
        "cv-signal.pt.json": _json_bytes(public_detector),
    }
    for name in sorted(_component_files(classifier)):
        payload = (classifier_path.parent / name).read_bytes()
        expected = next(c["sha256"] for c in
                        [classifier["family_model"], *classifier["experts"].values()]
                        if c["file"] == name)
        if _sha(payload) != expected:
            raise ValueError("classifier component checksum mismatch")
        files[name] = payload
    if any(len(value) > FILE_LIMIT for value in files.values()):
        raise ValueError("model file exceeds safe release size limit")
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "release_tag": release_tag,
        "evaluation": verdict,
        "checkpoint_privacy": "local-path-markers-detected" if private_markers else "no-common-local-path-markers",
        "compatibility": {
            "classifier_feature_version": classifier["feature_version"],
            "detector_preprocessing_version": detector["preprocessing_version"],
        },
        "files": {name: {"sha256": _sha(blob), "size_bytes": len(blob)}
                  for name, blob in sorted(files.items())},
        "license": {
            "detector_framework": "Ultralytics YOLO (AGPL-3.0 or appropriate license)",
            "distribution_reviewed": verdict["status"] == "approved",
        },
    }
    files["manifest.json"] = _json_bytes(manifest)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".zip", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with zipfile.ZipFile(temporary, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in sorted(files.items()):
                archive.writestr(name, value)
        if temporary.stat().st_size > ARCHIVE_LIMIT:
            raise ValueError("bundle exceeds maximum download size")
        temporary.replace(path)
        checksum = _sha(path.read_bytes())
        path.with_name(path.name + ".sha256").write_text(f"{checksum}  {path.name}\n")
    finally:
        temporary.unlink(missing_ok=True)
    return {"archive": str(path), "sha256": checksum, **manifest}


def _verify_archive(data: bytes, *, allow_development: bool) -> tuple[dict, dict[str, bytes]]:
    if len(data) > ARCHIVE_LIMIT:
        raise ValueError("model archive exceeds maximum size")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid model archive") from exc
    with archive:
        members = archive.infolist()
        names = [entry.filename for entry in members]
        if (len(names) != len(set(names)) or "manifest.json" not in names or
                any(not SAFE_FILE_RE.fullmatch(name) for name in names) or
                any(entry.is_dir() or entry.file_size > FILE_LIMIT for entry in members) or
                sum(entry.file_size for entry in members) > ARCHIVE_LIMIT):
            raise ValueError("unsafe archive paths, duplicates or decompression sizes")
        metadata_info = archive.getinfo("manifest.json")
        if metadata_info.file_size > 128 * 1024:
            raise ValueError("model manifest exceeds safe size limit")
        manifest_raw = archive.read("manifest.json")
        manifest = json.loads(manifest_raw)
        if manifest.get("schema") != BUNDLE_SCHEMA or not TAG_RE.fullmatch(manifest.get("release_tag", "")):
            raise ValueError("unsupported model bundle")
        if not allow_development and manifest.get("evaluation", {}).get("status") != "approved":
            raise ValueError("development models may not be automatically installed")
        expected = manifest.get("files")
        if not isinstance(expected, dict) or REQUIRED_FILES - set(expected):
            raise ValueError("bundle is missing required models")
        if set(names) != (set(expected) | {"manifest.json"}):
            raise ValueError("bundle manifest/file list mismatch")
        payloads = {}
        for name, metadata in expected.items():
            if (not isinstance(metadata, dict) or not isinstance(metadata.get("size_bytes"), int)
                    or metadata["size_bytes"] < 0 or metadata["size_bytes"] > FILE_LIMIT
                    or not isinstance(metadata.get("sha256"), str)):
                raise ValueError("invalid release file metadata")
            value = archive.read(name)
            if _sha(value) != metadata.get("sha256") or len(value) != metadata.get("size_bytes"):
                raise ValueError(f"model file hash/length mismatch: {name}")
            payloads[name] = value
    classifier = json.loads(payloads["classifier.json"])
    detector = json.loads(payloads["cv-signal.pt.json"])
    _validate_public_metadata(classifier, detector)
    if set(_component_files(classifier)) - set(payloads):
        raise ValueError("classifier component missing from bundle")
    if _sha(payloads["cv-signal.pt"]) != detector.get("model_sha256"):
        raise ValueError("detector hash does not match metadata")
    for item in [classifier["family_model"], *classifier["experts"].values()]:
        if _sha(payloads[item["file"]]) != item["sha256"]:
            raise ValueError("classifier component hash does not match metadata")
    return manifest, payloads


def install_bundle(
    archive_path: str | Path, data_dir: str | Path, *,
    expected_sha256: str | None = None, allow_development: bool = False,
) -> dict:
    data = Path(archive_path).read_bytes()
    if expected_sha256 and _sha(data) != expected_sha256:
        raise ValueError("release archive SHA-256 differs from GitHub checksum")
    manifest, payloads = _verify_archive(data, allow_development=allow_development)
    release_tag = manifest["release_tag"]
    model_root = Path(data_dir).expanduser().resolve() / "models"
    releases_root = model_root / "releases"
    releases_root.mkdir(parents=True, exist_ok=True)
    target = releases_root / release_tag
    if target.is_symlink():
        raise ValueError("model release directory must not be a symlink")
    if target.exists():
        if not target.is_dir():
            raise ValueError("model release destination is not a directory")
        for name, blob in payloads.items():
            if (target / name).is_symlink() or not (target / name).is_file() or _sha((target / name).read_bytes()) != _sha(blob):
                raise ValueError("existing release directory has different model bytes")
    else:
        with tempfile.TemporaryDirectory(dir=releases_root, prefix=".install-") as temp:
            stage = Path(temp)
            for name, blob in payloads.items():
                (stage / name).write_bytes(blob)
            (stage / "manifest.json").write_bytes(_json_bytes(manifest))
            # Loading the classifier validates weight dimensions and model compatibility.
            from .classifier import load_model
            load_model(stage / "classifier.json")
            stage.rename(target)
    pointer = {
        "classifier": str(target / "classifier.json"),
        "cv_detector": str(target / "cv-signal.pt"),
    }
    model_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=model_root, prefix=".pointer-", delete=False) as f:
        pending = Path(f.name)
        json.dump(pointer, f, indent=2, sort_keys=True)
        f.write("\n")
    try:
        pending.replace(model_root / "active-models.json")
    finally:
        pending.unlink(missing_ok=True)
    return {"release_tag": release_tag, "model_root": str(target),
            "pointer": str(model_root / "active-models.json"),
            "evaluation": manifest["evaluation"]["status"]}


def _download(url: str, *, max_bytes: int) -> bytes:
    request = urllib.request.Request(url, headers={
        "User-Agent": "sig-id-model-installer/1.0",
        "Accept": "application/vnd.github+json",
    })
    with urllib.request.urlopen(request, timeout=15) as response:
        if response.status != 200:
            raise ValueError("GitHub download failed")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("GitHub asset exceeds size limit")
    return data


def install_latest_from_github(data_dir: str | Path, *, repo: str = OFFICIAL_REPO) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("invalid GitHub repository")
    releases_url = f"https://api.github.com/repos/{repo}/releases?per_page=50"
    records = json.loads(_download(releases_url, max_bytes=1024 * 1024))
    if not isinstance(records, list):
        raise ValueError("unexpected GitHub release listing")
    release = next((item for item in records if
                    TAG_RE.fullmatch(item.get("tag_name", "")) and
                    not item.get("draft") and not item.get("prerelease")), None)
    if release is None:
        raise ValueError("no approved public sig_id model release found")
    assets = {item.get("name"): item for item in release.get("assets", [])}
    if ASSET_NAME not in assets or ASSET_NAME + ".sha256" not in assets:
        raise ValueError("model release lacks bundle or checksum asset")
    for name in (ASSET_NAME, ASSET_NAME + ".sha256"):
        url = assets[name].get("browser_download_url", "")
        expected_url = f"https://github.com/{repo}/releases/download/{release['tag_name']}/{name}"
        if url != expected_url:
            raise ValueError("unexpected GitHub release asset URL")
    digest_text = _download(assets[ASSET_NAME + ".sha256"]["browser_download_url"], max_bytes=512).decode()
    match = re.fullmatch(r"([0-9a-f]{64})  sig-id-models\.zip\s*", digest_text)
    if not match:
        raise ValueError("unrecognized release SHA-256 checksum format")
    archive_bytes = _download(assets[ASSET_NAME]["browser_download_url"], max_bytes=ARCHIVE_LIMIT)
    if _sha(archive_bytes) != match.group(1):
        raise ValueError("downloaded model bundle failed SHA-256 check")
    bundled_manifest, _ = _verify_archive(archive_bytes, allow_development=False)
    if bundled_manifest["release_tag"] != release["tag_name"]:
        raise ValueError("GitHub release tag and archive manifest differ")
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / ASSET_NAME
        path.write_bytes(archive_bytes)
        return install_bundle(path, data_dir, expected_sha256=match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true", help="create a model archive, locally only")
    group.add_argument("--install", type=Path, help="verify and install a local archive")
    group.add_argument("--install-latest", action="store_true", help="download approved GitHub release")
    parser.add_argument("--tag", help="model release tag, models-vYYYY.MM.DD")
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--approval-report", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path(os.getenv("SIG_ID_DATA_DIR", "~/.local/share/sig-id")).expanduser())
    parser.add_argument("--allow-development", action="store_true",
                        help="only for local testing; never use with GitHub auto-install")
    args = parser.parse_args()
    if args.build:
        if not args.tag or not args.classifier or not args.detector or not args.output:
            parser.error("--build requires --tag, --classifier, --detector, --output")
        result = build_bundle(args.tag, args.classifier, args.detector, args.output,
                              approval_report=args.approval_report)
    elif args.install:
        result = install_bundle(args.install, args.data_dir, allow_development=args.allow_development)
    else:
        if args.allow_development:
            parser.error("cannot allow development models from GitHub")
        result = install_latest_from_github(args.data_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
