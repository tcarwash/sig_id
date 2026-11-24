"""Model bundle packaging, integrity and first-run distribution tests."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from sigid import model_distribution as md


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _models(root: Path, *, reject_all: bool = False) -> tuple[Path, Path]:
    from sigid.classifier import FEATURE_COUNT_BY_VERSION
    from sigid.cv_detector import PREPROCESSING_VERSION

    root.mkdir(parents=True, exist_ok=True)
    version = "passband-log-spectrum-envelope-roi-fixed-window-v4"
    features = FEATURE_COUNT_BY_VERSION[version]
    arrays = {
        "mean": np.zeros(features, dtype=np.float32),
        "scale": np.ones(features, dtype=np.float32),
        "weights": np.zeros((features, 1), dtype=np.float32),
        "bias": np.zeros(1, dtype=np.float32),
    }
    components = []
    for name in ("classifier.family.npz", "classifier.expert-0.npz"):
        path = root / name
        np.savez(path, **arrays)
        components.append({"file": name, "sha256": _sha(path.read_bytes())})
    classifier = {
        "schema_version": 1,
        "model_type": "hierarchical_family_experts",
        "feature_version": version,
        "classifier_window_seconds": 2.0,
        "feature_count": features,
        "families": ["FSK"],
        "mode_classes_by_family": {"FSK": ["2fsk"]},
        "family_by_method": {"2fsk": "FSK"},
        "method_names": {"2fsk": "binary frequency shift keying"},
        "family_model": components[0],
        "experts": {"FSK": components[1]},
        "model_sha256": "a" * 64,
        "family_calibration": {"status": "reject_all" if reject_all else "ok"},
        "family_abstention_threshold": 1.000001 if reject_all else 0.55,
        "mode_calibration": {"FSK": {"status": "reject_all" if reject_all else "ok"}},
        "capture_source_path": "/private/user/real-captures/secret-id",
        "reviewed_real_training": {"reviewer_uuid": "private-id"},
    }
    classifier_path = root / "classifier.json"
    classifier_path.write_text(json.dumps(classifier))
    model_path = root / "cv-signal.pt"
    model_path.write_bytes(b"FAKE YOLO CHECKPOINT - ONLY FOR BUNDLE UNIT TESTS")
    metadata = {
        "schema_version": 1,
        "model_file": model_path.name,
        "model_sha256": _sha(model_path.read_bytes()),
        "preprocessing_version": PREPROCESSING_VERSION,
        "model_size_bytes": model_path.stat().st_size,
        "confidence_threshold": 0.5,
        "classes": ["signal"],
        "dataset_path": "/private/user/recordings/secret",
        "training_run": "/private/research/secret",
    }
    model_path.with_suffix(".pt.json").write_text(json.dumps(metadata))
    return classifier_path, model_path


def _approved_report(classifier: Path, detector: Path) -> dict:
    d = json.loads(classifier.read_text())
    y = json.loads(detector.with_suffix(".pt.json").read_text())
    return {
        "licensing_reviewed": True,
        "corresponding_source_available": True,
        "classifier_model_sha256": d["model_sha256"],
        "detector_model_sha256": y["model_sha256"],
        "live_cv_recall_iou50": 0.72,
        "live_cv_precision_iou50": 0.83,
        "false_positives_per_hour": 2.0,
        "independent_real_holdout": "holdout-sha-verified",
        "evaluator": "offline-evaluator",
    }


def test_bundle_sanitizes_private_training_paths_and_installs_atomically(tmp_path):
    classifier, detector = _models(tmp_path / "models")
    archive = tmp_path / "sig-id-models.zip"
    result = md.build_bundle("models-v2026.11.24", classifier, detector, archive)
    assert result["evaluation"]["status"] == "development"
    assert archive.stat().st_size < md.ARCHIVE_LIMIT
    assert (tmp_path / "sig-id-models.zip.sha256").exists()
    with zipfile.ZipFile(archive) as z:
        safe_meta = z.read("classifier.json")
        safe_cv = z.read("cv-signal.pt.json")
        assert b"private" not in safe_meta + safe_cv
        assert b"reviewed_real_training" not in safe_meta
        assert b"model_sha256" in safe_cv

    data_dir = tmp_path / "installed"
    with pytest.raises(ValueError, match="development"):
        md.install_bundle(archive, data_dir)
    assert not (data_dir / "models" / "active-models.json").exists()
    installed = md.install_bundle(archive, data_dir, allow_development=True)
    assert installed["evaluation"] == "development"
    pointer = json.loads((data_dir / "models" / "active-models.json").read_text())
    assert Path(pointer["classifier"]).is_file()
    assert Path(pointer["cv_detector"]).is_file()
    again = md.install_bundle(archive, data_dir, allow_development=True)
    assert again["pointer"] == installed["pointer"]


def test_release_gate_blocks_inadequate_classifier_and_unreviewed_license(tmp_path):
    classifier, detector = _models(tmp_path / "input", reject_all=True)
    report = _approved_report(classifier, detector)
    approved = tmp_path / "approval.json"
    approved.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="rejects every label"):
        md.build_bundle("models-v2026.11.24", classifier, detector,
                        tmp_path / "would-be-public.zip", approval_report=approved)
    assert not (tmp_path / "would-be-public.zip").exists()

    classifier, detector = _models(tmp_path / "other")
    report = _approved_report(classifier, detector)
    report["licensing_reviewed"] = False
    approved.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="licensing"):
        md.build_bundle("models-v2026.11.24", classifier, detector,
                        tmp_path / "would-be-public.zip", approval_report=approved)


def test_approved_release_and_corrupt_checksum_do_not_change_pointer(tmp_path):
    classifier, detector = _models(tmp_path / "input")
    approved = tmp_path / "approved.json"
    approved.write_text(json.dumps(_approved_report(classifier, detector)))
    archive = tmp_path / "sig-id-models.zip"
    built = md.build_bundle("models-v2026.11.24", classifier, detector,
                            archive, approval_report=approved)
    assert built["evaluation"]["status"] == "approved"
    root = tmp_path / "installed"
    md.install_bundle(archive, root, expected_sha256=built["sha256"])
    pointer = root / "models" / "active-models.json"
    initial = pointer.read_bytes()

    corrupted = tmp_path / "bad.zip"
    corrupted.write_bytes(archive.read_bytes() + b"evil")
    with pytest.raises(ValueError, match="SHA-256"):
        md.install_bundle(corrupted, root, expected_sha256=built["sha256"])
    assert pointer.read_bytes() == initial


def test_archive_rejects_path_traversal_and_disallowed_extra_files(tmp_path):
    classifier, detector = _models(tmp_path / "input")
    archive = tmp_path / "sig-id-models.zip"
    md.build_bundle("models-v2026.11.24", classifier, detector, archive)
    with zipfile.ZipFile(archive) as z:
        payloads = {name: z.read(name) for name in z.namelist()}
    payloads["../bad.txt"] = b"no"
    hostile = tmp_path / "hostile.zip"
    with zipfile.ZipFile(hostile, "w") as z:
        for name, blob in payloads.items():
            z.writestr(name, blob)
    with pytest.raises(ValueError, match="unsafe archive"):
        md.install_bundle(hostile, tmp_path / "target", allow_development=True)
    assert not (tmp_path / "bad.txt").exists()


def test_github_installer_requires_two_release_assets_and_matching_tag(tmp_path, monkeypatch):
    classifier, detector = _models(tmp_path / "input")
    approved = tmp_path / "approved.json"
    approved.write_text(json.dumps(_approved_report(classifier, detector)))
    archive = tmp_path / "sig-id-models.zip"
    md.build_bundle("models-v2026.11.24", classifier, detector, archive, approval_report=approved)
    release_assets = [{
        "tag_name": "models-v2026.11.24",
        "draft": False,
        "prerelease": False,
        "assets": [
            {"name": name, "browser_download_url":
             f"https://github.com/tcarwash/sig_id/releases/download/models-v2026.11.24/{name}"}
            for name in (md.ASSET_NAME, md.ASSET_NAME + ".sha256")
        ],
    }]
    requests = []
    def fake_download(url, *, max_bytes):
        requests.append(url)
        if url.endswith("/releases?per_page=50"):
            return json.dumps(release_assets).encode()
        if url.endswith(".sha256"):
            return (tmp_path / (md.ASSET_NAME + ".sha256")).read_bytes()
        return archive.read_bytes()

    monkeypatch.setattr(md, "_download", fake_download)
    installed = md.install_latest_from_github(tmp_path / "new-install")
    assert installed["evaluation"] == "approved"
    assert len(requests) == 3

    release_assets[0]["tag_name"] = "models-v2026.11.25"
    with pytest.raises(ValueError, match="unexpected GitHub release asset URL"):
        md.install_latest_from_github(tmp_path / "another-install")
    assert not (tmp_path / "another-install" / "models" / "active-models.json").exists()


def test_app_can_use_installed_model_catalog_without_source_tree_pointer(tmp_path, monkeypatch):
    from sigid import api
    classifier, detector = _models(tmp_path / "model-pair")
    bundle = tmp_path / md.ASSET_NAME
    md.build_bundle("models-v2026.11.24", classifier, detector, bundle)
    data = tmp_path / "app-data"
    md.install_bundle(bundle, data, allow_development=True)

    class DummyDetector:
        def __init__(self, path):
            self.model_path = str(path)
            self.model_sha256 = "dummy-test-sha"
            self.model_version = "dummy-test"
            self.compute_backend = "cpu"
            self.confidence_threshold = 0.5
    monkeypatch.setattr(api, "YoloSignalDetector", DummyDetector)
    monkeypatch.delenv("SIGID_ACTIVE_MODELS_JSON", raising=False)
    monkeypatch.delenv("SIG_ID_MODEL_PATH", raising=False)
    monkeypatch.delenv("SIG_ID_CV_MODEL_PATH", raising=False)
    app = api.create_app(data_dir=data)
    assert app.state.classifier is not None
    assert app.state.cv_detector is not None
    assert app.state.classifier_error is None
    assert app.state.cv_detector_error is None


def test_explicit_auto_install_failure_keeps_manual_app_available(tmp_path, monkeypatch):
    from sigid import api
    monkeypatch.setenv("SIG_ID_AUTO_INSTALL_MODELS", "1")
    monkeypatch.delenv("SIGID_ACTIVE_MODELS_JSON", raising=False)
    monkeypatch.delenv("SIG_ID_MODEL_PATH", raising=False)
    monkeypatch.delenv("SIG_ID_CV_MODEL_PATH", raising=False)
    calls = []
    def unavailable(directory):
        calls.append(str(directory))
        raise ValueError("no approved public release")
    monkeypatch.setattr(md, "install_latest_from_github", unavailable)
    app = api.create_app(data_dir=tmp_path / "offline")
    assert calls == [str(tmp_path / "offline")]
    assert app.state.classifier is None
    assert app.state.cv_detector is None
    assert app.state.capture_store is not None


def test_public_release_rejects_local_paths_embedded_in_checkpoint(tmp_path):
    classifier, detector = _models(tmp_path / 'source')
    detector.write_bytes(detector.read_bytes() + b' training /home/tester/private-data/sessions')
    meta = json.loads(detector.with_suffix('.pt.json').read_text())
    meta['model_sha256'] = _sha(detector.read_bytes())
    meta['model_size_bytes'] = detector.stat().st_size
    detector.with_suffix('.pt.json').write_text(json.dumps(meta))
    approved = tmp_path / 'approved.json'
    approved.write_text(json.dumps(_approved_report(classifier, detector)))
    with pytest.raises(ValueError, match='contains local training paths'):
        md.build_bundle('models-v2026.11.24', classifier, detector,
                        tmp_path / 'should-not-publish.zip', approval_report=approved)
