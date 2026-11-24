"""Frozen-lineage detector → classifier → optional native-decoder evaluation.

This is receive-only. Test labels/transcripts are never supplied to detection,
classification or decoder selection. Without independently transcribed message
truth, provisional text is NOT scored as a correct decoded payload.

Usage:
  python -m tools.bench_reviewed_chain --dataset <export> --detector <pt> \
      --classifier <classifier.json> --output <result.json> --device cpu
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import time
from typing import Any
import wave

import numpy as np

from sigid.classifier import (
    CLASSIFIER_WINDOW_SECONDS, FIXED_WINDOW_FEATURE_VERSIONS,
    ROI_FEATURE_VERSIONS, extract_features, load_model, propose_loaded,
)
from sigid.cv_detector import MAX_FREQUENCY_HZ, YoloSignalDetector
from sigid.cv_training import _match_pairs, _read_yolo_boxes
from sigid.native_live_decode import candidate_modes, decode_live_candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_manifest(dataset: Path, split: str) -> tuple[list[dict], dict]:
    """Reject split/provenance leakage BEFORE any model inference.

    Capture SHA and lineage group are separate isolation keys. This detects
    mistakes in either: duplicate derived recordings crossing source splits
    and original captures appearing under different labels.
    """
    if split not in ("test", "validation"):
        raise ValueError("evaluation split must be validation or test")
    manifest_path = dataset / "windows.jsonl"
    export_path = dataset / "export-report.json"
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    export = json.loads(export_path.read_text())
    if export.get("preprocessing_version") != "sig-id-live-waterfall-v2":
        raise ValueError("this frozen baseline requires v2 spectrogram images")
    if export.get("source_snapshot_sha256"):
        snapshot = dataset / "source_snapshot" / "dataset.sqlite3"
        if not snapshot.is_file() or sha256_file(snapshot) != export["source_snapshot_sha256"]:
            raise ValueError("frozen shadow source SQLite snapshot hash changed")
    if not manifest:
        raise ValueError("empty held-out manifest")
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    captures_by_split: dict[str, set[str]] = defaultdict(set)
    identities_by_split: dict[str, set[str]] = defaultdict(set)
    selected = []
    seen_names: set[str] = set()
    for row in manifest:
        part = row["split"]
        if part not in ("train", "validation", "test"):
            raise ValueError("unknown split name")
        lineage = str(row.get("lineage_group") or "")
        source_sha = str(row.get("capture_sha256") or "")
        if len(source_sha) != 64 or len(lineage) < 16:
            raise ValueError("missing source SHA-256 or lineage group")
        groups_by_split[part].add(lineage)
        captures_by_split[part].add(source_sha)
        identities_by_split[part].add(row["capture_id"])
        if part == split:
            image = row["image"]
            if image in seen_names:
                raise ValueError("duplicate held-out image")
            seen_names.add(image)
            selected.append(row)
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if (groups_by_split[first] & groups_by_split[second]
                or captures_by_split[first] & captures_by_split[second]
                or identities_by_split[first] & identities_by_split[second]):
            raise ValueError(f"source leakage across {first}/{second}")
    if not selected:
        raise ValueError("no held-out windows")
    selected.sort(key=lambda row: (str(row["source_kind"]), str(row["lineage_group"]),
                                   str(row["capture_sha256"]), int(row["sample_start"])))
    report = {
        "split": split, "window_count": len(selected),
        "lineage_group_count": len({r["lineage_group"] for r in selected}),
        "audio_hours": sum(r["sample_count"] / r["sample_rate"] for r in selected) / 3600,
        "window_source_counts": dict(Counter(r["source_kind"] for r in selected)),
        "lineage_group_source_counts": {
            kind: len({r["lineage_group"] for r in selected if r["source_kind"] == kind})
            for kind in sorted({r["source_kind"] for r in selected})
        },
        "ground_truth_boxes_by_source": dict(
            Counter({kind: sum(r.get("box_count", 0) for r in selected if r["source_kind"] == kind)
                     for kind in {r["source_kind"] for r in selected}})
        ),
        "export_report_sha256": sha256_file(export_path),
        "windows_manifest_sha256": sha256_file(manifest_path),
        "heldout_images_sha256": hashlib.sha256(
            "\n".join(f'{r["image"]} {r["image_sha256"]} {r["label_sha256"]}'
                      for r in sorted(selected, key=lambda x: x["image"])).encode()
        ).hexdigest(),
        "all_cross_split_source_checks_passed": True,
    }
    return selected, report


def resolve_sources(rows: list[dict], source_dirs: list[Path]) -> dict[str, dict]:
    """Read-only SQLite provenance. Fail on missing or contradictory sources."""
    targets = {r["capture_id"]: r["capture_sha256"] for r in rows}
    found: dict[str, dict] = {}
    for root in source_dirs:
        db_path = root / "dataset.sqlite3"
        if not db_path.is_file():
            raise FileNotFoundError(db_path)
        db = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        try:
            for batch_start in range(0, len(targets), 400):
                ids = list(targets)[batch_start:batch_start + 400]
                query = f"SELECT id,sha256,audio_path,sample_count,sample_rate,channels,source_kind FROM captures WHERE id IN ({','.join('?' for _ in ids)})"
                for stored in db.execute(query, ids):
                    row = dict(stored)
                    expected_sha = targets[row["id"]]
                    if row["sha256"] != expected_sha:
                        raise ValueError("capture SHA mismatch across source roots")
                    existing = found.get(row["id"])
                    if existing is not None and any(existing[k] != row[k] for k in ("sha256", "sample_count", "sample_rate")):
                        raise ValueError("capture ID maps to conflicting source records")
                    found[row["id"]] = row
        finally:
            db.close()
    missing = targets.keys() - found.keys()
    if missing:
        raise ValueError(f"missing {len(missing)} held-out capture provenance records")
    return found


def read_verified_pcm(source: dict) -> np.ndarray:
    path = Path(source["audio_path"]).resolve()
    if not path.is_file() or sha256_file(path) != source["sha256"]:
        raise ValueError("held-out audio hash changed or source is missing")
    with wave.open(str(path), "rb") as reader:
        if (reader.getsampwidth() != 2 or reader.getcomptype() != "NONE"
                or reader.getframerate() != int(source["sample_rate"])
                or reader.getnchannels() != int(source["channels"])
                or reader.getnframes() != int(source["sample_count"])):
            raise ValueError("held-out audio metadata is inconsistent with source manifest")
        pcm = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    return pcm.reshape(-1, int(source["channels"])).astype(np.float64).mean(axis=1) / 32768.0


def make_proposal(
    samples: np.ndarray, rate: int, detection: dict, metadata: dict, parameters: dict,
) -> dict:
    start = int(detection["sample_start"])
    count = int(detection["sample_count"])
    center = start + count // 2
    seconds = float(metadata.get("classifier_window_seconds") or CLASSIFIER_WINDOW_SECONDS)
    target = max(1, round(rate * seconds))
    selected_start = max(0, min(len(samples) - target, center - target // 2))
    crop = samples[selected_start:selected_start + target]
    if len(crop) < target:
        raise ValueError("insufficient classifier audio")
    version = metadata["feature_version"]
    features = extract_features(
        crop, rate, feature_version=version,
        classifier_window_seconds=seconds,
        frequency_range_hz=(float(detection["frequency_low_hz"]), float(detection["frequency_high_hz"]))
        if version in ROI_FEATURE_VERSIONS else None,
    )
    result = propose_loaded(features, metadata, parameters)
    return {"available": True, **result}


def _count_box_matches(pred: np.ndarray, reference: np.ndarray, threshold: float):
    matched = _match_pairs(pred, reference, threshold)
    return matched, (len(matched), len(pred) - len(matched), len(reference) - len(matched))


def summarize_counts(counts: tuple[int, int, int], hours: float) -> dict[str, Any]:
    tp, fp, fn = counts
    return dict(tp=tp, fp=fp, fn=fn, precision=tp / max(1, tp + fp),
                recall=tp / max(1, tp + fn), false_detections_per_audio_hour=fp / max(hours, 1e-12))


def load_payload_references(path: Path) -> dict[str, list[dict]]:
    """Optional independently transcribed payloads; never detector inputs."""
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or not isinstance(payload.get("references"), list):
        raise ValueError("unsupported reviewed payload reference format")
    keyed: dict[str, list[dict]] = defaultdict(list)
    observed: set[tuple[str, int, int]] = set()
    for row in payload["references"]:
        sha = row.get("capture_sha256")
        start, count = row.get("sample_start"), row.get("sample_count")
        expected = row.get("expected_text")
        if (not isinstance(sha, str) or len(sha) != 64 or not isinstance(start, int)
                or not isinstance(count, int) or start < 0 or count <= 0
                or not isinstance(expected, str) or not expected.strip()
                or row.get("independently_reviewed") is not True):
            raise ValueError("invalid or unreviewed payload reference")
        key = (sha, start, count)
        if key in observed:
            raise ValueError("duplicate reviewed payload reference")
        observed.add(key)
        keyed[sha].append(dict(sample_start=start, sample_count=count, expected_text=expected))
    return dict(keyed)


def benchmark_reviewed_chain(
    dataset: Path, source_dirs: list[Path], detector_path: Path, *,
    classifier_path: Path | None = None, split: str = "test",
    decode_source: str = "none", max_decode_attempts: int = 0,
    detector_override: Any = None, verify_pixels: bool = True,
    payload_references: dict[str, list[dict]] | None = None,
) -> dict:
    selected, integrity = load_frozen_manifest(dataset, split)
    sources = resolve_sources(selected, source_dirs)
    if payload_references is not None:
        unknown = set(payload_references) - {row["capture_sha256"] for row in selected}
        if unknown:
            raise ValueError("payload truth references must only name held-out captures")
    detector = detector_override or YoloSignalDetector(detector_path)
    threshold = float(detector.confidence_threshold)
    classifier = load_model(classifier_path) if classifier_path else None

    counts = {iou: [0, 0, 0] for iou in (0.25, 0.50)}
    per_kind = defaultdict(lambda: [0, 0, 0])
    by_group = defaultdict(lambda: [0, 0, 0])
    by_mode = defaultdict(lambda: [0, 0])  # GT total, GT detected at IoU .50
    hours_by_kind = Counter()
    errors = Counter()
    runtimes = []
    classification = Counter()
    decodes = Counter()
    native_attempts = 0
    audio_verified = 0
    image_verified = 0
    for row in selected:
        source = sources[row["capture_id"]]
        if (source["sha256"] != row["capture_sha256"]
                or source["source_kind"] != row["source_kind"]):
            raise ValueError("held-out source mismatch")
        root = dataset.resolve()
        image_path = (dataset / row["image"]).resolve()
        if not image_path.is_relative_to(root) or image_path.suffix.lower() != ".png":
            raise ValueError("untrusted image path outside dataset")
        label_path = (dataset / "labels" / split / f"{image_path.stem}.txt").resolve()
        if not label_path.is_relative_to(root):
            raise ValueError("untrusted label path outside dataset")
        if sha256_file(image_path) != row["image_sha256"] or sha256_file(label_path) != row["label_sha256"]:
            raise ValueError("held-out image or labels changed")
        image_verified += 1
        from PIL import Image
        with Image.open(image_path) as png:
            rgb = np.asarray(png.convert("RGB"))
        # Read input only after verifying both its source hash and its
        # immutable reviewed exported image+labels. Do not use GT to tune.
        audio = read_verified_pcm(source)
        audio_verified += 1
        offset = int(row["sample_start"])
        n = int(row["sample_count"])
        if int(row["sample_rate"]) != source["sample_rate"] or offset < 0 or offset+n > len(audio):
            raise ValueError("held-out window sample geometry changed")
        selected_audio = audio[offset:offset+n]
        if verify_pixels:
            from sigid.cv_detector import spectrogram_image
            if not np.array_equal(spectrogram_image(selected_audio, int(row["sample_rate"])), rgb):
                raise ValueError("held-out exported spectrogram does not match verified source PCM")
        expected = _read_yolo_boxes(label_path)
        if len(expected) != int(row["box_count"]):
            raise ValueError("ground-truth box count does not match manifest")
        started = time.perf_counter()
        found = detector.detect(rgb, sample_start=offset, sample_count=n, sample_rate=row["sample_rate"])
        runtimes.append((time.perf_counter() - started) * 1000)
        # YOLO boxes use image coordinates: newest samples are at the TOP.
        # The public detector result also provides chronological sample
        # bounds; never compare those directly to top-down image labels.
        predicted = np.asarray([
            d.get("box_xyxy_normalized") or [
                float(d["frequency_low_hz"]) / MAX_FREQUENCY_HZ,
                1.0 - (float(d["sample_start"]) + float(d["sample_count"]) - offset) / n,
                float(d["frequency_high_hz"]) / MAX_FREQUENCY_HZ,
                1.0 - (float(d["sample_start"]) - offset) / n,
            ]
            for d in found
        ], dtype=np.float64).reshape(-1, 4)
        for iou in (0.25, 0.50):
            _, local_counts = _count_box_matches(predicted, expected, iou)
            counts[iou] = [a+b for a, b in zip(counts[iou], local_counts)]
        matched, local = _count_box_matches(predicted, expected, 0.50)
        kind = row["source_kind"]
        group_key = (kind, row["lineage_group"])
        per_kind[kind] = [a+b for a, b in zip(per_kind[kind], local)]
        by_group[group_key] = [a+b for a, b in zip(by_group[group_key], local)]
        hours_by_kind[kind] += n / row["sample_rate"] / 3600
        matched_gt = {gt for _, gt in matched}
        labels = row.get("box_types") or []
        if len(labels) != len(expected):
            raise ValueError("box_type list length does not match labels")
        for idx, label in enumerate(labels):
            by_mode[label][0] += 1
            by_mode[label][1] += int(idx in matched_gt)

        matched_pred = {pred for pred, _ in matched}
        for index, detection in enumerate(found):
            proposal = None
            if classifier is not None:
                classification["attempted"] += 1
                try:
                    model_meta, model_params = classifier
                    # Model feature time indices are relative to this 10 s
                    # sample window (not the absolute original capture).
                    local_detection = dict(detection, sample_start=int(detection["sample_start"]) - offset)
                    proposal = make_proposal(selected_audio, int(row["sample_rate"]),
                                             local_detection, model_meta, model_params)
                    classification["ranked"] += bool(proposal.get("ranked_candidates"))
                    classification["accepted"] += int(not proposal.get("abstained", True))
                    classification["matched_candidates"] += int(index in matched_pred)
                except Exception:
                    classification["error"] += 1
            if (decode_source == "none" or native_attempts >= max_decode_attempts
                    or (decode_source == "real" and kind != "live")):
                continue
            if index not in matched_pred and len(expected) > 0:
                # For mixed scenes prefer genuine localized candidates; on
                # noise-only windows, also evaluate false-positive decodes.
                continue
            # The normal live API track registry calculates center_hz.
            # Direct detector.detect() returns only low/high bounds, so
            # supply the same derived geometry before routing native modes.
            center_hz = (float(detection["frequency_low_hz"])
                         + float(detection["frequency_high_hz"])) / 2
            trial = dict(detection, center_hz=center_hz,
                         bandwidth_hz=float(detection["frequency_high_hz"])
                                      - float(detection["frequency_low_hz"]),
                         classifier_proposal=proposal,
                         track_state="persistent", track_id="benchmark:unknown")
            modes = candidate_modes(trial)[:2]
            if not modes:
                continue
            start = max(offset, int(detection["sample_start"]))
            end = min(offset+n, start + min(int(detection["sample_count"]), round(4 * row["sample_rate"])))
            if end-start < round(0.5 * row["sample_rate"]):
                decodes["too_short"] += 1
                continue
            mono = selected_audio[start-offset:end-offset]
            pcm = np.clip(np.rint(mono * 32767), -32768, 32767).astype("<i2").tobytes()
            native_attempts += 1
            try:
                candidate = decode_live_candidate(pcm, int(row["sample_rate"]), start,
                                                 center_hz, tuple(modes))
                decodes["attempted"] += 1
                grade = str(candidate.get("grade") or "none")
                decodes[f"grade_{grade}"] += 1
                decodes["attempt_on_matched"] += int(index in matched_pred)
                decodes["attempt_on_negative"] += int(not len(expected))
                decodes["cpu_ms_total"] += float(candidate.get("cpu_ms") or 0)
                if payload_references is not None:
                    for reference in payload_references.get(source["sha256"], []):
                        reference_start = reference["sample_start"]
                        reference_end = reference_start + reference["sample_count"]
                        if reference_start >= start and reference_end <= end:
                            decodes["payload_truth_eligible"] += 1
                            # Normalized exact text comparison; no substring
                            # matching that could inflate noise coincidences.
                            decodes["correct_payloads"] += int(
                                candidate.get("text", "").strip() == reference["expected_text"].strip()
                            )
            except Exception as exc:
                decodes["error"] += 1
                decodes[f"error_type_{type(exc).__name__}"] += 1

    def grouped(kind: str) -> dict:
        lineages = [value for (source, _), value in by_group.items() if source == kind]
        positive = [group for group in lineages if group[0] + group[2] > 0]
        return dict(
            windows=sum(1 for r in selected if r["source_kind"] == kind),
            lineages=len(lineages), positive_lineages=len(positive),
            ground_truth_boxes=sum(g[0]+g[2] for g in lineages),
            box_weighted=summarize_counts(tuple(per_kind[kind]), hours_by_kind[kind]),
            positive_lineage_macro_recall=(
                statistics.mean(g[0] / (g[0] + g[2]) for g in positive) if positive else None
            ),
            per_lineage_box_totals=sorted((g[0] + g[2] for g in lineages), reverse=True),
        )
    kinds = sorted(set(x["source_kind"] for x in selected))
    return {
        "schema_version": 1,
        "protocol": "frozen-heldout-review-lineage-v1",
        "data_integrity": {**integrity, "verified_audio_hashes": audio_verified,
                           "verified_image_and_label_hashes": image_verified,
                           "pixels_recomputed_identical": bool(verify_pixels)},
        "model": {
            "detector_sha256": detector.model_sha256,
            "detector_confidence_threshold": threshold,
            "classifier_sha256": classifier[0]["model_sha256"] if classifier else None,
        },
        "metrics": {
            "iou_0_25": summarize_counts(tuple(counts[0.25]), integrity["audio_hours"]),
            "iou_0_50": summarize_counts(tuple(counts[0.50]), integrity["audio_hours"]),
            "source_breakdown": {kind: grouped(kind) for kind in kinds},
            "signal_type_recall": {
                k: dict(boxes=v[0], matched=v[1], recall=v[1]/max(1,v[0]))
                for k, v in sorted(by_mode.items())
            },
            "classifier": dict(classification),
            "native_decoder": dict(
                decodes,
                max_attempts=max_decode_attempts,
                attempted=native_attempts,
                evaluation_truth=(
                    "independently_reviewed_payload_file"
                    if payload_references is not None else "no_verified_transcripts"
                ),
                correct_payloads=(
                    int(decodes["correct_payloads"])
                    if decodes.get("payload_truth_eligible", 0) > 0 else None
                ),
            ),
            "runtime_ms_per_detector_window": {
                "n": len(runtimes), "p50": float(np.percentile(runtimes, 50)),
                "p95": float(np.percentile(runtimes, 95)),
            },
        },
        "limitations": [
            "Test contains few independent real receiver lineages; report macro alongside box-weighted recall.",
            "Reviewed boxes do not include a complete independently transcribed payload dataset.",
            "Classifier candidate routing is not accepted identity and native text is not automatically correct.",
            "This evaluates fixed 10 s reviewed windows, not a continuously scheduled real-time session.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--detector", type=Path, required=True)
    ap.add_argument("--classifier", type=Path)
    ap.add_argument("--source-data-dir", type=Path, action="append", default=[],
                    help="Repeat for every source data root; defaults to the frozen export source_data_dirs")
    ap.add_argument("--split", choices=("validation", "test"), default="test")
    ap.add_argument("--native-decode", choices=("none", "real", "all"), default="none")
    ap.add_argument("--max-native-attempts", type=int, default=0)
    ap.add_argument("--payload-references", type=Path,
                    help="independently reviewed SHA256+sample-bound text reference file; never used for routing")
    ap.add_argument("--skip-pixel-verification", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.max_native_attempts < 0:
        ap.error("--max-native-attempts must be nonnegative")
    report = json.loads((args.dataset / "export-report.json").read_text())
    source_dirs = args.source_data_dir or [Path(p) for p in report.get("source_data_dirs", [])]
    if not source_dirs:
        ap.error("no source data directories supplied")
    references = load_payload_references(args.payload_references) if args.payload_references else None
    result = benchmark_reviewed_chain(
        args.dataset, source_dirs, args.detector, classifier_path=args.classifier,
        split=args.split, decode_source=args.native_decode,
        max_decode_attempts=args.max_native_attempts,
        verify_pixels=not args.skip_pixel_verification,
        payload_references=references,
    )
    result["data_integrity"]["payload_reference_sha256"] = (
        sha256_file(args.payload_references) if args.payload_references else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "heldout_windows": result["data_integrity"]["window_count"],
        "real": result["metrics"]["source_breakdown"].get("live"),
        "overall_iou_0_50": result["metrics"]["iou_0_50"],
        "native_decoder": result["metrics"]["native_decoder"],
    }, indent=2))


if __name__ == "__main__":
    main()
