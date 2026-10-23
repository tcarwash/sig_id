"""Evaluate class-agnostic YOLO proposals through the existing classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .classifier import extract_features, load_model, propose
from .cv_dataset import _lineage_groups
from .cv_detector import MAX_FREQUENCY_HZ, YoloSignalDetector, audio_windows
from .cv_training import _match_pairs, _read_yolo_boxes
from .detection import detect_tracks
from .dataset import CaptureStore
from .spectrum import waterfall_samples


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_cv_classifier_flow(
    dataset_dir: str | Path,
    data_dir: str | Path,
    *,
    detector_path: str | Path,
    classifier_path: str | Path,
    confidence: float = 0.25,
    match_iou: float = 0.50,
) -> dict[str, Any]:
    """Measure detector proposals and classifier outcomes on reviewed test windows.

    Classification uses the same time-cropped, frequency-unfiltered audio path as
    the workbench's existing classifier endpoint. No decoder is selected or run.
    """
    if not 0.0 <= confidence <= 1.0 or not 0.0 < match_iou <= 1.0:
        raise ValueError("confidence and match_iou must be between 0 and 1")
    dataset = Path(dataset_dir).expanduser().resolve()
    root = Path(data_dir).expanduser()
    report_path = dataset / "export-report.json"
    windows_path = dataset / "windows.jsonl"
    report_bytes = report_path.read_bytes()
    windows_bytes = windows_path.read_bytes()
    report = json.loads(report_bytes)
    if report.get("schema_version") not in {1, 2}:
        raise ValueError("unsupported CV dataset report schema")
    manifest = [
        json.loads(line)
        for line in windows_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    test_by_capture: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for window in manifest:
        if window.get("split") == "test":
            sample_start = int(window["sample_start"])
            if sample_start in test_by_capture[window["capture_id"]]:
                raise ValueError("CV test manifest contains duplicate capture/sample windows")
            test_by_capture[window["capture_id"]][sample_start] = window
    if not test_by_capture:
        raise ValueError("dataset has no held-out test windows")

    store = CaptureStore(root / "dataset.sqlite3")
    captures = store.list_captures()
    lineage = _lineage_groups(captures)
    detector = YoloSignalDetector(detector_path)
    classifier_metadata, classifier_parameters = load_model(classifier_path)

    detection_counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    heuristic_counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    classifier_counts = {
        "matched_known_targets": 0,
        "matched_known_covered": 0,
        "matched_known_correct": 0,
        "matched_known_abstained": 0,
        "matched_unknown_targets": 0,
        "matched_unknown_covered": 0,
        "matched_unknown_abstained": 0,
        "unmatched_proposals": 0,
        "unmatched_classifier_evaluable": 0,
        "unmatched_accepted_known": 0,
        "too_short_proposals": 0,
        "unsupported_sample_rate": 0,
    }
    per_signal_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"matched": 0, "covered": 0, "correct": 0, "abstained": 0}
    )
    windows_seen = 0
    missing_test_windows = []

    for capture_id, windows in test_by_capture.items():
        capture = store.capture(capture_id)
        if capture["sha256"] != next(iter(windows.values()))["capture_sha256"]:
            raise ValueError(f"capture {capture_id} no longer matches the exported test data")
        if _sha256_file(Path(capture["audio_path"])) != capture["sha256"]:
            raise ValueError(f"capture {capture_id} audio no longer matches its stored hash")
        if any(row["capture_sha256"] != capture["sha256"] for row in windows.values()):
            raise ValueError(f"test windows for capture {capture_id} contain inconsistent source hashes")
        remaining = set(windows)
        for sample_start, samples, sample_rate in audio_windows(capture["audio_path"]):
            window = windows.get(sample_start)
            if window is None:
                continue
            remaining.discard(sample_start)
            if len(samples) != int(window["sample_count"]) or sample_rate != int(window["sample_rate"]):
                raise ValueError(f"test window geometry changed for capture {capture_id} at sample {sample_start}")
            image_path = (dataset / window["image"]).resolve()
            if not image_path.is_relative_to(dataset):
                raise ValueError("CV test manifest image path escapes the dataset directory")
            label_path = dataset / "labels" / "test" / f"{image_path.stem}.txt"
            if window.get("image_sha256") and _sha256_file(image_path) != window["image_sha256"]:
                raise ValueError(f"CV test image hash mismatch: {image_path.name}")
            if window.get("label_sha256") and _sha256_file(label_path) != window["label_sha256"]:
                raise ValueError(f"CV test label hash mismatch: {label_path.name}")
            expected_boxes = _read_yolo_boxes(label_path)
            from PIL import Image

            with Image.open(image_path) as opened:
                image = np.asarray(opened.convert("RGB"))
            heuristic_spectrum = waterfall_samples(
                samples,
                sample_rate,
                fft_size=4096,
                hop_size=512,
                max_rows=512,
                sample_offset=sample_start,
                total_sample_count=int(capture["sample_count"]),
            )
            heuristic_tracks = detect_tracks(heuristic_spectrum)
            heuristic_boxes = []
            window_end = sample_start + len(samples)
            for track in heuristic_tracks:
                start = max(sample_start, int(track["sample_start"]))
                end = min(window_end, int(track["sample_start"]) + int(track["sample_count"]))
                low = max(0.0, float(track["frequency_low_hz"]))
                high = min(MAX_FREQUENCY_HZ, float(track["frequency_high_hz"]))
                if end <= start or high <= low:
                    continue
                heuristic_boxes.append([
                    low / MAX_FREQUENCY_HZ,
                    (start - sample_start) / len(samples),
                    high / MAX_FREQUENCY_HZ,
                    (end - sample_start) / len(samples),
                ])
            heuristic_boxes_array = np.asarray(heuristic_boxes, dtype=np.float64).reshape((-1, 4))
            heuristic_matches = _match_pairs(heuristic_boxes_array, expected_boxes, match_iou)
            heuristic_counts["true_positive"] += len(heuristic_matches)
            heuristic_counts["false_positive"] += len(heuristic_boxes_array) - len(heuristic_matches)
            heuristic_counts["false_negative"] += len(expected_boxes) - len(heuristic_matches)
            detections = detector.detect(
                image,
                sample_start=sample_start,
                sample_count=len(samples),
                sample_rate=sample_rate,
                confidence=confidence,
            )
            predicted_boxes = np.asarray([
                [
                    item["frequency_low_hz"] / MAX_FREQUENCY_HZ,
                    (item["sample_start"] - sample_start) / len(samples),
                    item["frequency_high_hz"] / MAX_FREQUENCY_HZ,
                    (item["sample_start"] + item["sample_count"] - sample_start) / len(samples),
                ]
                for item in detections
            ], dtype=np.float64).reshape((-1, 4))
            matches = _match_pairs(predicted_boxes, expected_boxes, match_iou)
            target_for_prediction = {prediction: target for prediction, target in matches}
            detection_counts["true_positive"] += len(matches)
            detection_counts["false_positive"] += len(detections) - len(matches)
            detection_counts["false_negative"] += len(expected_boxes) - len(matches)
            box_types = window.get("box_types", [])
            if len(box_types) != len(expected_boxes):
                box_types = ["unknown"] * len(expected_boxes)

            for prediction_index, detection_item in enumerate(detections):
                local_start = max(0, detection_item["sample_start"] - sample_start)
                local_end = min(len(samples), local_start + detection_item["sample_count"])
                cropped_audio = samples[local_start:local_end]
                matched_target = target_for_prediction.get(prediction_index)
                if matched_target is None:
                    classifier_counts["unmatched_proposals"] += 1
                    expected_type = None
                    per_type_counts = None
                else:
                    expected_type = str(box_types[matched_target] or "unknown")
                    if expected_type.lower() in {"unknown", "no_signal"}:
                        classifier_counts["matched_unknown_targets"] += 1
                        per_type_counts = None
                    else:
                        classifier_counts["matched_known_targets"] += 1
                        per_type_counts = per_signal_type[expected_type]
                        per_type_counts["matched"] += 1
                if len(cropped_audio) < 1024:
                    classifier_counts["too_short_proposals"] += 1
                    if matched_target is not None:
                        if per_type_counts is None:
                            classifier_counts["matched_unknown_abstained"] += 1
                        else:
                            per_type_counts["abstained"] += 1
                            classifier_counts["matched_known_abstained"] += 1
                    continue
                if sample_rate < 8000:
                    classifier_counts["unsupported_sample_rate"] += 1
                    if matched_target is not None:
                        if per_type_counts is None:
                            classifier_counts["matched_unknown_abstained"] += 1
                        else:
                            per_type_counts["abstained"] += 1
                            classifier_counts["matched_known_abstained"] += 1
                    continue
                cropped_audio = cropped_audio[:1_000_000]
                features = extract_features(
                    cropped_audio,
                    sample_rate,
                    feature_version=classifier_metadata["feature_version"],
                )
                proposal = propose(
                    features,
                    classifier_parameters,
                    classifier_metadata["classes"],
                    threshold=float(classifier_metadata["abstention_threshold"]),
                    family_threshold=float(classifier_metadata["family_abstention_threshold"]),
                    novelty_threshold=classifier_metadata.get("novelty_distance_threshold"),
                )
                covered = bool(proposal.get("method") or proposal.get("family"))
                if matched_target is None:
                    classifier_counts["unmatched_classifier_evaluable"] += 1
                    if covered:
                        classifier_counts["unmatched_accepted_known"] += 1
                    continue

                if per_type_counts is None:
                    classifier_counts["matched_unknown_covered"] += int(covered)
                    classifier_counts["matched_unknown_abstained"] += int(not covered)
                    continue

                if covered:
                    per_type_counts["covered"] += 1
                    classifier_counts["matched_known_covered"] += 1
                    correct = proposal.get("method") == expected_type or proposal.get("family") == expected_type
                    per_type_counts["correct"] += int(correct)
                    classifier_counts["matched_known_correct"] += int(correct)
                else:
                    per_type_counts["abstained"] += 1
                    classifier_counts["matched_known_abstained"] += 1
            windows_seen += 1
        missing_test_windows.extend({"capture_id": capture_id, "sample_start": start} for start in sorted(remaining))

    if missing_test_windows:
        raise ValueError(f"{len(missing_test_windows)} held-out windows no longer match their source captures")

    tp = detection_counts["true_positive"]
    heuristic_tp = heuristic_counts["true_positive"]
    known_targets = classifier_counts["matched_known_targets"]
    covered = classifier_counts["matched_known_covered"]
    return {
        "schema_version": 1,
        "detector_sha256": detector.model_sha256,
        "classifier_sha256": classifier_metadata["model_sha256"],
        "dataset_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "windows_manifest_sha256": hashlib.sha256(windows_bytes).hexdigest(),
        "confidence_threshold": confidence,
        "match_iou": match_iou,
        "test_windows_evaluated": windows_seen,
        "missing_test_windows": missing_test_windows,
        "lineage_group_count": len({lineage[capture_id] for capture_id in test_by_capture}),
        "detection": {
            **detection_counts,
            "precision": tp / max(1, tp + detection_counts["false_positive"]),
            "recall": tp / max(1, tp + detection_counts["false_negative"]),
        },
        "heuristic_baseline": {
            **heuristic_counts,
            "precision": heuristic_tp / max(1, heuristic_tp + heuristic_counts["false_positive"]),
            "recall": heuristic_tp / max(1, heuristic_tp + heuristic_counts["false_negative"]),
            "method": "existing spectral peak-clustering track detector with default settings",
        },
        "classifier": {
            "counts": classifier_counts,
            "matched_known_targets": {
                "coverage": covered / max(1, known_targets),
                "accuracy_when_covered": classifier_counts["matched_known_correct"] / max(1, covered),
                "accuracy_including_abstentions": classifier_counts["matched_known_correct"] / max(1, known_targets),
            },
            "matched_unknown_targets": {
                "known_accept_rate": classifier_counts["matched_unknown_covered"]
                / max(1, classifier_counts["matched_unknown_targets"]),
            },
            "unmatched_proposals": {
                "known_accept_rate": classifier_counts["unmatched_accepted_known"]
                / max(1, classifier_counts["unmatched_classifier_evaluable"]),
            },
            "accuracy_by_signal_type": {
                label: {
                    **counts,
                    "coverage": counts["covered"] / max(1, counts["matched"]),
                    "accuracy_when_covered": counts["correct"] / max(1, counts["covered"]),
                    "accuracy_including_abstentions": counts["correct"] / max(1, counts["matched"]),
                }
                for label, counts in sorted(per_signal_type.items())
            },
        },
        "decoder_evaluation": {
            "status": "not_run",
            "reason": "Reviewed CV boxes do not yet carry decoder parameters or reference payloads for a reproducible held-out decode score.",
        },
        "interpretation": "Detector matches and classifier results are held-out proposals; they do not establish protocol identity or decoder interoperability.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="CV export dataset with capture-held-out test windows")
    parser.add_argument("--data-dir", type=Path, default=Path("~/.local/share/sig-id").expanduser())
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_cv_classifier_flow(
        args.dataset, args.data_dir, detector_path=args.detector,
        classifier_path=args.classifier, confidence=args.confidence, match_iou=args.match_iou,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = args.output.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
