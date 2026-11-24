"""Evaluate old and new CV detectors on exactly the same held-out export.

Confidence thresholds are selected on validation by their respective model
trainers; this script performs no threshold search on test windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sigid.cv_training import _evaluate_test_set


def concise(metrics: dict) -> dict:
    c = metrics["condition_breakdowns_iou_0_50"]
    loc = metrics.get("localization_at_iou_0_50", {})
    return {
        "overall": metrics["metrics"]["iou_0.50"],
        "source_kind": c.get("source_kind", {}),
        "signal_overlap": c.get("signal_overlap", {}),
        "false_detections_per_audio_hour": metrics.get("false_detections_per_audio_hour"),
        "center_frequency_error_hz_mae": loc.get("frequency_center_error_hz_mae"),
        "reviewed_test_windows": metrics.get("reviewed_test_windows"),
    }


def release_preconditions(dataset: Path, candidate: Path) -> dict[str, bool]:
    """Never approve a pilot or an evaluation with too few real lineages."""
    rows = [json.loads(line) for line in (dataset / "windows.jsonl").read_text().splitlines() if line.strip()]
    test_live = [row for row in rows if row.get("split") == "test"
                 and row.get("source_kind") == "live" and int(row.get("box_count") or 0) > 0]
    lineages = {row.get("lineage_group") for row in test_live if row.get("lineage_group")}
    metadata = json.loads(candidate.with_suffix(candidate.suffix + ".json").read_text())
    return {
        "at_least_five_independent_heldout_live_lineages": len(lineages) >= 5,
        "candidate_trained_on_full_training_split": (
            float(metadata.get("train_fraction", 1.0)) == 1.0
            and not metadata.get("pilot_partial_training", False)
        ),
    }


def compare_detectors(
    dataset: Path, baseline: Path, candidate: Path, *, device: str = "0",
) -> dict:
    from ultralytics import YOLO
    from sigid.cv_detector import YoloSignalDetector

    baseline_loader = YoloSignalDetector(str(baseline))
    candidate_loader = YoloSignalDetector(str(candidate))
    names = [("baseline", baseline, baseline_loader), ("candidate", candidate, candidate_loader)]
    results = {}
    for role, path, loader in names:
        threshold = loader.confidence_threshold
        raw = _evaluate_test_set(
            YOLO(str(path)), dataset, confidence=threshold, device=device,
            split="test", include_confidence_sweep=False,
        )
        results[role] = {
            "model": str(path), "sha256": loader.model_sha256,
            "confidence_threshold": threshold, "metrics": concise(raw),
        }
    old = results["baseline"]["metrics"]
    new = results["candidate"]["metrics"]
    old_live = old["source_kind"].get("live", {})
    new_live = new["source_kind"].get("live", {})
    old_multi = old["signal_overlap"].get("multiple", {})
    new_multi = new["signal_overlap"].get("multiple", {})
    live_boxes = int(new_live.get("true_positive", 0)) + int(new_live.get("false_negative", 0))
    multi_boxes = int(new_multi.get("true_positive", 0)) + int(new_multi.get("false_negative", 0))
    baseline_fp = float(old.get("false_detections_per_audio_hour") or 0)
    candidate_fp = float(new.get("false_detections_per_audio_hour") or 0)
    tests = {
        **release_preconditions(dataset, candidate),
        "enough_held_out_live_boxes": live_boxes >= 10,
        "live_recall_not_worse": float(new_live.get("recall") or 0) >= float(old_live.get("recall") or 0),
        "overall_recall_not_worse_by_more_than_0.03":
            float(new["overall"]["recall"]) >= float(old["overall"]["recall"]) - 0.03,
        "false_detection_rate_bounded": candidate_fp <= max(5.0, baseline_fp * 1.35),
        "multiple_signal_recall_not_worse": multi_boxes < 10 or
            float(new_multi.get("recall") or 0) >= float(old_multi.get("recall") or 0),
    }
    return {
        "schema_version": 1, "test_dataset": str(dataset),
        "baseline": results["baseline"], "candidate": results["candidate"],
        "gate_checks": tests, "candidate_eligible": all(tests.values()),
        "note": "Conservative gates, not a statistical significance test. A pilot is not release eligible; real test also requires at least five independent positive lineages.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--device", default="0")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    report = compare_detectors(args.dataset, args.baseline, args.candidate, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    for role in ("baseline", "candidate"):
        m = report[role]["metrics"]
        print(role, "overall",m["overall"],"live",m["source_kind"].get("live"),
              "multiple",m["signal_overlap"].get("multiple"),
              "fp/hour",m["false_detections_per_audio_hour"])
    print("eligible", report["candidate_eligible"], report["gate_checks"])


if __name__ == "__main__":
    main()
