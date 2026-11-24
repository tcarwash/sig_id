"""Compare hierarchical classifiers on untouched, frozen real-session annotations.

Evaluation-only. No model fitting, calibration or threshold selection here.
Use the original frozen shadow CV windows to select real capture hashes,
not new synthetic sources and not adjacent training windows.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from sigid.classifier import load_model, propose_loaded
from sigid.dataset import CaptureStore
from sigid.training import _features_for


def compare_real(
    *, data_dir: Path, frozen_windows: Path, model_paths: list[Path],
) -> dict:
    rows = [json.loads(line) for line in frozen_windows.read_text().splitlines() if line.strip()]
    frozen = {row["capture_sha256"] for row in rows if row["split"] in ("test", "validation")}
    if not frozen:
        raise ValueError("frozen real-session source list is empty")
    store = CaptureStore(data_dir / "dataset.sqlite3")
    model_components = [load_model(path) for path in model_paths]
    common_classes = set.intersection(*(set(meta["classes"]) for meta, _ in model_components))
    by_sha = {item["sha256"]: item for item in store.list_captures()}
    if not frozen <= set(by_sha):
        raise ValueError("frozen evaluated real source was not found")
    samples = []
    ignored = Counter()
    for sha in sorted(frozen):
        capture = by_sha[sha]
        if capture["source_kind"] == "synthetic":
            raise ValueError("frozen evaluation source is synthetic")
        for region in store.regions(capture["id"]):
            mode = str(region.get("mode_id") or "").casefold()
            if (region.get("review_state") != "accepted"
                    or region.get("lineage_state") != "active"
                    or mode not in common_classes
                    or region.get("frequency_low_hz") is None
                    or region.get("frequency_high_hz") is None):
                ignored[mode or "unlabeled"] += 1
                continue
            if region["sample_count"] < capture["sample_rate"]:
                ignored[f"{mode}_short"] += 1
                continue
            samples.append({
                "mode": mode, "capture_id": capture["id"], "capture_sha256": sha,
                "audio_path": capture["audio_path"],
                "sample_rate": int(capture["sample_rate"]),
                "sample_start": int(region["sample_start"]),
                "sample_count": int(region["sample_count"]),
                "frequency_low_hz": float(region["frequency_low_hz"]),
                "frequency_high_hz": float(region["frequency_high_hz"]),
                "lineage_group": capture["source_metadata"].get("auto_capture_session_id") or sha,
            })
    if not samples:
        raise ValueError("no common mode labels in frozen real samples")
    result = {
        "frozen_capture_count": len(frozen),
        "common_model_classes": sorted(common_classes),
        "annotated_region_count": len(samples),
        "annotated_capture_count": len({sample["capture_sha256"] for sample in samples}),
        "session_count": len({sample["lineage_group"] for sample in samples}),
        "region_modes": dict(Counter(sample["mode"] for sample in samples)),
        "excluded_regions": dict(ignored),
        "models": {},
        "interpretation": "Real session-heldout examples; multiple regions/crops from the same source are correlated. No confidence threshold is tuned here.",
    }
    for path, (meta, parameters) in zip(model_paths, model_components, strict=True):
        region_results = []
        for row in samples:
            x, labels, _ = _features_for([row], meta["feature_version"], meta["classifier_window_seconds"])
            ranks = []
            accepted = 0
            for feat in x:
                prediction = propose_loaded(feat.reshape(1, -1), meta, parameters)
                ranks.append([candidate["mode_id"] for candidate in prediction.get("ranked_candidates", [])])
                accepted += bool(prediction.get("method_id"))
            n = len(ranks)
            region_results.append({
                "capture_sha256": row["capture_sha256"],
                "mode": row["mode"],
                "session": row["lineage_group"],
                "crops": n,
                "top1_rate": sum(bool(rank and rank[0] == row["mode"]) for rank in ranks) / n,
                "top3_rate": sum(row["mode"] in rank[:3] for rank in ranks) / n,
                "accepted": accepted,
            })
        crops = sum(item["crops"] for item in region_results)
        by_capture = defaultdict(list)
        for item in region_results:
            by_capture[item["capture_sha256"]].append(item["top1_rate"])
        result["models"][str(path)] = {
            "model_sha256": meta["model_sha256"],
            "calibration_status": meta["family_calibration"].get("status"),
            "evaluated_crops": crops,
            "crop_weighted_top1_accuracy": round(sum(item["top1_rate"] * item["crops"] for item in region_results) / crops, 4),
            "crop_weighted_top3_accuracy": round(sum(item["top3_rate"] * item["crops"] for item in region_results) / crops, 4),
            "region_mean_top1_accuracy": round(sum(item["top1_rate"] for item in region_results) / len(region_results), 4),
            "capture_mean_top1_accuracy": round(sum(sum(vals) / len(vals) for vals in by_capture.values()) / len(by_capture), 4),
            "accepted": sum(item["accepted"] for item in region_results),
            "modes": {
                mode: {
                    "regions": sum(item["mode"] == mode for item in region_results),
                    "top1_mean": round(np.mean([item["top1_rate"] for item in region_results if item["mode"] == mode]), 4),
                }
                for mode in sorted({item["mode"] for item in region_results})
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--frozen-windows", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_real(
        data_dir=args.data_dir, frozen_windows=args.frozen_windows, model_paths=args.model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
