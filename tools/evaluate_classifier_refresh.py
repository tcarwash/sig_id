"""Compare staged hierarchical classifiers on independent source-held-out waveforms.

Reports top-k routing evidence separately from calibrated identity acceptance.
The source corpus and Fldigi test rows are never fitted during this evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sigid.classifier import load_model, propose_loaded
from sigid.dataset import CaptureStore
from sigid.hierarchical_training import _fldigi_synthetic_rows
from sigid.training import _features_for, _validated_rows


def evaluate_models(
    split_path: Path, corpus_dir: Path, fldigi_dir: Path, model_paths: list[Path],
) -> dict:
    store = CaptureStore(corpus_dir / "dataset.sqlite3")
    corpus_rows = [row for row in _validated_rows(split_path, store) if row["split"] == "test"]
    fid_rows, _ = _fldigi_synthetic_rows([fldigi_dir], known_hashes=set())
    sources = {"corpus": corpus_rows, "fldigi": [r for r in fid_rows if r["split"] == "test"]}
    result = {"sources": {k: len(v) for k, v in sources.items()}, "models": {}}
    for path in model_paths:
        meta, parameters = load_model(path)
        by_source = {}
        for source, rows in sources.items():
            if not rows:
                by_source[source] = {"total": 0, "in_taxonomy": 0}
                continue
            x, labels, _ = _features_for(
                rows, meta["feature_version"], float(meta["classifier_window_seconds"]),
            )
            available = set(meta["classes"])
            correct_top1 = correct_top3 = accepted_correct = accepted = valid = 0
            by_mode = {}
            for features, label in zip(x, labels):
                name = str(label)
                if name not in available:
                    continue
                valid += 1
                p = propose_loaded(features.reshape(1,-1), meta, parameters)
                ranked = [c["mode_id"] for c in p.get("ranked_candidates", [])]
                top1 = bool(ranked and ranked[0] == name)
                top3 = name in ranked[:3]
                is_accepted = bool(p.get("method_id"))
                if top1: correct_top1 += 1
                if top3: correct_top3 += 1
                if is_accepted:
                    accepted += 1
                    accepted_correct += int(p.get("method_id") == name)
                mode_stats = by_mode.setdefault(name, {"total": 0, "top1": 0, "top3": 0})
                mode_stats["total"] += 1
                mode_stats["top1"] += int(top1)
                mode_stats["top3"] += int(top3)
            by_source[source] = {
                "total": len(rows), "in_taxonomy": valid,
                "top1": correct_top1,
                "top3": correct_top3,
                "top1_rate": correct_top1 / valid if valid else None,
                "top3_rate": correct_top3 / valid if valid else None,
                "accepted": accepted,
                "accepted_correct": accepted_correct,
                "by_mode": by_mode,
            }
        result["models"][str(path)] = {
            "model_sha256": meta["model_sha256"],
            "calibration_status": meta.get("family_calibration", {}).get("status"),
            "scores": by_source,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--fldigi", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_models(args.split, args.corpus, args.fldigi, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    for model, m in result["models"].items():
        print(Path(model).parent.name, m["calibration_status"])
        for source, d in m["scores"].items():
            print(" ",source, "n",d["in_taxonomy"],"top1",d.get("top1_rate"),
                  "top3",d.get("top3_rate"),"accepted",d.get("accepted"))


if __name__ == "__main__":
    main()
