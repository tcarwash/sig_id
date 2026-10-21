"""Repeat lineage-safe model evaluation across independent split seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import CaptureStore
from .splits import create_split_manifest
from .training import train_and_evaluate


METRICS = (
    "coverage",
    "accepted_accuracy",
    "raw_accuracy",
    "noise_false_positive_rate",
    "unknown_family_false_positive_rate",
)


def _summary(values: list[float | None]) -> dict[str, float | int | None]:
    observed = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not observed.size:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(observed.size),
        "mean": float(observed.mean()),
        "std": float(observed.std(ddof=0)),
        "min": float(observed.min()),
        "max": float(observed.max()),
    }


def evaluate_split_seeds(
    corpus_index: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    seeds: list[int],
    epochs: int = 1200,
    ood_corpus_index: str | Path | None = None,
) -> dict[str, Any]:
    """Train on distinct source-group splits and summarize held-out stability."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("at least one unique split seed is required")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1 for seed in seeds):
        raise ValueError("split seeds must be unsigned 32-bit integers")
    corpus_path = Path(corpus_index).expanduser().resolve(strict=True)
    root = Path(output_dir).expanduser().resolve()
    store = CaptureStore(Path(data_dir).expanduser().resolve() / "dataset.sqlite3")
    records = []
    by_assignment: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        split = create_split_manifest(corpus_path, store, seed=seed, assignment_version=2)
        assignment = sorted((row["capture_id"], row["split"]) for row in split["records"])
        assignment_hash = hashlib.sha256(json.dumps(assignment, separators=(",", ":")).encode()).hexdigest()
        split_path = root / f"split-{seed}.json"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        duplicate = by_assignment.get(assignment_hash)
        if duplicate:
            record = {
                "seed": seed,
                "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
                "assignment_sha256": assignment_hash,
                "model_sha256": duplicate["model_sha256"],
                "model_file": duplicate["model_file"],
                "duplicate_of_seed": duplicate["seed"],
                "test": duplicate["test"],
                "test_by_condition": duplicate["test_by_condition"],
                "validation_by_condition": duplicate["validation_by_condition"],
                "ood_test": duplicate["ood_test"],
            }
            records.append(record)
            continue
        model_path = root / f"model-{seed}.npz"
        trained = train_and_evaluate(
            split_path,
            data_dir,
            model_path,
            epochs=epochs,
            ood_corpus_index=ood_corpus_index,
        )
        evaluation = trained["evaluation"]
        record = {
            "seed": seed,
            "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "assignment_sha256": assignment_hash,
            "model_sha256": trained["model"]["model_sha256"],
            "model_file": model_path.name,
            "test": evaluation["test"],
            "test_by_condition": evaluation.get("test_by_condition", {}),
            "validation_by_condition": evaluation.get("validation_by_condition", {}),
            "ood_test": evaluation.get("ood_test", {}).get("metrics"),
        }
        records.append(record)
        by_assignment[assignment_hash] = record

    independent_runs = [row for row in records if "duplicate_of_seed" not in row]
    summary = {metric: _summary([row["test"].get(metric) for row in independent_runs]) for metric in METRICS}
    for scope in ("method", "family"):
        key = f"{scope}_false_accept_rate"
        summary[f"ood_{key}"] = _summary([
            row["ood_test"].get(key) if row["ood_test"] is not None else None for row in independent_runs
        ])
    conditions = sorted({
        condition
        for row in independent_runs
        for condition in row["test_by_condition"]
    })
    summary["by_condition"] = {}
    for scope in ("validation", "test"):
        field = f"{scope}_by_condition"
        summary_key = f"{scope}_by_condition"
        scope_conditions = sorted({
            condition for row in independent_runs for condition in row[field]
        })
        summary[summary_key] = {
            condition: {
                metric: _summary([
                    row[field].get(condition, {}).get("metrics", {}).get(metric)
                    for row in independent_runs
                ])
                for metric in ("coverage", "accepted_accuracy", "raw_accuracy")
            }
            for condition in scope_conditions
        }
    summary["by_condition"] = summary["test_by_condition"]
    return {
        "schema_version": 1,
        "evaluation_kind": "multi-seed lineage-safe synthetic model evaluation",
        "scope": "split variability only; sources remain synthetic and do not establish field performance",
        "corpus_index_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "ood_corpus_index_sha256": (
            hashlib.sha256(Path(ood_corpus_index).expanduser().resolve(strict=True).read_bytes()).hexdigest()
            if ood_corpus_index is not None else None
        ),
        "epochs": epochs,
        "split_assignment_version": 2,
        "split_seeds": seeds,
        "requested_seed_runs": len(records),
        "unique_group_assignments": len(independent_runs),
        "duplicate_assignments_excluded_from_summary": len(records) - len(independent_runs),
        "runs": records,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat model evaluation across lineage-safe split seeds")
    parser.add_argument("--corpus", type=Path, required=True, help="known synthetic corpus index")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("SIG_ID_DATA_DIR", "~/.local/share/sig-id")).expanduser())
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for per-seed splits, models and report")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--ood-corpus", type=Path, help="separate unknown-family corpus for calibration and test")
    args = parser.parse_args()
    try:
        result = evaluate_split_seeds(
            args.corpus,
            args.data_dir,
            args.output_dir,
            seeds=args.seeds,
            epochs=args.epochs,
            ood_corpus_index=args.ood_corpus,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    output = args.output_dir / "multi-seed-evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
