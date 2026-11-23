"""Balance reviewed live *training* windows for the CV detector.

Run only on a fresh sig-id CV export. This creates hard-linked copies of
original train images and their labels, preserving every original pixel and
bounding box. It never touches validation/test and never invents new independent
lineages. Oversampling is deliberately counted as repeated examples, not as
additional reviewed captures.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def rebalance_cv_train(dataset_dir: str | Path, *, live_repetitions: int = 3) -> dict:
    dataset = Path(dataset_dir).expanduser().resolve(strict=True)
    if not 1 <= live_repetitions <= 10:
        raise ValueError("live_repetitions must be within 1..10")
    manifest = dataset / "windows.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    report_path = dataset / "training-balance.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        if existing.get("live_repetitions") != live_repetitions:
            raise ValueError("dataset already balanced with a different recipe")
        return existing
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    selected = [
        row for row in records if row.get("split") == "train"
        and row.get("source_kind") == "live"
        and row.get("box_count", 0) > 0
    ]
    for row in selected:
        source_image = dataset / row["image"]
        source_label = dataset / "labels/train" / (source_image.stem + ".txt")
        if not source_image.is_file() or not source_label.is_file():
            raise ValueError(f"missing reviewed image or label for {source_image}")
    copies = []
    for row in selected:
        source_image = dataset / row["image"]
        source_label = dataset / "labels/train" / (source_image.stem + ".txt")
        for index in range(1, live_repetitions):
            stem = source_image.stem + f"-live-repeat-{index}"
            target_image = source_image.with_name(stem + ".png")
            target_label = source_label.with_name(stem + ".txt")
            for source, target in ((source_image, target_image), (source_label, target_label)):
                if not target.exists():
                    os.link(source, target)
                elif not target.samefile(source):
                    raise ValueError(f"cannot overwrite different data at {target}")
            copies.append({
                "source_image": row["image"], "repeat_image": str(target_image.relative_to(dataset)),
                "capture_id": row["capture_id"], "lineage_group": row["lineage_group"],
            })
    report = {
        "schema_version": 1, "scope": "train-only-positive-reviewed-live",
        "live_repetitions": live_repetitions,
        "independent_new_captures": 0,
        "source_positive_live_windows": len(selected),
        "repeated_train_images": len(copies),
        "repeat_records": copies,
        "validation_and_test_unchanged": True,
        "note": "Pixel-identical training oversampling, not new data; evaluate on source-held-out live windows.",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--live-repetitions", type=int, default=3)
    args = parser.parse_args()
    report = rebalance_cv_train(args.dataset, live_repetitions=args.live_repetitions)
    print(json.dumps({k: v for k, v in report.items() if k != "repeat_records"}, indent=2))


if __name__ == "__main__":
    main()
