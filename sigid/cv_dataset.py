"""Build leakage-resistant YOLO datasets from reviewed signal regions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .cv_detector import IMAGE_HEIGHT, IMAGE_WIDTH, MAX_FREQUENCY_HZ, WINDOW_SECONDS, audio_windows, spectrogram_image
from .dataset import CaptureStore


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lineage_groups(captures: list[dict[str, Any]]) -> dict[str, str]:
    by_id = {item["id"]: item for item in captures}
    parent = {capture_id: capture_id for capture_id in by_id}

    def find(capture_id: str) -> str:
        while parent[capture_id] != capture_id:
            parent[capture_id] = parent[parent[capture_id]]
            capture_id = parent[capture_id]
        return capture_id

    def union(left: str, right: str) -> None:
        if left in parent and right in parent:
            a, b = find(left), find(right)
            if a != b:
                parent[max(a, b)] = min(a, b)

    by_hash: dict[str, str] = {}
    for item in captures:
        capture_id = item["id"]
        if item.get("parent_capture_id"):
            union(capture_id, item["parent_capture_id"])
        for related in item["source_metadata"].get("component_capture_ids", []):
            union(capture_id, related)
        digest = item["sha256"]
        if digest in by_hash:
            union(capture_id, by_hash[digest])
        else:
            by_hash[digest] = capture_id

    members: dict[str, list[str]] = {}
    for capture_id in by_id:
        members.setdefault(find(capture_id), []).append(capture_id)
    stable_ids = {
        root: hashlib.sha256("|".join(sorted(by_id[item]["sha256"] for item in ids)).encode()).hexdigest()
        for root, ids in members.items()
    }
    return {capture_id: stable_ids[find(capture_id)] for capture_id in by_id}


def _splits(groups: set[str], seed: int) -> dict[str, str]:
    if len(groups) < 3:
        raise ValueError("CV export needs at least three independent capture/lineage groups")
    ordered = sorted(groups, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    count = len(ordered)
    test_count = max(1, int(round(count * 0.15)))
    val_count = max(1, int(round(count * 0.15)))
    if test_count + val_count >= count:
        test_count = val_count = 1
    assignment = {group: "test" for group in ordered[:test_count]}
    assignment.update({group: "validation" for group in ordered[test_count:test_count + val_count]})
    assignment.update({group: "train" for group in ordered[test_count + val_count:]})
    return assignment


def _review_covers_window(review: dict[str, Any], start: int, count: int) -> bool:
    if review["state"] != "complete" or review["task"] != "detection" or review["is_stale"]:
        return False
    if review["sample_start"] > start or review["sample_start"] + review["sample_count"] < start + count:
        return False
    low, high = review["frequency_low_hz"], review["frequency_high_hz"]
    if low is not None and (low > 0 or high < MAX_FREQUENCY_HZ):
        return False
    # A review with excluded or uncertain areas is not a clean training target.
    return not review["ignore_regions"] and not review["uncertain_regions"]


def _window_boxes(regions: list[dict[str, Any]], start: int, count: int) -> tuple[list[str], str | None]:
    lines = []
    end = start + count
    for region in regions:
        if region["lineage_state"] != "active" or region["review_state"] != "accepted":
            continue
        region_start = region["sample_start"]
        region_end = region_start + region["sample_count"]
        if region_start >= end or region_end <= start:
            continue
        low, high = region["frequency_low_hz"], region["frequency_high_hz"]
        if low is None or high is None:
            return [], f"accepted region {region['id']} has no frequency bounds"
        x1, x2 = max(0.0, low), min(MAX_FREQUENCY_HZ, high)
        y1, y2 = max(start, region_start), min(end, region_end)
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / (2 * MAX_FREQUENCY_HZ), (y1 + y2 - 2 * start) / (2 * count)
        width, height = (x2 - x1) / MAX_FREQUENCY_HZ, (y2 - y1) / count
        lines.append(f"0 {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}")
    return lines, None


def export_cv_dataset(
    data_dir: str | Path, output_dir: str | Path, *, seed: int = 1,
) -> dict[str, Any]:
    """Export fresh, complete reviewed detection windows into YOLO format."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("CV dataset export requires the optional 'cv' install extra") from exc
    root = Path(data_dir).expanduser()
    destination = Path(output_dir).expanduser().resolve()
    store = CaptureStore(root / "dataset.sqlite3")
    captures = store.list_captures()
    groups = _lineage_groups(captures)
    split_by_group = _splits(set(groups.values()), seed)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"output directory must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        (destination / "images" / split).mkdir(parents=True, exist_ok=True)
        (destination / "labels" / split).mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, Any]] = []
    skipped = {"unreviewed_window": 0, "unsupported_region": 0}
    window_samples_by_capture: dict[str, int] = {}
    for capture in captures:
        regions = store.regions(capture["id"])
        reviews = store.scene_coverages(capture["id"])
        sample_rate = int(capture["sample_rate"])
        full_window = sample_rate * WINDOW_SECONDS
        for start, samples, actual_rate in audio_windows(capture["audio_path"]):
            count = len(samples)
            window_samples_by_capture[capture["id"]] = window_samples_by_capture.get(capture["id"], 0) + 1
            covering = [review for review in reviews if _review_covers_window(review, start, count)]
            if not covering:
                skipped["unreviewed_window"] += 1
                continue
            labels, error = _window_boxes(regions, start, count)
            if error:
                skipped["unsupported_region"] += 1
                continue
            image = spectrogram_image(samples, actual_rate)
            group = groups[capture["id"]]
            split = split_by_group[group]
            stem = f"{capture['id']}-{start:012d}"
            image_path = destination / "images" / split / f"{stem}.png"
            label_path = destination / "labels" / split / f"{stem}.txt"
            Image.fromarray(image, mode="RGB").save(image_path, optimize=True)
            label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            metadata_rows.append({
                "image": str(image_path.relative_to(destination)),
                "capture_id": capture["id"],
                "capture_sha256": capture["sha256"],
                "lineage_group": group,
                "split": split,
                "sample_start": start,
                "sample_count": count,
                "sample_rate": actual_rate,
                "window_sample_count": full_window,
                "preprocessing_version": "sig-id-cv-gray-v1",
                "review_ids": [review["coverage_id"] for review in covering],
                "box_count": len(labels),
                "image_sha256": _sha256(image_path),
                "label_sha256": _sha256(label_path),
            })

    if not metadata_rows:
        shutil.rmtree(destination)
        raise ValueError("no windows have fresh, complete detection reviews and usable annotations")
    data_yaml = {
        "path": str(destination),
        "train": "images/train",
        "val": "images/validation",
        "test": "images/test",
        "nc": 1,
        "names": ["signal"],
    }
    (destination / "data.yaml").write_text(
        "\n".join([f"path: {destination}", "train: images/train", "val: images/validation",
                  "test: images/test", "nc: 1", "names: [signal]", ""]), encoding="utf-8",
    )
    with (destination / "windows.jsonl").open("w", encoding="utf-8") as stream:
        for row in metadata_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "schema_version": 1,
        "preprocessing_version": "sig-id-cv-gray-v1",
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "window_seconds": WINDOW_SECONDS,
        "max_frequency_hz": MAX_FREQUENCY_HZ,
        "seed": seed,
        "group_count": len(split_by_group),
        "split_groups": {split: sum(value == split for value in split_by_group.values()) for split in ("train", "validation", "test")},
        "split_images": {split: sum(row["split"] == split for row in metadata_rows) for split in ("train", "validation", "test")},
        "positive_images": sum(row["box_count"] > 0 for row in metadata_rows),
        "negative_images": sum(row["box_count"] == 0 for row in metadata_rows),
        "positive_images_by_split": {
            split: sum(row["split"] == split and row["box_count"] > 0 for row in metadata_rows)
            for split in ("train", "validation", "test")
        },
        "negative_images_by_split": {
            split: sum(row["split"] == split and row["box_count"] == 0 for row in metadata_rows)
            for split in ("train", "validation", "test")
        },
        "skipped_windows": skipped,
        "captures_with_windows": len(window_samples_by_capture),
    }
    (destination / "export-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("~/.local/share/sig-id").expanduser())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(export_cv_dataset(args.data_dir, args.output, seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
