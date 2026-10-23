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
from .synthetic_dataset import synthetic_frequency_bounds


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


def _splits(
    groups: set[str], seed: int, group_strata: dict[str, set[str]] | None = None,
) -> dict[str, str]:
    if len(groups) < 7:
        raise ValueError("CV export needs at least seven independent groups to reserve three each for validation and test")
    count = len(groups)
    test_count = max(3, int(round(count * 0.15)))
    val_count = max(3, int(round(count * 0.15)))
    if test_count + val_count >= count:
        raise ValueError("CV export must leave at least one independent group in training after reserving validation and test groups")
    split_counts = {"test": test_count, "validation": val_count, "train": count - test_count - val_count}
    split_weights = {name: value / count for name, value in split_counts.items()}
    buckets: dict[tuple[str, ...], list[str]] = {}
    group_strata = group_strata or {}
    for group in groups:
        buckets.setdefault(tuple(sorted(group_strata.get(group, set()))), []).append(group)

    assignment: dict[str, str] = {}
    for stratum, members in sorted(buckets.items(), key=lambda item: (item[0], len(item[1]))):
        remaining_counts = {name: int(len(members) * weight) for name, weight in split_weights.items()}
        remainder_order = sorted(
            split_weights,
            key=lambda name: (
                -(len(members) * split_weights[name] - remaining_counts[name]),
                hashlib.sha256(f"{seed}:{stratum}:{name}".encode()).hexdigest(),
            ),
        )
        for name in remainder_order[:len(members) - sum(remaining_counts.values())]:
            remaining_counts[name] += 1
        ordered = sorted(
            members,
            key=lambda value: hashlib.sha256(f"{seed}:{stratum}:{value}".encode()).hexdigest(),
        )
        offset = 0
        for name in ("test", "validation", "train"):
            next_offset = offset + remaining_counts[name]
            assignment.update({group: name for group in ordered[offset:next_offset]})
            offset = next_offset
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


def _synthetic_capture_frequency_bounds(
    store: CaptureStore, capture: dict[str, Any], seen: set[str] | None = None,
) -> tuple[float, float] | None:
    """Recover missing legacy synthetic geometry from its verified sidecar manifest."""
    capture_id = capture["id"]
    visited = set() if seen is None else seen
    if capture_id in visited or capture["source_kind"] != "synthetic":
        return None
    visited.add(capture_id)
    metadata = capture["source_metadata"]
    manifest_name = metadata.get("manifest_file")
    if not isinstance(manifest_name, str) or Path(manifest_name).name != manifest_name:
        return None
    manifest_path = Path(capture["audio_path"]).parent / manifest_name
    try:
        raw_manifest = manifest_path.read_bytes()
        if hashlib.sha256(raw_manifest).hexdigest() != metadata.get("manifest_sha256"):
            return None
        manifest = json.loads(raw_manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    inherited = None
    parent_id = capture.get("parent_capture_id")
    if manifest.get("generator") == "sig-id-awgn-augmentation" and parent_id:
        try:
            parent = store.capture(parent_id)
        except KeyError:
            parent = None
        if parent:
            parent_regions = store.regions(parent_id)
            if parent_regions and parent_regions[0]["frequency_low_hz"] is not None:
                inherited = (
                    float(parent_regions[0]["frequency_low_hz"]),
                    float(parent_regions[0]["frequency_high_hz"]),
                )
            else:
                inherited = _synthetic_capture_frequency_bounds(store, parent, visited)
    return synthetic_frequency_bounds(manifest, inherited_bounds=inherited)


def _window_boxes(
    regions: list[dict[str, Any]], start: int, count: int,
) -> tuple[list[str], list[str], str | None]:
    lines = []
    box_types = []
    end = start + count
    for region in regions:
        if region["lineage_state"] != "active" or region["review_state"] != "accepted":
            continue
        evidence = region.get("evidence") or {}
        if evidence.get("generator") == "sig-id-noise" or evidence.get("noise_only") is True:
            # The controlled noise fixture has a full-capture label for dataset
            # provenance, but it is explicitly a detector-negative example.
            continue
        region_start = region["sample_start"]
        region_end = region_start + region["sample_count"]
        if region_start >= end or region_end <= start:
            continue
        low, high = region["frequency_low_hz"], region["frequency_high_hz"]
        if low is None or high is None:
            return [], [], f"accepted region {region['id']} has no frequency bounds"
        x1, x2 = max(0.0, low), min(MAX_FREQUENCY_HZ, high)
        y1, y2 = max(start, region_start), min(end, region_end)
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / (2 * MAX_FREQUENCY_HZ), (y1 + y2 - 2 * start) / (2 * count)
        width, height = (x2 - x1) / MAX_FREQUENCY_HZ, (y2 - y1) / count
        lines.append(f"0 {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}")
        box_types.append(str(region["method"] or region["family"] or "unknown"))
    return lines, box_types, None


def _window_conditions(
    regions: list[dict[str, Any]], capture: dict[str, Any], start: int, count: int,
) -> dict[str, Any]:
    end = start + count
    active = []
    touched = []
    for region in regions:
        if region["lineage_state"] != "active" or region["review_state"] != "accepted":
            continue
        evidence = region.get("evidence") or {}
        region_start = int(region["sample_start"])
        region_end = region_start + int(region["sample_count"])
        if region_start < end and region_end > start:
            touched.append((region, evidence, region_start, region_end))
            low, high = region["frequency_low_hz"], region["frequency_high_hz"]
            overlaps_model_band = (
                low is not None and high is not None
                and max(0.0, float(low)) < min(MAX_FREQUENCY_HZ, float(high))
            )
            if (
                overlaps_model_band
                and evidence.get("generator") != "sig-id-noise"
                and evidence.get("noise_only") is not True
            ):
                active.append((region, evidence, region_start, region_end))

    labels = sorted({
        str(region["method"] or region["family"] or "unknown")
        for region, _, _, _ in active
    })
    if not labels:
        labels = ["no_signal"]
    source = capture["source_metadata"]
    generator = source.get("generator")
    parameters = source.get("parameters") or {}
    snr_db = None
    if generator == "sig-id-noise":
        noise_condition = "noise_only"
    elif generator == "sig-id-awgn-augmentation":
        snr_db = parameters.get("requested_snr_db")
        valid_snr = isinstance(snr_db, (int, float)) and not isinstance(snr_db, bool) and np.isfinite(snr_db)
        noise_condition = f"awgn_snr_{float(snr_db):g}_db" if valid_snr else "awgn"
    elif any(evidence.get("component_generator") == "sig-id-noise" for _, evidence, _, _ in touched):
        noise_condition = "noise_component"
    elif any(evidence.get("component_generator") == "sig-id-awgn-augmentation" for _, evidence, _, _ in touched):
        noise_condition = "awgn_component"
    else:
        noise_condition = "not_tagged"

    scene_drifts = [
        abs(float(evidence["drift_hz_per_second"]))
        for _, evidence, _, _ in touched
        if "drift_hz_per_second" in evidence
    ]
    truncated = sum(region_start < start or region_end > end for _, _, region_start, region_end in active)
    return {
        "source_kind": capture["source_kind"],
        "source_generator": generator,
        "signal_types": labels,
        "signal_overlap": "none" if not active else "multiple" if len(active) > 1 else "single",
        "noise_condition": noise_condition,
        "snr_db": float(snr_db) if isinstance(snr_db, (int, float)) and not isinstance(snr_db, bool) and np.isfinite(snr_db) else None,
        "drift_condition": (
            "unknown" if not scene_drifts else "drifted" if any(value > 0 for value in scene_drifts) else "no_drift"
        ),
        "maximum_absolute_drift_hz_per_second": max(scene_drifts) if scene_drifts else None,
        "boundary_truncation": "truncated" if truncated else "not_truncated",
        "truncated_signal_count": truncated,
    }


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
    regions_by_capture: dict[str, list[dict[str, Any]]] = {}
    reviews_by_capture: dict[str, list[dict[str, Any]]] = {}
    group_strata: dict[str, set[str]] = {}
    for capture in captures:
        capture_id = capture["id"]
        regions = store.regions(capture_id)
        if capture["source_kind"] == "synthetic" and any(
            region["frequency_low_hz"] is None for region in regions
            if region["review_state"] == "accepted"
        ):
            inferred_bounds = _synthetic_capture_frequency_bounds(store, capture)
            if inferred_bounds:
                for region in regions:
                    if region["frequency_low_hz"] is None:
                        region["frequency_low_hz"], region["frequency_high_hz"] = inferred_bounds
        reviews = store.scene_coverages(capture_id)
        regions_by_capture[capture_id] = regions
        reviews_by_capture[capture_id] = reviews
        group = groups[capture_id]
        for review in reviews:
            if not _review_covers_window(review, review["sample_start"], review["sample_count"]):
                continue
            boxes, _, error = _window_boxes(regions, review["sample_start"], review["sample_count"])
            if error:
                continue
            group_strata.setdefault(group, set()).add("positive" if boxes else "negative")
    split_by_group = _splits(set(groups.values()), seed, group_strata)
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
        regions = regions_by_capture[capture["id"]]
        reviews = reviews_by_capture[capture["id"]]
        sample_rate = int(capture["sample_rate"])
        full_window = sample_rate * WINDOW_SECONDS
        for start, samples, actual_rate in audio_windows(capture["audio_path"]):
            count = len(samples)
            window_samples_by_capture[capture["id"]] = window_samples_by_capture.get(capture["id"], 0) + 1
            covering = [review for review in reviews if _review_covers_window(review, start, count)]
            if not covering:
                skipped["unreviewed_window"] += 1
                continue
            labels, box_types, error = _window_boxes(regions, start, count)
            if error:
                skipped["unsupported_region"] += 1
                continue
            image = spectrogram_image(samples, actual_rate)
            group = groups[capture["id"]]
            split = split_by_group[group]
            conditions = _window_conditions(regions, capture, start, count)
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
                "box_types": box_types,
                "conditions": conditions,
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
        "schema_version": 2,
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
        "test_condition_categories": {
            dimension: sorted({
                value
                for row in metadata_rows if row["split"] == "test"
                for value in (
                    row["conditions"][dimension]
                    if isinstance(row["conditions"][dimension], list)
                    else [row["conditions"][dimension]]
                ) if value is not None
            })
            for dimension in (
                "source_kind", "source_generator", "signal_types", "signal_overlap",
                "noise_condition", "snr_db", "drift_condition", "boundary_truncation",
            )
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
