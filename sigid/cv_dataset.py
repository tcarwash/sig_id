"""Build leakage-resistant YOLO datasets from reviewed signal regions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .cv_detector import (
    IMAGE_HEIGHT, IMAGE_WIDTH, MAX_FREQUENCY_HZ, PREPROCESSING_VERSION,
    WINDOW_SECONDS, audio_windows, spectrogram_image,
)
from .dataset import CaptureStore
from .synthetic_dataset import synthetic_frequency_bounds

RE_REVIEW_INTERVAL = timedelta(days=90)


def _export_progress(message: str) -> None:
    print(f"[cv-export] {message}", file=sys.stderr, flush=True)


class _CaptureStoreSet:
    """Small read-only facade over multiple dataset roots for one export."""

    def __init__(self, data_dirs: list[str | Path]) -> None:
        self.stores: list[CaptureStore] = []
        self.by_capture_id: dict[str, CaptureStore] = {}
        self.captures: list[dict[str, Any]] = []
        for data_dir in data_dirs:
            store = CaptureStore(Path(data_dir).expanduser() / "dataset.sqlite3")
            self.stores.append(store)
            for capture in store.list_captures():
                if capture["id"] in self.by_capture_id:
                    continue
                self.by_capture_id[capture["id"]] = store
                self.captures.append(capture)

    def list_captures(self) -> list[dict[str, Any]]:
        return self.captures

    def capture(self, capture_id: str) -> dict[str, Any]:
        return self.by_capture_id[capture_id].capture(capture_id)

    def regions(self, capture_id: str) -> list[dict[str, Any]]:
        return self.by_capture_id[capture_id].regions(capture_id)

    def scene_coverages(self, capture_id: str) -> list[dict[str, Any]]:
        return self.by_capture_id[capture_id].scene_coverages(capture_id)


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
    auto_capture_sessions: dict[str, str] = {}
    for item in captures:
        capture_id = item["id"]
        if item.get("parent_capture_id"):
            union(capture_id, item["parent_capture_id"])
        for related in item["source_metadata"].get("component_capture_ids", []):
            union(capture_id, related)
        auto_session = item["source_metadata"].get("auto_capture_session_id")
        if auto_session:
            if auto_session in auto_capture_sessions:
                union(capture_id, auto_capture_sessions[auto_session])
            else:
                auto_capture_sessions[auto_session] = capture_id
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
    group_sizes: dict[str, int] | None = None,
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
    group_sizes = group_sizes or {}
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
        size_by_group = {group: max(1, int(group_sizes.get(group, 1))) for group in members}
        total_size = sum(size_by_group.values())
        target_sizes = {
            name: total_size * remaining_counts[name] / len(members)
            for name in split_weights
        }
        assigned_sizes = {name: 0 for name in split_weights}
        ordered = sorted(
            members,
            key=lambda value: (
                -size_by_group[value],
                hashlib.sha256(f"{seed}:{stratum}:{value}".encode()).hexdigest(),
            ),
        )
        average_size = total_size / len(members)
        for group in ordered:
            group_size = size_by_group[group]
            candidates = [name for name in split_weights if remaining_counts[name] > 0]

            def cost(candidate: str) -> tuple[float, str]:
                score = 0.0
                for name in split_weights:
                    projected = assigned_sizes[name] + (group_size if name == candidate else 0)
                    scale = max(target_sizes[name], average_size, 1.0)
                    score += ((projected - target_sizes[name]) / scale) ** 2
                tie_breaker = hashlib.sha256(
                    f"{seed}:{stratum}:{group}:{candidate}".encode()
                ).hexdigest()
                return score, tie_breaker

            selected = min(candidates, key=cost)
            assignment[group] = selected
            assigned_sizes[selected] += group_size
            remaining_counts[selected] -= 1
    return assignment


def _ensure_reviewed_groups_in_each_split(
    assignment: dict[str, str], reviewed_groups: set[str], seed: int,
) -> dict[str, str]:
    """Keep reviewed source lineages represented across all three splits."""
    result = dict(assignment)
    targets = ("train", "validation", "test")
    for target in targets:
        if any(result[group] == target for group in reviewed_groups):
            continue
        counts = {split: sum(value == split for value in result.values()) for split in targets}
        donor = max(
            targets,
            key=lambda split: sum(result[group] == split for group in reviewed_groups),
        )
        candidates = [
            group for group in reviewed_groups
            if result[group] == donor and counts[donor] > (1 if donor == "train" else 3)
        ]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda group: hashlib.sha256(f"{seed}:{target}:{group}".encode()).hexdigest())
        result[chosen] = target
    return result


def _review_covers_window(review: dict[str, Any], start: int, count: int) -> bool:
    if review["state"] != "complete" or review["task"] != "detection" or review["is_stale"]:
        return False
    if review["sample_start"] > start or review["sample_start"] + review["sample_count"] < start + count:
        return False
    low, high = review["frequency_low_hz"], review["frequency_high_hz"]
    if low is not None and (low > 0 or high < MAX_FREQUENCY_HZ):
        return False
    try:
        reviewed_at = datetime.fromisoformat(review["created_utc"].replace("Z", "+00:00"))
        if datetime.now(timezone.utc) - reviewed_at >= RE_REVIEW_INTERVAL:
            return False
    except (KeyError, TypeError, ValueError):
        return False
    # A review with excluded or uncertain areas is not a clean training target.
    return not review["ignore_regions"] and not review["uncertain_regions"]


def _is_saved_ten_second_sample(capture: dict[str, Any]) -> bool:
    """True only for a canonical saved live-review queue window."""
    metadata = capture.get("source_metadata") or {}
    rate = int(capture.get("sample_rate") or 0)
    return (
        capture.get("source_kind") in {"live", "wav"}
        and metadata.get("auto_review_queue") is True
        and metadata.get("window_seconds") == WINDOW_SECONDS
        and metadata.get("exclude_training") is not True
        and rate > 0
        and int(capture.get("sample_count") or 0) == rate * WINDOW_SECONDS
    )


def _is_human_verified_noise_sample(
    capture: dict[str, Any], reviews: list[dict[str, Any]], regions: list[dict[str, Any]],
) -> bool:
    """Require a complete human review and no active signal annotations."""
    if not _is_saved_ten_second_sample(capture):
        return False
    count = int(capture["sample_count"])
    if not any(_review_covers_window(review, 0, count) for review in reviews):
        return False
    # Active regions mean signal content or unresolved annotation work.
    # Explicitly rejected regions are reviewed false alarms.
    return not any(
        region["lineage_state"] == "active" and region["review_state"] != "rejected"
        for region in regions
    )


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
        if evidence.get("generator") in {"sig-id-noise", "sig-id-recorded-noise"} or evidence.get("noise_only") is True:
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
        cx = (x1 + x2) / (2 * MAX_FREQUENCY_HZ)
        # Spectrogram rows run newest-at-top, like the capture waterfall.
        cy = 1.0 - (y1 + y2 - 2 * start) / (2 * count)
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
                and evidence.get("generator") not in {"sig-id-noise", "sig-id-recorded-noise"}
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
    if generator in {"sig-id-noise", "sig-id-recorded-noise"}:
        noise_condition = "noise_only"
    elif generator == "sig-id-awgn-augmentation":
        snr_db = parameters.get("requested_snr_db")
        valid_snr = isinstance(snr_db, (int, float)) and not isinstance(snr_db, bool) and np.isfinite(snr_db)
        noise_condition = f"awgn_snr_{float(snr_db):g}_db" if valid_snr else "awgn"
    elif any(evidence.get("component_generator") in {"sig-id-noise", "sig-id-recorded-noise"} for _, evidence, _, _ in touched):
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
    additional_data_dirs: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Export fresh, complete reviewed detection windows into YOLO format."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("CV dataset export requires the optional 'cv' install extra") from exc
    destination = Path(output_dir).expanduser().resolve()
    roots = list(dict.fromkeys([Path(data_dir).expanduser(), *(Path(item).expanduser() for item in (additional_data_dirs or []))]))
    started_at = time.monotonic()
    _export_progress(f"loading captures and review coverage from {len(roots)} dataset root(s)")
    store = _CaptureStoreSet(roots)
    captures = store.list_captures()
    verified_recorded_noise_policy = (
        "saved 10-second samples with fresh complete human review and no active annotations"
    )
    unverified_recorded_noise_ids = {
        capture["id"] for capture in captures
        if capture["source_metadata"].get("generator") == "sig-id-recorded-noise"
        and capture["source_metadata"].get("selection_policy") != verified_recorded_noise_policy
    }
    excluded_unverified_recorded_noise = len(unverified_recorded_noise_ids)
    if unverified_recorded_noise_ids:
        captures = [
            capture for capture in captures
            if capture["id"] not in unverified_recorded_noise_ids
            and not (set(capture["source_metadata"].get("component_capture_ids", [])) & unverified_recorded_noise_ids)
        ]
    _export_progress(f"checking reviews and annotations for {len(captures)} captures")
    groups = _lineage_groups(captures)
    regions_by_capture: dict[str, list[dict[str, Any]]] = {}
    reviews_by_capture: dict[str, list[dict[str, Any]]] = {}
    group_strata: dict[str, set[str]] = {}
    group_sizes: dict[str, int] = {}
    for capture_index, capture in enumerate(captures, start=1):
        if capture_index == 1 or capture_index % 25 == 0 or capture_index == len(captures):
            _export_progress(f"review scan: capture {capture_index}/{len(captures)}")
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
        if capture["source_metadata"].get("exclude_training") is True:
            continue
        group = groups[capture_id]
        # Split by the lineages and class strata that will actually export.
        # Counting full windows (rather than review records) prevents unused or
        # partial reviews from skewing both the split quotas and the weights.
        sample_rate = int(capture["sample_rate"])
        sample_count = int(capture["sample_count"])
        window_count = sample_rate * WINDOW_SECONDS
        for start in range(0, sample_count - window_count + 1, window_count):
            covering = [
                review for review in reviews
                if _review_covers_window(review, start, window_count)
            ]
            if not covering:
                continue
            boxes, _, error = _window_boxes(regions, start, window_count)
            if error or (
                not boxes and capture["source_kind"] != "synthetic"
                and not _is_human_verified_noise_sample(capture, reviews, regions)
            ):
                continue
            group_strata.setdefault(group, set()).add("positive" if boxes else "negative")
            group_sizes[group] = group_sizes.get(group, 0) + 1
    # Unreviewed and partial captures cannot receive validation/test quota.
    split_by_group = _splits(set(group_strata), seed, group_strata, group_sizes)
    split_by_group = _ensure_reviewed_groups_in_each_split(
        split_by_group, set(group_strata), seed,
    )
    _export_progress(
        "review scan complete: "
        f"{len(group_strata)} groups with usable coverage across "
        f"{len(split_by_group)} independent lineages"
    )
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"output directory must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        (destination / "images" / split).mkdir(parents=True, exist_ok=True)
        (destination / "labels" / split).mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, Any]] = []
    skipped = {"incomplete_window": 0, "unreviewed_window": 0, "unsupported_region": 0}
    excluded_noise_source_windows = 0
    window_samples_by_capture: dict[str, int] = {}
    windows_seen = 0
    last_window_progress = time.monotonic()
    for capture_index, capture in enumerate(captures, start=1):
        _export_progress(f"exporting capture {capture_index}/{len(captures)}: {capture['id']}")
        regions = regions_by_capture[capture["id"]]
        reviews = reviews_by_capture[capture["id"]]
        # Keep borrowed real noise recordings for provenance and lineage
        # grouping, but never export their original audio as a training image.
        if capture["source_metadata"].get("exclude_training") is True:
            for _ in audio_windows(capture["audio_path"]):
                excluded_noise_source_windows += 1
                windows_seen += 1
                now = time.monotonic()
                if windows_seen % 100 == 0 or now - last_window_progress >= 3.0:
                    _export_progress(
                        f"processed {windows_seen} audio windows; exported {len(metadata_rows)} images"
                    )
                    last_window_progress = now
            continue
        sample_rate = int(capture["sample_rate"])
        full_window = sample_rate * WINDOW_SECONDS
        for start, samples, actual_rate in audio_windows(capture["audio_path"]):
            windows_seen += 1
            now = time.monotonic()
            if windows_seen % 100 == 0 or now - last_window_progress >= 3.0:
                _export_progress(
                    f"processed {windows_seen} audio windows; exported {len(metadata_rows)} images"
                )
                last_window_progress = now
            count = len(samples)
            if actual_rate != sample_rate or count != full_window:
                # Training images must all represent the same complete ten-second
                # span. A short end-of-file remainder is not a training example.
                skipped["incomplete_window"] += 1
                continue
            window_samples_by_capture[capture["id"]] = window_samples_by_capture.get(capture["id"], 0) + 1
            covering = [review for review in reviews if _review_covers_window(review, start, count)]
            if not covering:
                skipped["unreviewed_window"] += 1
                continue
            labels, box_types, error = _window_boxes(regions, start, count)
            if error:
                skipped["unsupported_region"] += 1
                continue
            if not labels and capture["source_kind"] != "synthetic" and not _is_human_verified_noise_sample(
                capture, reviews, regions,
            ):
                skipped["unreviewed_window"] += 1
                continue
            end = start + count
            modulation_labels = [
                {
                    "region_id": region["id"],
                    "mode_id": region.get("mode_id"),
                    "family": region["family"],
                    "method": region["method"],
                    "protocol": region["protocol"],
                    "variant": region["variant"],
                    "sample_start": max(start, region["sample_start"]),
                    "sample_count": min(end, region["sample_start"] + region["sample_count"])
                    - max(start, region["sample_start"]),
                    "frequency_low_hz": region["frequency_low_hz"],
                    "frequency_high_hz": region["frequency_high_hz"],
                }
                for region in regions
                if region["lineage_state"] == "active"
                and region["review_state"] == "accepted"
                and region["sample_start"] < end
                and region["sample_start"] + region["sample_count"] > start
                and not (region.get("evidence") or {}).get("noise_only")
            ]
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
                "source_kind": capture["source_kind"],
                "source_generator": capture["source_metadata"].get("generator"),
                "lineage_group": group,
                "split": split,
                "sample_start": start,
                "sample_count": count,
                "sample_rate": actual_rate,
                "window_sample_count": full_window,
                "preprocessing_version": PREPROCESSING_VERSION,
                "review_ids": [review["coverage_id"] for review in covering],
                "box_count": len(labels),
                "box_types": box_types,
                "modulation_labels": modulation_labels,
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
        "preprocessing_version": PREPROCESSING_VERSION,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "window_seconds": WINDOW_SECONDS,
        "max_frequency_hz": MAX_FREQUENCY_HZ,
        "seed": seed,
        "group_count": len(split_by_group),
        "split_groups": {
            split: len({row["lineage_group"] for row in metadata_rows if row["split"] == split})
            for split in ("train", "validation", "test")
        },
        "assigned_split_groups": {split: sum(value == split for value in split_by_group.values()) for split in ("train", "validation", "test")},
        "source_data_dirs": [str(root.resolve()) for root in roots],
        "split_images": {split: sum(row["split"] == split for row in metadata_rows) for split in ("train", "validation", "test")},
        "images_by_source_kind": {
            kind: sum(row["source_kind"] == kind for row in metadata_rows)
            for kind in sorted({row["source_kind"] for row in metadata_rows})
        },
        "images_by_generator": {
            generator: sum(row["source_generator"] == generator for row in metadata_rows)
            for generator in sorted({row["source_generator"] for row in metadata_rows if row["source_generator"]})
        },
        "negative_images_by_source_kind": {
            kind: sum(row["box_count"] == 0 and row["source_kind"] == kind for row in metadata_rows)
            for kind in sorted({row["source_kind"] for row in metadata_rows})
        },
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
        "excluded_noise_source_windows": excluded_noise_source_windows,
        "excluded_unverified_recorded_noise_captures": excluded_unverified_recorded_noise,
        "captures_with_windows": len(window_samples_by_capture),
    }
    (destination / "export-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _export_progress(
        f"complete: {len(metadata_rows)} spectrograms exported "
        f"(train {report['split_images']['train']}, validation {report['split_images']['validation']}, "
        f"test {report['split_images']['test']}); {time.monotonic() - started_at:.1f}s elapsed"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("~/.local/share/sig-id").expanduser())
    parser.add_argument("--also-data-dir", type=Path, action="append", default=[],
                        help="include another dataset root; may be repeated")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(export_cv_dataset(args.data_dir, args.output, seed=args.seed,
                                       additional_data_dirs=args.also_data_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
