"""Validate a v4 train-only synthetic CV supplement before handoff to Wave B.

Hard-fail on leakage, accidental validation/test images, invalid lineage,
missing image/label bytes, or stale/mismatched v4 scene provenance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from sigid.dataset import CaptureStore


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_train_only_release(
    source_data_dir: str | Path,
    scene_index: str | Path,
    train_export_dir: str | Path,
    exclusions_file: str | Path,
) -> dict[str, Any]:
    source_root = Path(source_data_dir).resolve(strict=True)
    index_file = Path(scene_index).resolve(strict=True)
    export_root = Path(train_export_dir).resolve(strict=True)
    exclusion_path = Path(exclusions_file).resolve(strict=True)
    store = CaptureStore(source_root / "dataset.sqlite3")
    index = json.loads(index_file.read_text())
    exclusions = json.loads(exclusion_path.read_text())
    excluded = set(exclusions["excluded_capture_sha256"])
    heldout = set(exclusions["frozen_holdout_capture_sha256"])
    if not excluded or not heldout or not heldout <= excluded:
        raise ValueError("frozen real holdout exclusion list is absent or invalid")
    if index["config"].get("scene_recipe_version") != 4:
        raise ValueError("scene index isn't version 4")
    if not index["config"].get("holdout_noise_exclusion_applied"):
        raise ValueError("scene index did not apply heldout-noise SHA exclusion")
    index_exclusion_hash = hashlib.sha256("|".join(sorted(excluded)).encode()).hexdigest()
    if index["config"].get("excluded_noise_source_sha256") != index_exclusion_hash:
        raise ValueError("frozen heldout exclusion set changed since scene generation")

    report = json.loads((export_root / "export-report.json").read_text())
    if report.get("train_only") is not True:
        raise ValueError("CV export was not created with --train-only")
    if report.get("scene_index_sha256") != _digest(index_file):
        raise ValueError("scene index changed since CV image export")
    if report.get("split_images", {}).get("test", -1) != 0 or report.get("split_images", {}).get("validation", -1) != 0:
        raise ValueError("train-only supplement must contain zero validation/test images")

    rows = [json.loads(line) for line in (export_root / "windows.jsonl").read_text().splitlines() if line.strip()]
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("CV export has non-training windows or is empty")
    if any(row.get("source_kind") != "synthetic" for row in rows):
        raise ValueError("train-only supplement unexpectedly includes non-synthetic source audio")
    if len(rows) != report["split_images"]["train"]:
        raise ValueError("CV export image count disagrees with manifest")
    scene_by_id = {}
    noise_original_shas: set[str] = set()
    signal_source_shas: set[str] = set()
    repeated_sources: set[str] = set()
    snr_bins: Counter[str] = Counter()
    noise_backgrounds: Counter[str] = Counter()
    for record in index["records"]:
        cid = record["scene_capture_id"]
        if cid in scene_by_id:
            raise ValueError("scene index contains duplicate capture IDs")
        scene_by_id[cid] = record
        scene = store.capture(cid)
        if scene["sha256"] != record["scene_sha256"]:
            raise ValueError("scene index does not match scene audio hash")
        component_ids = scene["source_metadata"].get("signal_component_capture_ids")
        if not component_ids or set(component_ids) != set(record["source_capture_ids"]):
            raise ValueError("scene component IDs do not agree with scene index")
        component_shas = {store.capture(component_id)["sha256"] for component_id in component_ids}
        if component_shas != set(record["source_sha256"]):
            raise ValueError("scene signal SHA lineage differs from source index")
        repeated_sources.update(signal_source_shas & component_shas)
        signal_source_shas.update(component_shas)
        if component_shas & excluded or component_shas & heldout:
            raise ValueError("heldout original recording unexpectedly included as signal")
        noise = store.capture(record["noise_capture_id"])
        if noise["source_metadata"].get("generator") == "sig-id-recorded-noise":
            borrowed = {item["sha256"] for item in noise["source_metadata"]["sources"]}
            if borrowed & excluded:
                raise ValueError("frozen-holdout original noise recording leaked into training")
            noise_original_shas.update(borrowed)
        noise_backgrounds[record["noise_source"]] += 1
        for component in record["signal_snr_measurements"]:
            snr_bins[component["snr_profile_bin"]] += 1

    if repeated_sources:
        raise ValueError("synthetic source reused in more than one scene")
    if noise_original_shas & excluded:
        raise ValueError("heldout recording reused as synthetic background")

    scene_rows = 0
    verified_images = 0
    for row in rows:
        # Validate both image and label content with declared checksums.
        image = (export_root / row["image"]).resolve(strict=True)
        if not image.is_relative_to(export_root):
            raise ValueError("image path escapes the export root")
        if _digest(image) != row["image_sha256"]:
            raise ValueError("train image hash changed after export")
        label_parts = Path(row["image"]).parts
        if len(label_parts) != 3 or label_parts[:2] != ("images", "train"):
            raise ValueError("train image relative path has unexpected layout")
        expected_label = export_root / "labels" / "train" / (Path(label_parts[-1]).stem + ".txt")
        if _digest(expected_label) != row["label_sha256"]:
            raise ValueError("train label hash changed after export")
        verified_images += 1
        if row["capture_id"] in scene_by_id:
            scene_rows += 1
            components = row["conditions"].get("synthetic_components")
            if not components or len(components) != len(scene_by_id[row["capture_id"]]["signal_snr_measurements"]):
                raise ValueError("scene window lost per-component SNR provenance")
            if row["conditions"]["synthetic_snr_profile"] != "balanced":
                raise ValueError("train-only scene is not in balanced SNR mode")
    if scene_rows != len(scene_by_id):
        raise ValueError("at least one indexed scene is absent from exported windows")
    return {
        "valid_train_only_release": True,
        "source_data_dir": str(source_root),
        "scene_index_sha256": _digest(index_file),
        "cv_export_report_sha256": _digest(export_root / "export-report.json"),
        "frozen_exclusions_sha256": _digest(exclusion_path),
        "train_images": len(rows),
        "verified_image_and_label_pairs": verified_images,
        "scene_images": scene_rows,
        "scene_count": len(scene_by_id),
        "independent_signal_source_waveform_count": len(signal_source_shas),
        "independent_recorded_noise_capture_count": len(noise_original_shas),
        "excluded_evaluation_recording_count": len(excluded),
        "noise_backgrounds": dict(noise_backgrounds),
        "signal_snr_bins": dict(snr_bins),
        "validation_images": 0,
        "test_images": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument("--scene-index", type=Path, required=True)
    parser.add_argument("--train-export", type=Path, required=True)
    parser.add_argument("--frozen-exclusions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_train_only_release(args.source_data_dir, args.scene_index, args.train_export, args.frozen_exclusions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
