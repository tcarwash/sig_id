"""Freeze fully reviewed *new real sessions* into a detector-only shadow test.

No relabeling or writes to the user's source DB. The tool snapshots SQLite
through a read-only connection, computes native capture lineage including
whole recording sessions, excludes ANY session used in the frozen training
export, and generates fresh v2 images from SHA-verified PCM. The resulting
shadow split must not be used for model threshold tuning or training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from PIL import Image

from sigid.cv_dataset import (
    _is_saved_ten_second_sample, _lineage_groups, _review_covers_window, _window_boxes,
)
from sigid.cv_detector import PREPROCESSING_VERSION, spectrogram_image
from sigid.dataset import CaptureStore
from tools.bench_reviewed_chain import read_verified_pcm, sha256_file


def build_shadow_holdout(
    source_root: Path, prior_dataset: Path, destination: Path,
) -> dict:
    if destination.exists():
        raise FileExistsError("shadow holdout is immutable; choose an unused output directory")
    source_db = source_root / "dataset.sqlite3"
    if not source_db.is_file():
        raise FileNotFoundError(source_db)
    original_rows = [
        json.loads(line) for line in (prior_dataset / "windows.jsonl").read_text().splitlines()
        if line.strip()
    ]
    prior_source_hashes = {row["capture_sha256"] for row in original_rows}
    prior_live_ids = {row["capture_id"] for row in original_rows if row["source_kind"] == "live"}
    destination.mkdir(parents=True, exist_ok=False)
    snapshot_root = destination / "source_snapshot"
    snapshot_root.mkdir()
    snapshot_db = snapshot_root / "dataset.sqlite3"
    try:
        with sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True) as original:
            with sqlite3.connect(snapshot_db) as cloned:
                original.backup(cloned)
        store = CaptureStore(snapshot_db)  # schema/index changes affect clone only
        # CaptureStore may migrate its private snapshot on first open, so
        # record the final snapshot digest after that initialization, not
        # the pre-migration SQLite backup bytes.
        source_sha = sha256_file(snapshot_db)
        captures = store.list_captures()
        by_id = {row["id"]: row for row in captures}
        lineage = _lineage_groups(captures)

        unknown_prior_live = prior_live_ids - by_id.keys()
        if unknown_prior_live:
            raise ValueError(f"{len(unknown_prior_live)} original live IDs absent from source snapshot")
        prior_groups = {lineage[cid] for cid in prior_live_ids}
        prior_sessions = {
            by_id[cid]["source_metadata"].get("auto_capture_session_id") or cid
            for cid in prior_live_ids
        }
        selected = []
        reasons = Counter()
        for capture in captures:
            cid = capture["id"]
            if not _is_saved_ten_second_sample(capture):
                reasons["not_canonical_review_window"] += 1
                continue
            session = capture["source_metadata"].get("auto_capture_session_id") or cid
            if cid in prior_live_ids or capture["sha256"] in prior_source_hashes:
                reasons["already_in_prior_export"] += 1
                continue
            if session in prior_sessions or lineage[cid] in prior_groups:
                reasons["source_lineage_used_in_training"] += 1
                continue
            regions = store.regions(cid)
            if any(region["lineage_state"] == "active"
                   and region["review_state"] not in ("accepted", "rejected")
                   for region in regions):
                reasons["unresolved_active_labels"] += 1
                continue
            if not any(_review_covers_window(review, 0, int(capture["sample_count"]))
                       for review in store.scene_coverages(cid)):
                reasons["unreviewed_or_stale_coverage"] += 1
                continue
            lines, types, issue = _window_boxes(regions, 0, int(capture["sample_count"]))
            if issue:
                reasons["incomplete_signal_geometry"] += 1
                continue
            if not lines:
                reasons["no_accepted_signal_boxes"] += 1
                continue
            selected.append((capture, types, lines, lineage[cid]))

        if not selected:
            raise ValueError("no independent fully reviewed shadow windows")
        (destination / "images" / "test").mkdir(parents=True)
        (destination / "labels" / "test").mkdir(parents=True)
        manifests = []
        for capture, types, lines, group in sorted(selected, key=lambda x: x[0]["sha256"]):
            cid = capture["id"]
            pcm = read_verified_pcm(capture)
            if len(pcm) != capture["sample_count"]:
                raise ValueError("PCM source length mismatch")
            image_name = f"{cid}-000000000000.png"
            rel_image = Path("images") / "test" / image_name
            image_path = destination / rel_image
            Image.fromarray(spectrogram_image(pcm, int(capture["sample_rate"])), mode="RGB").save(image_path)
            label_path = destination / "labels" / "test" / f"{image_path.stem}.txt"
            label_path.write_text("\n".join(lines) + "\n")
            manifests.append({
                "split": "test", "capture_id": cid,
                "capture_sha256": capture["sha256"],
                "source_kind": "live", "source_generator": None,
                "lineage_group": group,
                "sample_start": 0, "sample_count": capture["sample_count"],
                "sample_rate": capture["sample_rate"],
                "image": str(rel_image),
                "image_sha256": sha256_file(image_path),
                "label_sha256": sha256_file(label_path),
                "box_count": len(lines), "box_types": types,
                "preprocessing_version": PREPROCESSING_VERSION,
                "conditions": {"source_kind": "live", "signal_overlap": "multiple" if len(lines)>1 else "single"},
            })
        (destination / "windows.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in manifests) + "\n"
        )
        groups = Counter(row["lineage_group"] for row in manifests)
        report = {
            "schema_version": 2,
            "preprocessing_version": PREPROCESSING_VERSION,
            "scope": "shadow-only independent real receiver sessions; NEVER training or threshold selection",
            "source_data_dirs": [str(snapshot_root)],
            "source_snapshot_sha256": source_sha,
            "original_export_manifest_sha256": sha256_file(prior_dataset / "windows.jsonl"),
            "split_images": {"train": 0, "validation": 0, "test": len(manifests)},
            "test_live_lineages": len(groups),
            "test_live_boxes": sum(x["box_count"] for x in manifests),
            "lineage_group_window_counts": dict(sorted(groups.items())),
            "exclusion_reasons": dict(reasons),
            "reproducible": True,
            "independence": "strict exclusion by original capture SHA, source-session ID, and full-current lineage group",
        }
        (destination / "export-report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n"
        )
        return {k: report[k] for k in (
            "test_live_lineages", "test_live_boxes", "split_images",
            "independence", "source_snapshot_sha256",
        )}
    except Exception:
        # On any partial failure, never publish a possibly incomplete
        # benchmark as if it were a valid frozen test.
        marker = destination / "INVALID_INCOMPLETE"
        marker.write_text("Shadow snapshot was interrupted or invalid; do not evaluate.\n")
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-data-dir", type=Path, required=True)
    p.add_argument("--prior-dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(build_shadow_holdout(args.source_data_dir, args.prior_dataset,
                                          args.output), indent=2))


if __name__ == "__main__":
    main()
