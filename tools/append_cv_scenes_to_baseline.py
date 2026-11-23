"""Safely extend a frozen, lineage-held-out detector export with new TRAIN scenes.

This is an alternative to regenerating old spectrograms: the original
validation/test image bytes and their source groups remain unchanged. Newly
composed scenes are train-only and rejected if they reuse review-derived noise
from a held-out capture, or signal components from a held-out capture.

All images use the exact same versioned CV preprocessing function as the app.
Copying the frozen source dataset preserves all original image/label bytes.
Never edit the frozen source export.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from sigid.cv_dataset import _window_boxes, _window_conditions
from sigid.cv_detector import PREPROCESSING_VERSION, spectrogram_image
from sigid.dataset import CaptureStore
from sigid.wav import WavSource


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extend_frozen_cv_dataset(
    baseline: Path, store_dir: Path, scene_index: Path, output: Path, *,
    max_noise_train: int = 30,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite staged dataset {output}")
    baseline = baseline.resolve(strict=True)
    store_dir = store_dir.resolve(strict=True)
    index = json.loads(scene_index.read_text())
    if index.get("config", {}).get("two_signal_layout") != "mixture-disjoint-and-simultaneous-v1":
        raise ValueError("scene corpus must contain the audited two-signal recipe")
    rows = [json.loads(line) for line in (baseline / "windows.jsonl").read_text().splitlines() if line]
    holdout_rows = [r for r in rows if r["split"] in {"validation", "test"}]
    holdout_sha = {r["capture_sha256"] for r in holdout_rows}
    holdout_groups = {r["lineage_group"] for r in holdout_rows}
    frozen_holdout_hash = sha256(
        "".join(f"{r['image_sha256']}:{r['label_sha256']}\n" for r in holdout_rows).encode()
    ).hexdigest()
    source_report = json.loads((baseline / "export-report.json").read_text())
    if source_report.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("frozen dataset preprocessing is incompatible")
    if not all((baseline / r["image"]).is_file() for r in rows):
        raise ValueError("frozen dataset image manifest is incomplete")
    # The source and staging folders may be on different filesystems;
    # hard-link attempts fail with EXDEV. Copying is safe on both layouts.
    shutil.copytree(baseline, output, copy_function=shutil.copy2)
    # The baseline export is immutable. Replace only staged metadata.
    for metadata_name in ("data.yaml", "windows.jsonl", "export-report.json"):
        (output / metadata_name).unlink()
    for stale in output.rglob("*.cache"):
        if stale.is_file():
            stale.unlink()
    (output / "data.yaml").write_text(
        f"path: {output.resolve()}\ntrain: images/train\nval: images/validation\n"
        "test: images/test\nnc: 1\nnames: [signal]\n"
    )
    store = CaptureStore(store_dir / "dataset.sqlite3")
    extras = []
    blocked_noise = blocked_signal = included_concurrent = included_pairs = 0

    def add_capture(capture_id: str, *, require_signal: bool) -> dict:
        capture = store.capture(capture_id)
        regions = store.regions(capture_id)
        sample_rate = int(capture["sample_rate"])
        count = int(capture["sample_count"])
        if count != sample_rate * 10:
            raise ValueError("staged CV scenes must be exact 10-second audio windows")
        labels, box_types, error = _window_boxes(regions, 0, count)
        if error or (require_signal and not labels) or (not require_signal and labels):
            raise ValueError(f"invalid reviewed scene boxes for {capture_id}: {error}")
        audio = np.asarray(
            [sample for block in WavSource(capture["audio_path"]).blocks()
             for sample in block.samples], dtype=np.float64
        )
        if audio.size != count:
            raise ValueError(f"audio sample count mismatch for {capture_id}")
        stem = capture_id + "-000000000000"
        image_path = output / "images/train" / (stem + ".png")
        label_path = output / "labels/train" / (stem + ".txt")
        if image_path.exists() or label_path.exists():
            raise ValueError("staged scene would overwrite a frozen image")
        Image.fromarray(spectrogram_image(audio, sample_rate), "RGB").save(image_path)
        label_path.write_text("\n".join(labels) + ("\n" if labels else ""))
        conditions = _window_conditions(regions, capture, 0, count)
        lineage = sha256(("20261122-staged:" + capture["sha256"]).encode()).hexdigest()
        if lineage in holdout_groups:
            raise ValueError("staged lineage collides with frozen held-out group")
        return {
            "box_count": len(labels), "box_types": box_types,
            "capture_id": capture_id, "capture_sha256": capture["sha256"],
            "conditions": conditions, "image": str(image_path.relative_to(output)),
            "image_sha256": _hash(image_path), "label_sha256": _hash(label_path),
            "lineage_group": lineage, "modulation_labels": [],
            "preprocessing_version": PREPROCESSING_VERSION,
            "sample_count": count, "sample_start": 0, "sample_rate": sample_rate,
            "source_generator": capture["source_metadata"].get("generator"),
            "source_kind": "synthetic", "split": "train",
            "window_sample_count": count,
            "review_ids": [],
            "staged_from_audited_scene_corpus": True,
        }

    for record in index["records"]:
        scene_id = record["scene_capture_id"]
        noise = store.capture(record["noise_capture_id"])
        sources = noise["source_metadata"].get("sources", [])
        if any(s.get("sha256") in holdout_sha for s in sources):
            blocked_noise += 1
            continue
        if any(store.capture(cid)["sha256"] in holdout_sha for cid in record["source_capture_ids"]):
            blocked_signal += 1
            continue
        regs = [r for r in store.regions(scene_id) if r.get("method") is not None]
        if len(regs) == 2:
            included_pairs += 1
            a, b = regs
            intersection = max(0, min(a["sample_start"] + a["sample_count"],
                                      b["sample_start"] + b["sample_count"])
                               - max(a["sample_start"], b["sample_start"]))
            if intersection:
                included_concurrent += 1
                if intersection * 2 < min(a["sample_count"], b["sample_count"]):
                    raise ValueError("simultaneous scene violates configured overlap")
        extras.append(add_capture(scene_id, require_signal=True))
    negative_count = 0
    for record in index.get("noise_only_records", []):
        if negative_count >= max_noise_train:
            break
        cid = record.get("capture_id") or record.get("noise_capture_id")
        if not isinstance(cid, str):
            raise ValueError("noise-only index record missing capture id")
        if store.capture(cid)["sha256"] in holdout_sha:
            continue
        extras.append(add_capture(cid, require_signal=False))
        negative_count += 1
    if not extras:
        raise ValueError("no safely isolated new scenes available")
    rows.extend(extras)
    (output / "windows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    report = source_report.copy()
    for name in ("train", "validation", "test"):
        subset = [r for r in rows if r["split"] == name]
        report["split_images"][name] = len(subset)
        report["positive_images_by_split"][name] = sum(r["box_count"] > 0 for r in subset)
        report["negative_images_by_split"][name] = sum(r["box_count"] == 0 for r in subset)
        report["split_groups"][name] = len({r["lineage_group"] for r in subset})
    report["group_count"] = len({r["lineage_group"] for r in rows})
    report["positive_images"] = sum(r["box_count"] > 0 for r in rows)
    report["negative_images"] = sum(r["box_count"] == 0 for r in rows)
    report["captures_with_windows"] = len({r["capture_id"] for r in rows})
    report["images_by_source_kind"] = dict(Counter(r["source_kind"] for r in rows))
    report["images_by_generator"] = dict(Counter(
        r["source_generator"] for r in rows if r["source_generator"]
    ))
    report["negative_images_by_source_kind"] = dict(Counter(
        r["source_kind"] for r in rows if r["box_count"] == 0
    ))
    report["source_data_dirs"] = list(dict.fromkeys(
        list(report["source_data_dirs"]) + [str(store_dir)]
    ))
    report["frozen_holdout_extension"] = {
        "new_positive_train_scenes": sum(r["box_count"] > 0 for r in extras),
        "new_noise_only_train_scenes": negative_count,
        "new_two_signal_train_scenes": included_pairs,
        "new_simultaneous_train_scenes": included_concurrent,
        "excluded_shared_heldout_noise_scenes": blocked_noise,
        "excluded_shared_heldout_signal_scenes": blocked_signal,
        "frozen_holdout_rows": len(holdout_rows),
        "frozen_holdout_sha256": frozen_holdout_hash,
        "heldout_image_and_label_bytes_unchanged": True,
        "test_and_validation_splits_original": True,
        "reproducible_from": {"baseline": str(baseline), "scene_index": str(scene_index)},
    }
    (output / "export-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report["frozen_holdout_extension"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-noise-train", type=int, default=30)
    args = parser.parse_args()
    result = extend_frozen_cv_dataset(
        args.baseline, args.store, args.scenes, args.output,
        max_noise_train=args.max_noise_train,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
