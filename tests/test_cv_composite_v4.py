"""Tests for combining existing CV evaluation splits with v4 train-only data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.compose_cv_training_dataset import compose


def _digest(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _source(root: Path, name: str, *, extra: bool) -> None:
    rows = []
    for split in ("train",) if extra else ("train", "validation", "test"):
        count = 2 if split == "train" and extra else 1
        for num in range(count):
            stem = f"{name}-{split}-{num}"
            image = f"images/{split}/{stem}.png"
            label = f"labels/{split}/{stem}.txt"
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
            payload = f"PNG-FIXTURE-{stem}".encode("utf-8")
            label_bytes = b"0 0.5 0.5 0.2 0.3\n"
            (root / image).write_bytes(payload)
            (root / label).write_bytes(label_bytes)
            rows.append({
                "split": split, "image": image,
                "image_sha256": _digest(payload), "label_sha256": _digest(label_bytes),
                "capture_sha256": _digest(f"capture-{stem}".encode()),
                "lineage_group": f"lineage-{stem}",
                "capture_id": stem, "sample_start": 0,
                "box_count": 1,
            })
    (root / "windows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "export-report.json").write_text(json.dumps({
        "schema_version": 2,
        "preprocessing_version": "sig-id-live-waterfall-v2",
        "split_images": {key: sum(row["split"] == key for row in rows) for key in ("train", "validation", "test")},
        "train_only": extra,
        "scene_index_sha256": "a" * 64 if extra else None,
    }))


def test_combined_data_adds_only_training_and_leaves_heldout_pixel_hashes(tmp_path):
    original = tmp_path / "base"
    supplement = tmp_path / "supplement"
    output = tmp_path / "combined"
    _source(original, "old", extra=False)
    _source(supplement, "new", extra=True)
    result = compose(original, supplement, output)
    assert result["split_images"] == {"train": 3, "validation": 1, "test": 1}
    assert result["composite_dataset"]["source_train_added"] == 2
    assert result["composite_dataset"]["heldout_image_sha256_unchanged"] is True
    rows = [json.loads(line) for line in (output / "windows.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    assert [row["combined_provenance"] for row in rows].count("v4") == 2
    assert all(row["split"] == "train" for row in rows if row["combined_provenance"] == "v4")
    assert {(row["capture_sha256"], row["image_sha256"], row["label_sha256"]) for row in rows if row["split"] == "test"} == {
        (json.loads(line)["capture_sha256"], json.loads(line)["image_sha256"], json.loads(line)["label_sha256"])
        for line in (original / "windows.jsonl").read_text().splitlines()
        if json.loads(line)["split"] == "test"
    }
    with pytest.raises(FileExistsError):
        compose(original, supplement, output)


def test_composite_rejects_shared_heldout_lineage(tmp_path):
    base, supplement, out = tmp_path / "old", tmp_path / "v4", tmp_path / "out"
    _source(base, "old", extra=False)
    _source(supplement, "new", extra=True)
    path = supplement / "windows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["lineage_group"] = "lineage-old-test-0"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="lineage"):
        compose(base, supplement, out)
    assert not out.exists()


def test_composite_rejects_corrupted_source_bytes(tmp_path):
    base, supplement, out = tmp_path / "old", tmp_path / "v4", tmp_path / "out"
    _source(base, "old", extra=False)
    _source(supplement, "new", extra=True)
    (supplement / "images" / "train" / "new-train-0.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        compose(base, supplement, out)
    assert not out.exists()


def test_real_repeat_only_resamples_existing_train_and_not_heldouts(tmp_path):
    from tools.compose_cv_training_dataset import compose
    import json
    base = tmp_path / "base"
    supplement = tmp_path / "extra"
    _source(base, "old", extra=False)
    _source(supplement, "new", extra=True)
    rows = [json.loads(line) for line in (base / "windows.jsonl").read_text().splitlines()]
    for row in rows:
        row["source_kind"] = "live" if row["split"] == "train" else "synthetic"
        row["review_ids"] = ["confirmed-detection-review"]
    (base / "windows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    dest = tmp_path / "weighted"
    report = compose(base, supplement, dest, real_train_repeat=4)
    assert report["split_images"] == {"train": 6, "validation": 1, "test": 1}
    assert report["composite_dataset"]["real_training_resampled_images"] == 3
    assert report["composite_dataset"]["real_source_independent_count"] == 1
    images = [json.loads(line) for line in (dest / "windows.jsonl").read_text().splitlines()]
    assert len([row for row in images if row.get("sampling_repeat_index")]) == 3
    assert all(row["split"] == "train" for row in images if row.get("sampling_repeat_index"))
    assert len([row for row in images if row["split"] in ("test", "validation")]) == 2
    assert all(row["image_sha256"] == rows[0]["image_sha256"] for row in images if row.get("sampling_repeat_index"))
