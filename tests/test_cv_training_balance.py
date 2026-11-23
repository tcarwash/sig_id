"""No-leakage regression checks for live-only CV train oversampling."""

import json
from pathlib import Path

import pytest

from tools.prepare_cv_training_balance import rebalance_cv_train


def test_live_train_positives_repeat_without_touching_holdouts(tmp_path):
    for split in ("train", "validation", "test"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
    rows = []
    for split, kind, boxes, stem in [
        ("train", "live", 1, "real"),
        ("train", "live", 0, "noise"),
        ("train", "synthetic", 1, "synthetic"),
        ("validation", "live", 1, "validation"),
        ("test", "live", 1, "test"),
    ]:
        img = f"images/{split}/{stem}.png"
        path = tmp_path / img
        path.write_bytes(b"test image bytes")
        (tmp_path / f"labels/{split}/{stem}.txt").write_text(
            "0 .5 .5 .2 .2\n" if boxes else ""
        )
        rows.append({
            "split": split, "source_kind": kind, "box_count": boxes,
            "image": img, "capture_id": stem, "lineage_group": stem,
        })
    (tmp_path / "windows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    report = rebalance_cv_train(tmp_path, live_repetitions=3)
    assert report["source_positive_live_windows"] == 1
    assert report["repeated_train_images"] == 2
    assert (tmp_path / "images/train/real-live-repeat-1.png").samefile(
        tmp_path / "images/train/real.png"
    )
    assert (tmp_path / "labels/train/real-live-repeat-2.txt").samefile(
        tmp_path / "labels/train/real.txt"
    )
    assert len(list((tmp_path / "images/validation").glob("*.png"))) == 1
    assert len(list((tmp_path / "images/test").glob("*.png"))) == 1
    assert rebalance_cv_train(tmp_path, live_repetitions=3) == report
    with pytest.raises(ValueError):
        rebalance_cv_train(tmp_path, live_repetitions=4)
