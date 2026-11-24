"""Fresh human-reviewed real export must protect original held-out sessions."""
from __future__ import annotations

import json
from pathlib import Path
import wave

import numpy as np
import pytest

from sigid.cv_dataset import export_cv_dataset
from sigid.dataset import CaptureStore, SignalLabel


def _wav(path: Path, *, freq: float):
    rate = 12000
    t = np.arange(10 * rate) / rate
    data = (0.3 * 32767 * np.sin(2 * np.pi * freq * t)).astype("<i2")
    with wave.open(str(path), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(rate)
        dst.writeframes(data.tobytes())


def _setup(tmp_path, *, same_session=False):
    root = tmp_path / "receiver"
    root.mkdir()
    store = CaptureStore(root / "dataset.sqlite3")
    a, b = tmp_path / "holdout.wav", tmp_path / "new-reviewed.wav"
    _wav(a, freq=600)
    _wav(b, freq=1300)
    heldout_id = store.add_wav_capture(a, source_kind="live", source_metadata={
        "auto_capture_session_id": "frozen-session"})
    new_id = store.add_wav_capture(b, source_kind="live", source_metadata={
        "auto_capture_session_id": "frozen-session" if same_session else "new-session"})
    store.add_region(
        new_id, sample_start=24000, sample_count=72000,
        frequency_low_hz=1250, frequency_high_hz=1350,
        label=SignalLabel(source="human", review_state="accepted",
                          notes="Confirmed signal geometry, identity unknown"),
    )
    store.create_scene_coverage(
        new_id, sample_start=0, sample_count=120000,
        task="detection", taxonomy_scope=[], state="complete",
        notes="Operator checked all signals in this recording",
    )
    heldout_sha = store.capture(heldout_id)["sha256"]
    base = tmp_path / "base.jsonl"
    base.write_text(json.dumps({
        "split": "test", "source_kind": "live", "capture_sha256": heldout_sha
    }) + "\n")
    excluded = tmp_path / "frozen.json"
    excluded.write_text(json.dumps({
        "schema_version": 1,
        "excluded_capture_sha256": [heldout_sha],
        "frozen_holdout_capture_sha256": [heldout_sha],
    }))
    return root, store, base, excluded, new_id


def test_only_newly_human_reviewed_real_train_windows_export(tmp_path):
    root, store, base, excluded, new_id = _setup(tmp_path)
    destination = tmp_path / "output"
    report = export_cv_dataset(
        root, destination, train_only=True, reviewed_real_only=True,
        base_cv_manifest=base, frozen_exclusions=excluded,
    )
    assert report["split_images"] == {"train": 1, "validation": 0, "test": 0}
    assert report["positive_images_by_split"]["train"] == 1
    assert report["reviewed_real_train_only"] is True
    assert report["real_review_provenance"]["protected_session_count"] == 1
    windows = [json.loads(line) for line in (destination / "windows.jsonl").read_text().splitlines()]
    assert len(windows) == 1
    assert windows[0]["capture_id"] == new_id
    assert windows[0]["modulation_labels"][0]["mode_id"] is None


def test_same_session_as_heldout_must_not_export(tmp_path):
    root, store, base, excluded, new_id = _setup(tmp_path, same_session=True)
    with pytest.raises(ValueError, match="no windows"):
        export_cv_dataset(
            root, tmp_path / "output", train_only=True, reviewed_real_only=True,
            base_cv_manifest=base, frozen_exclusions=excluded,
        )


def test_real_export_refuses_missing_freeze_manifest(tmp_path):
    root, store, base, excluded, new_id = _setup(tmp_path)
    with pytest.raises(ValueError, match="requires train-only"):
        export_cv_dataset(
            root, tmp_path / "output", train_only=True, reviewed_real_only=True,
            base_cv_manifest=base,
        )
