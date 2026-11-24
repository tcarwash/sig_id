"""Preview receiver imports preserve provenance but never invent annotations."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import wave

import numpy as np

from sigid.dataset import CaptureStore
from tools.import_review_sources import import_receiver_candidates


def _wav(path: Path, *, frequency: int = 900) -> None:
    rate = 12000
    samples = (0.05 * 32767 * np.sin(2 * np.pi * frequency * np.arange(10 * rate) / rate)).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())


def test_imported_audio_is_only_a_review_candidate(tmp_path):
    source_dir = tmp_path / "external"
    dest_dir = tmp_path / "main"
    source_dir.mkdir()
    dest_dir.mkdir()
    external_store = CaptureStore(source_dir / "dataset.sqlite3")
    local_store = CaptureStore(dest_dir / "dataset.sqlite3")
    wave_path = source_dir / "radio.wav"
    _wav(wave_path)
    source_capture = external_store.add_wav_capture(wave_path, source_kind="live", source_metadata={
        "device_name": "USB Audio CODEC", "auto_review_queue": True,
        "auto_capture_session_id": "real-session-a",
    })
    lab_path = source_dir / "lab.wav"
    _wav(lab_path, frequency=1000)
    external_store.add_wav_capture(lab_path, source_kind="live", source_metadata={
        "device_name": "sigid_signal_lab.monitor",
        "auto_review_queue": True, "auto_capture_session_id": "synthetic-session",
    })
    report = import_receiver_candidates(
        source_dir, dest_dir, device_match="USB Audio CODEC",
        backup=tmp_path / "capture-backup.sqlite3",
    )
    assert report["imported_count"] == 1
    assert report["newly_verified_labels"] == 0
    assert report["review_only"] is True
    items = local_store.list_captures()
    assert len(items) == 1
    assert items[0]["sha256"] == external_store.capture(source_capture)["sha256"]
    assert items[0]["source_metadata"]["active_learning_unverified_source"] is True
    assert items[0]["source_metadata"]["auto_capture_session_id"] == "real-session-a"
    assert local_store.regions(items[0]["id"]) == []
    assert local_store.scene_coverages(items[0]["id"]) == []
    assert Path(report["backup_path"]).exists()
    with sqlite3.connect(report["backup_path"]) as backup:
        assert backup.execute("select count(*) from captures").fetchone()[0] == 0
    second = import_receiver_candidates(
        source_dir, dest_dir, device_match="USB Audio CODEC",
        backup=tmp_path / "second-backup.sqlite3",
    )
    assert second["imported_count"] == 0
    assert second["skipped"]["duplicate_sha"] == 1
