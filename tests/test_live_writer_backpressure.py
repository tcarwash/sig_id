"""Long-running capture regression: slow review indexing never blocks WAV writing."""
from __future__ import annotations

import struct
import threading
import time
import wave

from sigid.dataset import CaptureStore
from sigid.live_audio import LiveCaptureManager

from test_live_audio import FakeAudio


def _await(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("capture did not make expected forward progress")


def test_slow_review_database_does_not_block_audio_writer(tmp_path, monkeypatch):
    store = CaptureStore(tmp_path / "dataset.sqlite3")
    backend = FakeAudio()
    manager = LiveCaptureManager(store, tmp_path / "captures", backend)
    entered = threading.Event()
    release = threading.Event()
    real_add = store.add_wav_capture

    def slow_review_index(path, **kwargs):
        if str(path).split("/")[-1].startswith("review-"):
            entered.set()
            if not release.wait(timeout=8):
                raise TimeoutError("test did not release slow index")
        return real_add(path, **kwargs)

    monkeypatch.setattr(store, "add_wav_capture", slow_review_index)
    started = manager.start(device_id=1, sample_rate=8000, max_duration_seconds=60)
    callback = backend.last_stream.kwargs["callback"]
    sample = struct.pack("<h", 1234) * 8000

    try:
        for i in range(27):
            callback(sample, 8000, None, None)
            time.sleep(0.006)
        assert entered.wait(timeout=4)
        _await(lambda: manager.status()["frames_written"] >= 160000)
        snapshot = manager.status()
        assert snapshot["state"] == "recording"
        assert snapshot["writer_queue_depth"] == 0
        assert snapshot["review_queue_depth"] >= 1
        assert snapshot["saved_review_clips"] == 0
        assert snapshot["review_queue_capacity"] == 4
        assert snapshot["review_windows_dropped"] == 0
    finally:
        release.set()

    done = manager.stop()
    assert done["capture"]["sample_count"] == 1024 + 27 * 8000
    assert done["capture"]["source_metadata"]["truncated_reason"] is None
    assert len([c for c in store.list_captures() if c["source_metadata"].get("auto_review_queue")]) == 2
    assert manager.status() == {"active": False}


def test_overflow_discards_only_review_copies_not_full_recording(tmp_path, monkeypatch):
    store = CaptureStore(tmp_path / "dataset.sqlite3")
    backend = FakeAudio()
    manager = LiveCaptureManager(store, tmp_path / "captures", backend)
    entered = threading.Event()
    release = threading.Event()
    real_add = store.add_wav_capture

    def slow_review_index(path, **kwargs):
        if str(path).split("/")[-1].startswith("review-"):
            entered.set()
            if not release.wait(timeout=8):
                raise TimeoutError("test did not release slow index")
        return real_add(path, **kwargs)

    monkeypatch.setattr(store, "add_wav_capture", slow_review_index)
    manager.start(device_id=1, sample_rate=8000, max_duration_seconds=60)
    callback = backend.last_stream.kwargs["callback"]
    sample = struct.pack("<h", 1234) * 8000

    try:
        for i in range(80):
            callback(sample, 8000, None, None)
            time.sleep(0.006)
        assert entered.wait(timeout=4)
        _await(lambda: manager.status()["frames_written"] >= 640000)
        snapshot = manager.status()
        assert snapshot["state"] == "recording"
        assert snapshot["review_windows_dropped"] >= 2
        assert snapshot["review_queue_depth"] <= 4
        assert snapshot["truncated_reason"] is None
    finally:
        release.set()

    done = manager.stop()
    assert done["capture"]["sample_count"] == 1024 + 80 * 8000
    assert done["capture"]["source_metadata"]["truncated_reason"] is None
    with wave.open(done["capture"]["audio_path"], "rb") as reader:
        assert reader.getnframes() == 1024 + 80 * 8000
