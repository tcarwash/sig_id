"""Listen-only mode and reversible bulk clearing never erase training data."""
from __future__ import annotations

import wave

import numpy as np
from fastapi.testclient import TestClient

from sigid.api import create_app
from sigid.dataset import CaptureStore
from test_live_audio import FakeAudio


def _wav(path):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(12000)
        w.writeframes(np.zeros(120000, dtype="<i2").tobytes())


def test_listen_only_writes_no_wav_or_queued_reviews(tmp_path, monkeypatch):
    monkeypatch.delenv("SIG_ID_AUTO_INSTALL_MODELS", raising=False)
    app = create_app(data_dir=tmp_path, audio_backend=FakeAudio())
    with TestClient(app) as client:
        start = client.post("/api/live/start", json={
            "capture_mode": "scan_only", "device_id": 1,
            "sample_rate": 12000, "max_duration_seconds": 5,
        })
        assert start.status_code == 200, start.text
        assert start.json()["capture_mode"] == "scan_only"
        status = client.get("/api/live").json()
        assert status["active"] is True
        assert status["capture_mode"] == "scan_only"
        assert client.get("/api/live/analysis").status_code == 200
        stop = client.post("/api/live/stop")
        assert stop.status_code == 200, stop.text
        assert stop.json()["capture"] is None
        assert client.get("/api/live/review-queue").json()["pending_count"] == 0
    assert list((tmp_path / "captures").glob("*.wav")) == []
    assert CaptureStore(tmp_path / "dataset.sqlite3").list_captures() == []


def test_clear_queue_keeps_recordings_reviews_and_can_restore(tmp_path, monkeypatch):
    monkeypatch.delenv("SIG_ID_AUTO_INSTALL_MODELS", raising=False)
    store = CaptureStore(tmp_path / "dataset.sqlite3")
    ids = []
    for index in range(3):
        path = tmp_path / f"clip-{index}.wav"
        _wav(path)
        cid = store.add_wav_capture(path, source_kind="live", source_metadata={
            "auto_review_queue": True, "device_name": "Receiver",
            "window_start_sample": 0, "window_seconds": 10,
        })
        ids.append(cid)
    # A manually reviewed window is not a pending task and must be unaffected.
    store.set_live_reviewed(ids[-1], True)
    other = tmp_path / "not-in-queue.wav"
    _wav(other)
    other_id = store.add_wav_capture(other, source_kind="live", source_metadata={
        "auto_review_queue": False,
    })
    original_bytes = {x["id"]: (tmp_path / f"clip-{i}.wav").read_bytes() for i,x in enumerate(store.capture(cid) for cid in ids)}
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        initial = client.get("/api/live/review-queue").json()
        assert initial["pending_count"] == 2
        assert initial["dismissed_count"] == 0
        cleared = client.post("/api/live/review-queue/clear")
        assert cleared.status_code == 200, cleared.text
        assert cleared.json() == {"dismissed": 2, "deleted": 0, "annotations_changed": 0}
        body = client.get("/api/live/review-queue").json()
        assert body["pending_count"] == 0
        assert body["dismissed_count"] == 2
        assert body["reviewed_count"] == 1
        assert client.post("/api/live/review-queue/clear").json()["dismissed"] == 0
        for cid in ids[:2]:
            meta = store.capture(cid)["source_metadata"]
            assert "auto_review_queue_dismissed_utc" in meta
            assert "auto_review_queue_reviewed_utc" not in meta
        assert "auto_review_queue_dismissed_utc" not in store.capture(other_id)["source_metadata"]
        assert "auto_review_queue_dismissed_utc" not in store.capture(ids[-1])["source_metadata"]
        assert all(store.regions(cid) == [] and store.scene_coverages(cid) == [] for cid in ids)
        restored = client.post("/api/live/review-queue/restore-dismissed")
        assert restored.status_code == 200
        assert restored.json() == {"restored": 2, "deleted": 0, "annotations_changed": 0}
        again = client.get("/api/live/review-queue").json()
        assert again["pending_count"] == 2 and again["dismissed_count"] == 0
        assert client.post("/api/live/review-queue/restore-dismissed").json()["restored"] == 0
    for i, cid in enumerate(ids):
        assert (tmp_path / f"clip-{i}.wav").read_bytes() == original_bytes[cid]


def test_default_start_option_listens_without_recording():
    from pathlib import Path
    import re
    html = (Path(__file__).parents[1] / "sigid" / "web" / "index.html").read_text()
    assert re.search(r'<option value="scan_only" selected>Listen live · no recording</option>', html)
    assert 'id="clear-pending-review-queue"' in html
    assert 'id="restore-dismissed-review-queue"' in html
