"""The workbench prioritizes staged review items without making labels."""
from __future__ import annotations

import hashlib
import json
import wave

import numpy as np
from fastapi.testclient import TestClient

from sigid.api import create_app
from sigid.dataset import CaptureStore


def _wav(path):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(12000)
        w.writeframes(np.zeros(120000, dtype="<i2").tobytes())


def test_curated_queue_orders_existing_pending_capture_without_accepting_labels(tmp_path, monkeypatch):
    store = CaptureStore(tmp_path / "dataset.sqlite3")
    recordings = []
    for index in range(2):
        path = tmp_path / f"sample-{index}.wav"
        _wav(path)
        capture_id = store.add_wav_capture(path, source_kind="live", source_metadata={
            "auto_review_queue": True,
            "device_name": "receiver",
            "window_seconds": 10,
            "window_start_sample": 0,
            "auto_capture_session_id": f"session-{index}",
        })
        recordings.append(store.capture(capture_id))
    result = {
        "schema_version": 1,
        "training_eligible": False,
        "frozen_exclusion_manifest_sha256": "a" * 64,
        "selected": [{
            "capture_id": recordings[1]["id"],
            "capture_sha256": recordings[1]["sha256"],
            "priority_reasons": ["unannotated", "session-diverse"],
        }],
    }
    (tmp_path / "active-learning-review-batch.json").write_text(json.dumps(result))
    monkeypatch.delenv("SIG_ID_AUTO_INSTALL_MODELS", raising=False)
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/live/review-queue")
        assert response.status_code == 200
        body = response.json()
        assert body["active_learning_prioritized_count"] == 1
        assert body["pending"][0]["id"] == recordings[1]["id"]
        assert body["pending"][0]["source_metadata"]["active_learning_rank"] == 1
        assert body["pending"][0]["source_metadata"]["active_learning_reason"] == [
            "unannotated", "session-diverse"]
    assert all(store.regions(c["id"]) == [] for c in recordings)
    assert all(store.scene_coverages(c["id"]) == [] for c in recordings)
