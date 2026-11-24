"""Active-learning review batches must never turn proposals into truth."""
from __future__ import annotations

import hashlib
import json

import pytest

from sigid.active_learning import rank_review_batch, protected_source_sha


def _capture(session: str, index: int, *, device: str = "RX input", summary: dict | None = None):
    return {
        "id": f"capture-{session}-{index}",
        "sha256": hashlib.sha256(f"{session}-{index}".encode()).hexdigest(),
        "source_kind": "live",
        "sample_rate": 12000,
        "sample_count": 120000,
        "source_metadata": {
            "auto_review_queue": True,
            "auto_capture_session_id": session,
            "device_name": device,
            "review_trigger": "cv_signal" if index % 2 else "periodic_noise",
            "detection_summary": summary or {"cv_detection_count": index % 2},
        },
    }


class FakeStore:
    def __init__(self, *, complete=None, regions=None):
        self.complete = complete or set()
        self._regions = regions or {}

    def regions(self, capture_id):
        return self._regions.get(capture_id, [])

    def scene_coverages(self, capture_id):
        if capture_id not in self.complete:
            return []
        return [{
            "task": "detection", "state": "complete", "is_stale": False,
            "sample_start": 0, "sample_count": 120000,
            "frequency_low_hz": None,
        }]


def test_session_diverse_batch_never_accepts_prediction_as_truth():
    captures = (
        [_capture("session-a", i) for i in range(40)]
        + [_capture("session-b", i) for i in range(15)]
        + [_capture("session-c", i) for i in range(8)]
    )
    result = rank_review_batch(captures, FakeStore(), excluded_sha=set(), limit=10)
    assert result["selected_count"] == 10
    assert result["selected_independent_sessions"] == 3
    assert result["training_eligible"] is False
    assert all(row["label_trust"] == "unverified" for row in result["selected"])
    assert result["selected"] == rank_review_batch(
        captures, FakeStore(), excluded_sha=set(), limit=10)["selected"]
    assert all("mode_id" not in row for row in result["selected"])


def test_frozen_source_excludes_entire_session_and_verified_reviews():
    captures = [_capture("holdout", i) for i in range(5)] + [_capture("train", i) for i in range(6)]
    result = rank_review_batch(
        captures, FakeStore(complete={captures[5]["id"]}),
        excluded_sha={captures[0]["sha256"]}, limit=50,
    )
    assert all(row["capture_id"].startswith("capture-train-") for row in result["selected"])
    assert result["skipped"]["frozen-session"] == 5
    assert result["skipped"]["completed-detection-review"] == 1
    assert result["selected_count"] == 5


def test_loopback_and_duplicate_hashes_not_promoted():
    a = _capture("first", 0)
    b = _capture("second", 0)
    b["sha256"] = a["sha256"]
    loopback = _capture("lab", 0, device="sigid_signal_lab.monitor")
    result = rank_review_batch([a, b, loopback], FakeStore(), excluded_sha=set(), limit=10)
    assert result["selected_count"] == 1
    assert result["skipped"]["signal-lab-loopback"] == 1
    assert result["skipped"]["duplicate-source-sha"] == 1


def test_frozen_exclusions_require_complete_sha_manifest(tmp_path):
    a = "a" * 64
    p = tmp_path / "exclusions.json"
    p.write_text(json.dumps({
        "schema_version": 1,
        "excluded_capture_sha256": [a, "b" * 64],
        "frozen_holdout_capture_sha256": [a],
    }))
    assert protected_source_sha(p) == {a, "b" * 64}
    p.write_text(json.dumps({
        "schema_version": 1,
        "excluded_capture_sha256": ["b" * 64],
        "frozen_holdout_capture_sha256": [a],
    }))
    with pytest.raises(ValueError, match="incomplete"):
        protected_source_sha(p)
