"""Proposal disagreement can change review priority, never supervised truth."""
from __future__ import annotations

from sigid.active_learning import rank_review_batch
from tools.score_active_learning_queue import matched_track_count


def _capture(session: str, sha: str) -> dict:
    return {
        "id": session,
        "sha256": sha,
        "source_kind": "live",
        "sample_rate": 12000,
        "sample_count": 120000,
        "source_metadata": {
            "auto_review_queue": True,
            "auto_capture_session_id": session,
            "device_name": "Receiver",
            "review_trigger": "continuous_recording",
        },
    }


class EmptyStore:
    def regions(self, _capture_id):
        return []

    def scene_coverages(self, _capture_id):
        return []


def test_proposal_disagreement_prioritized_but_unverified():
    recordings = [_capture("one", "a" * 64), _capture("two", "b" * 64)]
    evidence = {
        "a" * 64: {"cv_count": 0, "track_count": 2, "matched_track_count": 0,
                   "proposal_only": True},
        "b" * 64: {"cv_count": 0, "track_count": 0, "matched_track_count": 0,
                   "proposal_only": True},
    }
    batch = rank_review_batch(
        recordings, EmptyStore(), excluded_sha=set(), limit=2,
        proposal_evidence_by_sha=evidence,
    )
    assert batch["selected"][0]["capture_sha256"] == "a" * 64
    assert "spectral-track-only-proposal" in batch["selected"][0]["priority_reasons"]
    assert "possible-negative-control-unverified" in batch["selected"][1]["priority_reasons"]
    assert batch["training_eligible"] is False
    assert all(x["label_trust"] == "unverified" and "mode_id" not in x
               for x in batch["selected"])


def test_frozen_session_excluded_before_scoring_evidence():
    recordings = [
        _capture("frozen", "a" * 64),
        {**_capture("frozen", "b" * 64), "id": "frozen-alt"},
        _capture("allowed", "c" * 64),
    ]
    batch = rank_review_batch(
        recordings, EmptyStore(), excluded_sha={"a" * 64}, limit=20,
        proposal_evidence_by_sha={
            "b" * 64: {"cv_count": 0, "track_count": 99, "matched_track_count": 0}
        },
    )
    assert [x["capture_sha256"] for x in batch["selected"]] == ["c" * 64]


def test_track_center_and_time_agreement_are_both_required():
    cv = [{"sample_start": 100, "sample_count": 500,
           "frequency_low_hz": 950.0, "frequency_high_hz": 1050.0}]
    tracks = [
        {"sample_start": 200, "sample_count": 100, "center_hz": 990.0},
        {"sample_start": 900, "sample_count": 100, "center_hz": 990.0},
        {"sample_start": 200, "sample_count": 100, "center_hz": 1200.0},
    ]
    assert matched_track_count(cv, tracks) == 1
    assert matched_track_count([], tracks) == 0


def test_invalid_evidence_never_makes_negative_label():
    c = _capture("one", "d" * 64)
    batch = rank_review_batch(
        [c], EmptyStore(), excluded_sha=set(), limit=1,
        proposal_evidence_by_sha={c["sha256"]: {
            "cv_count": "unknown", "track_count": -1, "matched_track_count": 0,
        }},
    )
    assert "possible-negative-control-unverified" not in batch["selected"][0]["priority_reasons"]
    assert batch["selected"][0]["annotation_status"] == "unannotated"
    assert batch["training_eligible"] is False


def test_second_pass_never_screens_a_frozen_session(tmp_path, monkeypatch):
    import json
    from tools import score_active_learning_queue as script

    protected = _capture("heldout", "a" * 64)
    same_session = {**_capture("heldout", "b" * 64), "id": "second-heldout"}
    allowed = _capture("eligible", "c" * 64)
    seen = []

    class FakeStore(EmptyStore):
        def __init__(self, _path):
            pass

        def list_captures(self):
            return [protected, same_session, allowed]

    class FakeDetector:
        model_sha256 = "f" * 64

        def __init__(self, _path):
            pass

    def fake_screen(capture, _detector):
        seen.append(capture["sha256"])
        return {"cv_count": 0, "track_count": 0,
                "matched_track_count": 0, "proposal_only": True}

    monkeypatch.setattr(script, "CaptureStore", FakeStore)
    monkeypatch.setattr("sigid.cv_detector.YoloSignalDetector", FakeDetector)
    monkeypatch.setattr(script, "screen_capture", fake_screen)
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps({
        "schema_version": 1, "excluded_capture_sha256": ["a" * 64],
        "frozen_holdout_capture_sha256": ["a" * 64],
    }))
    model = tmp_path / "baseline.pt"
    model.write_bytes(b"mock")
    result = script.prepare_screened_batch(
        tmp_path, exclusions, model, scan_limit=5, limit=5
    )
    assert seen == ["c" * 64]
    assert result["proposal_screening"]["screened_count"] == 1
    batch = json.loads((tmp_path / "active-learning-review-batch.json").read_text())
    assert [r["capture_sha256"] for r in batch["selected"]] == ["c" * 64]
    assert batch["training_eligible"] is False
