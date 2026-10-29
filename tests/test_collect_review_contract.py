"""Integration contracts between attempt collection and window review."""
from test_collect_decode_api import setup_saved, make_run, request
import pytest


def test_collection_review_completion_applies_to_target_window_not_whole_source(tmp_path):
    app, store, capture_id, _ = setup_saved(tmp_path)
    run_id = make_run(store, capture_id=capture_id, sample_start=96000, sample_count=16000)
    result = request(app, "POST", f"/api/captures/{capture_id}/decode-runs/{run_id}/collect", json={}).json()
    collected = result["capture_id"]
    response = request(app, "POST", f"/api/captures/{collected}/coverage", json={
        "sample_start": 80000, "sample_count": 80000, "frequency_low_hz": 0,
        "frequency_high_hz": 3000, "task": "detection", "taxonomy_scope": [], "state": "complete",
        "ignore_regions": [], "uncertain_regions": [], "notes": "Reviewed only this window",
    })
    assert response.status_code == 201, response.text
    assert all(item["id"] != collected for item in request(app, "GET", "/api/live/review-queue").json()["pending"])
    assert store.scene_coverages(capture_id) == []


@pytest.mark.parametrize("scope,association", [
    ("isolated_native_window", "exact_selected_capture_window"),
    ("isolated_live_mode_sweep", "exact_live_track_window"),
])
def test_collection_supports_other_exact_decoder_attempts(tmp_path, monkeypatch, scope, association):
    app, store, capture_id, _ = setup_saved(tmp_path)
    live = scope == "isolated_live_mode_sweep"
    run_id = make_run(store, capture_id=None if live else capture_id,
                      sample_count=16000, output_scope=scope, association=association,
                      status="tentative_output" if live else "completed")
    if live:
        monkeypatch.setattr(app.state.live_capture, "analysis_audio", lambda: {
            "active": True, "session_id": "session-one", "sample_rate": 8000,
            "channels": 1, "sample_offset": 8000, "window_sample_count": 80000,
            "pcm": b"\x00\x00" * 80000,
        })
    prefix = "/api/live/sessions/session-one" if live else f"/api/captures/{capture_id}"
    response = request(app, "POST", f"{prefix}/decode-runs/{run_id}/collect", json={})
    assert response.status_code == 201, response.text
    region = store.regions(response.json()["capture_id"])[0]
    assert region["review_state"] == "pending"
    assert region["evidence"]["decoder_run"]["output_scope"] == scope


def test_manual_decode_keeps_selected_frequency_bounds_for_later_collection(tmp_path):
    from sigid.api import create_app
    from test_fldigi_replay_api import FakeFldigi, FakeReplay
    import time
    _, store, capture_id, _ = setup_saved(tmp_path)
    app = create_app(tmp_path / "data", fldigi_client=FakeFldigi(), fldigi_replay_backend=FakeReplay())
    result = request(app, "POST", f"/api/captures/{capture_id}/decode", json={
        "mode": "RTTY", "sample_start": 1000, "sample_count": 8000,
        "frequency_low_hz": 1100, "frequency_high_hz": 1400,
    })
    assert result.status_code == 202, result.text
    run_id = result.json()["run_id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        run = next(item for item in store.decoder_runs(capture_id=capture_id) if item["id"] == run_id)
        if run["status"] not in {"queued", "running"}:
            break
        time.sleep(.01)
    collected = request(app, "POST", f"/api/captures/{capture_id}/decode-runs/{run_id}/collect", json={})
    assert collected.status_code == 201, collected.text
    [region] = store.regions(collected.json()["capture_id"])
    assert (region["frequency_low_hz"], region["frequency_high_hz"]) == (1100, 1400)
