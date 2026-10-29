"""Collection tests use synthetic audio and persisted runs, never radio hardware."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import wave
from pathlib import Path
from threading import Event

import httpx
import pytest

from sigid.api import create_app
from sigid.dataset import SignalLabel


def setup_saved(tmp_path):
    app = create_app(tmp_path / "data")
    store = app.state.capture_store
    path = tmp_path / "source.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x01\x00" * 160000)
    capture_id = store.add_wav_capture(path)
    region_id = store.add_region(capture_id, sample_start=12000, sample_count=96000,
                                 label=SignalLabel(source="human", review_state="accepted",
                                                   evidence={"original": True}))
    run_id = make_run(store, capture_id=capture_id, region_id=region_id)
    return app, store, capture_id, run_id


def make_run(store, *, capture_id=None, region_id=None, session_id="session-one", **overrides):
    settings = dict(session_id=session_id, capture_id=capture_id, region_id=region_id,
                    track_id="track-one", sample_start=12000, sample_count=96000,
                    sample_rate=8000, modulation_hypothesis="unknown",
                    backend="fldigi-isolated-replay", modes=["RTTY"], attempted_modes=["RTTY"],
                    current_mode="RTTY", carrier_hz=1234, output_offset=0,
                    status="completed", note="Unvalidated text",
                    output_scope="isolated_region_replay" if capture_id else "isolated_live_window_replay",
                    association="exact_capture_window" if capture_id else "exact_live_track_window",
                    classifier_evidence={"candidate": "FSK"},
                    detector_evidence={"frequency_low_hz": 1000, "frequency_high_hz": 1500})
    settings.update(overrides)
    run_id = store.create_decoder_run(**settings)
    store.update_decoder_run(run_id, {"attempted_modes": ["RTTY"], "current_mode": "RTTY",
                                      "status": settings["status"], "text": "EARLIER RESULT",
                                      "output_end": 14, "note": "Unvalidated text"})
    return run_id


def request(app, method, url, **kwargs):
    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.request(method, url, **kwargs)
    return asyncio.run(send())


def test_saved_collect_preserves_selected_run_without_accepting_or_copying_audio(tmp_path):
    app, store, source_id, run_id = setup_saved(tmp_path)
    before = store.regions(source_id)
    selected_run = store.decoder_runs(capture_id=source_id)[0]
    make_run(store, capture_id=source_id, current_mode="BPSK31")
    response = request(app, "POST", f"/api/captures/{source_id}/decode-runs/{run_id}/collect",
                       json={"notes": "Keep the earlier attempt"})
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["run_id"] == run_id
    capture = store.capture(result["capture_id"])
    assert capture["id"] != source_id
    assert Path(capture["audio_path"]).stat().st_ino == Path(store.capture(source_id)["audio_path"]).stat().st_ino
    assert capture["source_metadata"]["source_capture_id"] == source_id
    assert capture["source_metadata"]["decoder_run"] == selected_run
    [region] = store.regions(capture["id"])
    assert region["id"] == result["region_id"]
    assert (region["sample_start"], region["sample_count"]) == (12000, 96000)
    assert (region["frequency_low_hz"], region["frequency_high_hz"]) == (1000, 1500)
    assert region["review_state"] == "pending"
    assert region["mode_id"] is None
    assert region["evidence"]["decoder_run"] == selected_run
    assert region["notes"] == "Keep the earlier attempt"
    assert store.scene_coverages(capture["id"]) == []
    assert store.regions(source_id) == before
    assert [item["id"] for item in request(app, "GET", "/api/live/review-queue").json()["pending"]] == [capture["id"]]


def test_collection_retries_are_durable_and_concurrent_requests_dedupe(tmp_path):
    app, store, source_id, run_id = setup_saved(tmp_path)
    url = f"/api/captures/{source_id}/decode-runs/{run_id}/collect"
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: request(app, "POST", url, json={}), range(2)))
    assert all(response.status_code == 201 for response in responses)
    assert len({response.json()["capture_id"] for response in responses}) == 1
    assert sorted(response.json()["deduplicated"] for response in responses) == [False, True]
    restarted = create_app(tmp_path / "data")
    repeated = request(restarted, "POST", url, json={"notes": "retry does not modify evidence"})
    assert repeated.json()["capture_id"] == responses[0].json()["capture_id"]
    assert repeated.json()["deduplicated"] is True
    assert len(store.list_captures()) == 2


def test_live_collect_saves_exact_snapshot_and_maps_source_bounds(tmp_path, monkeypatch):
    app = create_app(tmp_path / "data")
    store = app.state.capture_store
    run_id = make_run(store, sample_count=16000)
    snapshot = {"active": True, "session_id": "session-one", "sample_rate": 8000,
                "channels": 1, "sample_offset": 8000, "window_sample_count": 80000,
                "pcm": b"\x13\x00" * 80000}
    monkeypatch.setattr(app.state.live_capture, "analysis_audio", lambda: snapshot)
    url = f"/api/live/sessions/session-one/decode-runs/{run_id}/collect"
    response = request(app, "POST", url, json={})
    assert response.status_code == 201, response.text
    result = response.json()
    capture = store.capture(result["capture_id"])
    [region] = store.regions(capture["id"])
    assert (region["sample_start"], region["sample_count"]) == (4000, 16000)
    assert region["review_state"] == "pending"
    assert region["evidence"]["decoder_run"]["sample_start"] == 12000
    assert capture["source_metadata"]["window_start_sample"] == 8000
    assert capture["source_metadata"]["decoder_run"] == store.decoder_runs(session_id="session-one")[0]
    assert store.scene_coverages(capture["id"]) == []
    with wave.open(capture["audio_path"], "rb") as wav:
        assert wav.readframes(wav.getnframes()) == snapshot["pcm"]
    monkeypatch.setattr(app.state.live_capture, "analysis_audio", lambda: None)
    retry = request(app, "POST", url, json={}).json()
    assert retry["capture_id"] == capture["id"]
    assert retry["deduplicated"] is True


@pytest.mark.parametrize("overrides,status", [
    ({"status": "queued"}, 409),
    ({"status": "running"}, 409),
    ({"output_scope": "shared_live_input", "association": "track_window_tentative"}, 422),
    ({"sample_rate": 16000}, 422),
    ({"association": "track_window_tentative"}, 422),
    ({"output_scope": "shared_live_input"}, 422),
    ({"sample_start": None, "sample_count": None}, 422),

    ({"sample_count": 160000}, 422),
])
def test_collect_rejects_unstable_or_inexact_saved_evidence(tmp_path, overrides, status):
    app, store, capture_id, _ = setup_saved(tmp_path)
    run_id = make_run(store, capture_id=capture_id, **overrides)
    response = request(app, "POST", f"/api/captures/{capture_id}/decode-runs/{run_id}/collect", json={})
    assert response.status_code == status, response.text
    assert len(store.list_captures()) == 1
    assert list((tmp_path / "data" / "captures").glob("*.wav")) == []


def test_collection_points_review_at_attempt_window_not_source_start(tmp_path):
    app, store, capture_id, _ = setup_saved(tmp_path)
    run_id = make_run(store, capture_id=capture_id, sample_start=96000, sample_count=16000)
    response = request(app, "POST", f"/api/captures/{capture_id}/decode-runs/{run_id}/collect", json={})
    assert response.status_code == 201, response.text
    capture = store.capture(response.json()["capture_id"])
    assert capture["sample_count"] == 160000  # Retain surrounding source, not a truncated decode.
    assert capture["source_metadata"]["review_window_start_sample"] == 80000
    assert capture["source_metadata"]["review_window_sample_count"] == 80000
    assert capture["source_metadata"]["window_start_sample"] == 0
    assert store.regions(capture["id"])[0]["sample_start"] == 96000


@pytest.mark.parametrize("overrides,snapshot_changes,status", [
    ({"status": "queued"}, {}, 409),
    ({"status": "running"}, {}, 409),
    ({"output_scope": "shared_live_input"}, {}, 422),
    ({"association": "track_window_tentative"}, {}, 422),
    ({"sample_rate": 16000}, {}, 422),
    ({"sample_start": 7999}, {}, 422),
    ({"sample_start": 72001}, {}, 422),
    ({}, {"active": False}, 409),
    ({}, {"session_id": "another-session"}, 409),
    ({}, {"pcm": b"\x00\x00" * 10000}, 422),
])
def test_live_collect_rejects_unstable_expired_or_mismatched_audio(tmp_path, monkeypatch, overrides, snapshot_changes, status):
    app = create_app(tmp_path / "data")
    store = app.state.capture_store
    run_id = make_run(store, sample_count=16000, **overrides)
    snapshot = {"active": True, "session_id": "session-one", "sample_rate": 8000,
                "channels": 1, "sample_offset": 8000, "window_sample_count": 80000,
                "pcm": b"\x13\x00" * 80000}
    snapshot.update(snapshot_changes)
    monkeypatch.setattr(app.state.live_capture, "analysis_audio", lambda: snapshot)
    response = request(app, "POST", f"/api/live/sessions/session-one/decode-runs/{run_id}/collect", json={})
    assert response.status_code == status, response.text
    assert store.list_captures() == []
    assert list((tmp_path / "data" / "captures").glob("*.wav")) == []


@pytest.mark.parametrize("target", ["missing-capture", "wrong-capture", "wrong-run", "wrong-session", "missing-audio"])
def test_collection_rejects_missing_source_or_wrong_run(tmp_path, target):
    app, store, capture_id, run_id = setup_saved(tmp_path)
    if target == "wrong-capture":
        capture_id = store.add_wav_capture(store.capture(capture_id)["audio_path"])
    elif target == "missing-capture":
        capture_id = "does-not-exist"
    elif target == "wrong-run":
        run_id = "does-not-exist"
    elif target == "missing-audio":
        Path(store.capture(capture_id)["audio_path"]).unlink()
    url = (f"/api/live/sessions/another-session/decode-runs/{run_id}/collect" if target == "wrong-session"
           else f"/api/captures/{capture_id}/decode-runs/{run_id}/collect")
    before = store.list_captures()
    response = request(app, "POST", url, json={})
    assert response.status_code == 404, response.text
    assert store.list_captures() == before
    assert list((tmp_path / "data" / "captures").glob("*.wav")) == []


def test_failed_collection_removes_partial_copy_without_touching_source(tmp_path, monkeypatch):
    app, store, capture_id, run_id = setup_saved(tmp_path)
    source = Path(store.capture(capture_id)["audio_path"])
    original = source.read_bytes()

    def no_hardlink(*args):
        raise OSError("cross-device link")

    def interrupted_copy(source_path, target):
        Path(target).write_bytes(b"partial")
        raise OSError("out of space")

    monkeypatch.setattr("sigid.api.os.link", no_hardlink)
    monkeypatch.setattr("sigid.api.shutil.copyfile", interrupted_copy)
    response = request(app, "POST", f"/api/captures/{capture_id}/decode-runs/{run_id}/collect", json={})
    assert response.status_code == 422, response.text
    assert len(store.list_captures()) == 1
    assert source.read_bytes() == original
    assert list((tmp_path / "data" / "captures").glob("*.wav")) == []


@pytest.mark.parametrize("bounds", [
    {"frequency_low_hz": -1, "frequency_high_hz": 1500},
    {"frequency_low_hz": 1000, "frequency_high_hz": 4001},
    {"frequency_low_hz": 1500, "frequency_high_hz": 1000},
    {"frequency_low_hz": 1000},
    {"frequency_low_hz": 1000, "frequency_high_hz": float("inf")},
    {"frequency_low_hz": float("nan"), "frequency_high_hz": 1500},
])
def test_collection_rejects_invalid_frequency_evidence(tmp_path, bounds):
    app, store, capture_id, _ = setup_saved(tmp_path)
    run_id = make_run(store, capture_id=capture_id, detector_evidence=bounds)
    response = request(app, "POST", f"/api/captures/{capture_id}/decode-runs/{run_id}/collect", json={})
    assert response.status_code == 422, response.text
    assert len(store.list_captures()) == 1
    assert list((tmp_path / "data" / "captures").glob("*.wav")) == []


def test_collection_revalidates_actual_source_rate(tmp_path):
    app, store, capture_id, run_id = setup_saved(tmp_path)
    with wave.open(store.capture(capture_id)["audio_path"], "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x01\x00" * 160000)
    response = request(app, "POST", f"/api/captures/{capture_id}/decode-runs/{run_id}/collect", json={})
    assert response.status_code == 422, response.text
    assert len(store.list_captures()) == 1
    assert list((tmp_path / "data" / "captures").glob("*.wav")) == []


def test_collection_deduplicates_across_app_instances(tmp_path, monkeypatch):
    app, store, capture_id, run_id = setup_saved(tmp_path)
    second_app = create_app(tmp_path / "data")
    second_store = second_app.state.capture_store
    creating = Event()
    second_scanned = Event()
    add_capture = store.add_wav_capture
    list_captures = second_store.list_captures

    def delayed_add(*args, **kwargs):
        creating.set()
        second_scanned.wait(timeout=0.25)
        return add_capture(*args, **kwargs)

    def observed_scan():
        result = list_captures()
        second_scanned.set()
        return result

    monkeypatch.setattr(store, "add_wav_capture", delayed_add)
    monkeypatch.setattr(second_store, "list_captures", observed_scan)
    url = f"/api/captures/{capture_id}/decode-runs/{run_id}/collect"
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(request, app, "POST", url, json={})
        assert creating.wait(timeout=5)
        second = pool.submit(request, second_app, "POST", url, json={})
        responses = [first.result(timeout=10), second.result(timeout=10)]
    assert all(response.status_code == 201 for response in responses)
    assert len({response.json()["capture_id"] for response in responses}) == 1
    assert sorted(response.json()["deduplicated"] for response in responses) == [False, True]
    assert len(store.list_captures()) == 2


def test_failed_region_insert_cleans_capture_and_hardlink(tmp_path, monkeypatch):
    app, store, capture_id, run_id = setup_saved(tmp_path)
    source = Path(store.capture(capture_id)["audio_path"])
    original = source.read_bytes()

    def fail_region(*args, **kwargs):
        raise ValueError("injected region failure")

    monkeypatch.setattr(store, "add_region", fail_region)
    response = request(app, "POST", f"/api/captures/{capture_id}/decode-runs/{run_id}/collect", json={})
    assert response.status_code == 422
    assert len(store.list_captures()) == 1
    assert source.read_bytes() == original
    assert list((tmp_path / "data" / "captures").glob("*.wav")) == []


@pytest.mark.parametrize("fallback_copy", [False, True])
def test_deleting_collected_reference_preserves_source_audio(tmp_path, monkeypatch, fallback_copy):
    app, store, capture_id, run_id = setup_saved(tmp_path)
    source = Path(store.capture(capture_id)["audio_path"])
    original = source.read_bytes()
    if fallback_copy:
        def no_hardlink(*args):
            raise OSError("cross-device link")
        monkeypatch.setattr("sigid.api.os.link", no_hardlink)
    result = request(app, "POST", f"/api/captures/{capture_id}/decode-runs/{run_id}/collect", json={})
    assert result.status_code == 201, result.text
    collected_id = result.json()["capture_id"]
    collected = Path(store.capture(collected_id)["audio_path"])
    assert collected.read_bytes() == original
    assert (collected.stat().st_ino != source.stat().st_ino) is fallback_copy
    assert request(app, "DELETE", f"/api/captures/{collected_id}").status_code == 200
    assert source.read_bytes() == original
    assert not collected.exists()
    assert request(app, "GET", f"/api/captures/{collected_id}").status_code == 404
    assert store.scene_coverages(capture_id) == []


def test_failed_attempt_without_frequency_evidence_still_queues_pending_review(tmp_path):
    app, store, capture_id, _ = setup_saved(tmp_path)
    run_id = make_run(store, capture_id=capture_id, status="failed", detector_evidence={})
    response = request(app, "POST", f"/api/captures/{capture_id}/decode-runs/{run_id}/collect", json={})
    assert response.status_code == 201, response.text
    collected_id = response.json()["capture_id"]
    [region] = store.regions(collected_id)
    assert region["review_state"] == "pending"
    assert region["mode_id"] is None
    assert region["frequency_low_hz"] is None
    assert region["frequency_high_hz"] is None
    assert region["evidence"]["decoder_run"]["status"] == "failed"
    assert store.scene_coverages(collected_id) == []


def test_retry_does_not_claim_missing_collected_audio_is_durable(tmp_path):
    app, store, capture_id, run_id = setup_saved(tmp_path)
    url = f"/api/captures/{capture_id}/decode-runs/{run_id}/collect"
    result = request(app, "POST", url, json={}).json()
    Path(store.capture(result["capture_id"])["audio_path"]).unlink()
    response = request(app, "POST", url, json={})
    assert response.status_code == 409, response.text
    assert len(store.list_captures()) == 2
