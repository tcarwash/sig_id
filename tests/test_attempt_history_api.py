"""Persisted per-source decoder attempts expose their saved settings and results."""
import asyncio
import wave

import httpx

from sigid.api import create_app


def _request(app, path):
    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.get(path)
    return asyncio.run(send())


def _make_run(store, *, capture_id, mode, sample_start, text, auto_identify_note):
    run_id = store.create_decoder_run(
        session_id=f"saved-capture:{capture_id}", capture_id=capture_id,
        track_id=f"capture:{capture_id}:samples:{sample_start}:8000",
        sample_start=sample_start, sample_count=8000, sample_rate=8000,
        modulation_hypothesis="unknown", backend="fldigi-isolated-replay",
        modes=[mode], attempted_modes=[mode], current_mode=mode,
        output_offset=0, status="completed", note=auto_identify_note,
        carrier_hz=1200.0, output_scope="isolated_region_replay",
        association="exact_capture_window",
        classifier_evidence={"model_sha256": "abc123"},
        detector_evidence={"selection_bounds": {
            "source": "operator_selection", "frequency_low_hz": 400.0,
            "frequency_high_hz": 1800.0,
        }},
    )
    store.update_decoder_run(run_id, {
        "attempted_modes": [mode], "current_mode": mode, "status": "completed",
        "text": text, "output_end": len(text), "note": auto_identify_note,
    })
    return run_id


def test_capture_history_returns_each_attempt_settings_and_result_in_order(tmp_path):
    app = create_app(tmp_path / "data")
    store = app.state.capture_store
    audio = tmp_path / "source.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x00\x00" * 24000)
    capture_id = store.add_wav_capture(audio)
    earlier = _make_run(store, capture_id=capture_id, mode="RTTY", sample_start=1000,
                        text="FIRST", auto_identify_note="Selected single mode.")
    later = _make_run(store, capture_id=capture_id, mode="BPSK31", sample_start=9000,
                      text="SECOND", auto_identify_note="Candidate sweep.")

    response = _request(app, f"/api/captures/{capture_id}/decode-runs")

    assert response.status_code == 200
    attempts = response.json()
    assert [attempt["id"] for attempt in attempts] == [earlier, later]
    assert [(attempt["sample_start"], attempt["sample_count"], attempt["sample_rate"])
            for attempt in attempts] == [(1000, 8000, 8000), (9000, 8000, 8000)]
    assert [(attempt["modes"], attempt["attempted_modes"], attempt["current_mode"])
            for attempt in attempts] == [(["RTTY"], ["RTTY"], "RTTY"), (["BPSK31"], ["BPSK31"], "BPSK31")]
    assert [(attempt["carrier_hz"], attempt["modulation_hypothesis"], attempt["backend"])
            for attempt in attempts] == [(1200.0, "unknown", "fldigi-isolated-replay")] * 2
    assert [attempt["output_text"] for attempt in attempts] == ["FIRST", "SECOND"]
    assert all(attempt["status"] == "completed" for attempt in attempts)
    assert all(attempt["output_scope"] == "isolated_region_replay"
               and attempt["association"] == "exact_capture_window" for attempt in attempts)
    assert attempts[0]["classifier_evidence"] == {"model_sha256": "abc123"}
    assert attempts[0]["detector_evidence"]["selection_bounds"] == {
        "source": "operator_selection", "frequency_low_hz": 400.0,
        "frequency_high_hz": 1800.0,
    }
    assert all("created_utc" in attempt and "updated_utc" in attempt for attempt in attempts)
    # The durable record does not store the original auto_identify toggle. A note
    # is not an exact substitute, so consumers must not claim exact replayability.
    assert all("auto_identify" not in attempt for attempt in attempts)


def test_live_session_history_returns_attempts_after_the_live_buffer_is_gone(tmp_path):
    app = create_app(tmp_path / "data")
    store = app.state.capture_store
    run_id = store.create_decoder_run(
        session_id="session-archived", track_id="session-archived:track-7",
        sample_start=32000, sample_count=8000, sample_rate=8000,
        modulation_hypothesis="FSK", backend="fldigi-isolated-replay",
        modes=["RTTY", "BPSK31"], attempted_modes=["RTTY"], current_mode="RTTY",
        output_offset=0, status="completed", note="Saved isolated attempt",
        carrier_hz=1500.0, output_scope="isolated_live_window_replay",
        association="exact_live_track_window",
        classifier_evidence={"candidate": "FSK"},
        detector_evidence={"frequency_low_hz": 500.0, "frequency_high_hz": 2200.0},
    )
    store.update_decoder_run(run_id, {
        "attempted_modes": ["RTTY"], "current_mode": "RTTY", "status": "completed",
        "text": "ARCHIVED LIVE RESULT", "output_end": 20, "note": "Completed",
    })

    response = _request(app, "/api/live/sessions/session-archived/decode-runs")

    assert response.status_code == 200
    [attempt] = response.json()
    assert attempt["id"] == run_id
    assert (attempt["sample_start"], attempt["sample_count"], attempt["sample_rate"]) == (32000, 8000, 8000)
    assert attempt["modes"] == ["RTTY", "BPSK31"]
    assert attempt["attempted_modes"] == ["RTTY"]
    assert attempt["carrier_hz"] == 1500.0
    assert attempt["output_text"] == "ARCHIVED LIVE RESULT"
    assert attempt["output_scope"] == "isolated_live_window_replay"
    assert attempt["association"] == "exact_live_track_window"
