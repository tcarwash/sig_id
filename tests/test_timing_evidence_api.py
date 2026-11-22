"""Read-only live/saved DSP timing evidence API regression tests."""

from __future__ import annotations

import asyncio
import wave

import httpx
import numpy as np
import pytest

from sigid.api import create_app


def _audio_fixture(path, *, rate: int = 12000, duration: float = 12.8) -> np.ndarray:
    samples = round(rate * duration)
    rng = np.random.default_rng(33)
    tones = rng.integers(0, 8, 100)
    offset = np.arange(samples)
    frequencies = 1100.0 + 6.25 * tones[(offset // round(rate * 0.16)).clip(max=99)]
    signal = 0.15 * np.sin(2 * np.pi * np.cumsum(frequencies) / rate)
    pcm = np.round(np.clip(signal, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(pcm.tobytes())
    return pcm


def test_inspecting_selected_live_and_saved_roi_is_read_only(tmp_path, monkeypatch):
    wav = tmp_path / "hopping.wav"
    pcm = _audio_fixture(wav)
    app = create_app(tmp_path / "state")
    snapshot = {
        "pcm": pcm.tobytes(), "channels": 1, "sample_rate": 12000,
        "sample_offset": 0, "sample_count": len(pcm),
        "window_sample_count": len(pcm), "active": True, "session_id": "timing-a",
    }
    monkeypatch.setattr(app.state.live_capture, "analysis_audio", lambda: snapshot)
    query = {
        "sample_start": 0, "sample_count": len(pcm),
        "frequency_low_hz": 1085.0, "frequency_high_hz": 1160.0,
    }

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/live/dsp-evidence", params={"session_id": "timing-a", **query}
            )
            assert response.status_code == 200, response.text
            data = response.json()
            assert data["identity_status"] == "unverified"
            assert data["decoder_invoked"] is False
            assert data["timing_evidence"]["symbol_rate_baud_candidate"] == pytest.approx(6.25, abs=0.25)
            assert data["timing_evidence"]["tone_spacing_hz_candidate"] == pytest.approx(6.25, abs=1.0)
            assert data["spectral_evidence"]["available"]

            uploaded = await client.post(
                "/api/captures",
                files={"file": ("hopping.wav", wav.read_bytes(), "audio/wav")},
            )
            assert uploaded.status_code == 200, uploaded.text
            capture_id = uploaded.json()["id"]
            saved = await client.get(f"/api/captures/{capture_id}/dsp-evidence", params=query)
            assert saved.status_code == 200, saved.text
            assert saved.json()["identity_status"] == "unverified"
            assert saved.json()["timing_evidence"]["symbol_rate_baud_candidate"] == pytest.approx(6.25, abs=0.25)
            assert app.state.capture_store.regions(capture_id) == []
            assert app.state.capture_store.decoder_runs(session_id="timing-a") == []

            wrong_session = await client.get(
                "/api/live/dsp-evidence", params={"session_id": "timing-b", **query}
            )
            assert wrong_session.status_code == 409
            stale = await client.get(
                "/api/live/dsp-evidence", params={
                    "session_id": "timing-a", **query, "sample_start": len(pcm) + 1200
                }
            )
            assert stale.status_code == 409
            invalid = await client.get(
                f"/api/captures/{capture_id}/dsp-evidence",
                params={**query, "frequency_low_hz": 2500, "frequency_high_hz": 1200},
            )
            assert invalid.status_code == 422
            too_far = await client.get(
                f"/api/captures/{capture_id}/dsp-evidence",
                params={**query, "sample_count": len(pcm) + 1},
            )
            assert too_far.status_code == 422

    asyncio.run(exercise())
