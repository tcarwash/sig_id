"""Regression: keyed narrow tones remain detectable independent of CV/ML."""

from __future__ import annotations

import asyncio

import httpx
import numpy as np

from sigid.api import create_app
from sigid.detection import detect_tracks
from sigid.spectrum import waterfall_samples


def keyed_tone_audio(rate: int = 12000, seconds: float = 12.0) -> np.ndarray:
    """Synthetic on/off keyed carrier, not a valid Morse message or decode."""
    count = round(rate * seconds)
    positions = np.arange(count)
    tone = np.sin(2 * np.pi * 1100.0 * positions / rate)
    gate = np.zeros(count)
    cursor = int(0.3 * rate)
    for units in [1, 1, 3, 1, 1, 3, 1, 3, 1, 1, 3, 1, 1, 3, 3, 1]:
        end = min(count, cursor + int(units * 0.08 * rate))
        gate[cursor:end] = 1
        cursor = end + int(0.08 * rate)
    return 0.35 * gate * tone + np.random.default_rng(8).normal(0, 0.001, count)


def test_original_narrow_tone_detector_recovers_keyed_carrier():
    rate = 12000
    audio = keyed_tone_audio(rate)
    spectrum = waterfall_samples(
        audio, rate, fft_size=4096, hop_size=512, max_rows=320,
        sample_offset=0, total_sample_count=len(audio),
    )
    tracks = detect_tracks(spectrum)
    narrow = [row for row in tracks if row["tone_count_candidate"] == 1
              and row["bandwidth_hz"] <= 45
              and row["peak_snr_db"] >= 14]
    assert narrow, tracks
    assert any(abs(row["center_hz"] - 1100) <= 25 for row in narrow)
    assert all(row["identification_state"] == "provisional" and row["protocol"] is None
               for row in narrow)
    assert detect_tracks(waterfall_samples(
        np.zeros_like(audio), rate, fft_size=4096, hop_size=512,
        max_rows=320, sample_offset=0, total_sample_count=len(audio),
    )) == []



def test_narrow_grouping_keeps_keyed_carrier_separate_from_nearby_tone():
    rate = 12000
    audio = keyed_tone_audio(rate)
    t = np.arange(audio.size) / rate
    audio += 0.16 * np.sin(2 * np.pi * 1290 * t)
    spectrum = waterfall_samples(
        audio, rate, fft_size=4096, hop_size=512, max_rows=320,
        sample_offset=0, total_sample_count=audio.size,
    )
    full = detect_tracks(spectrum)
    narrow = detect_tracks(spectrum, maximum_tone_gap_hz=35)
    assert any(row["tone_count_candidate"] >= 2 for row in full)
    assert any(row["tone_count_candidate"] == 1
               and abs(row["center_hz"] - 1100) <= 25 for row in narrow)
    assert any(row["tone_count_candidate"] == 1
               and abs(row["center_hz"] - 1290) <= 25 for row in narrow)
    assert all(row["protocol"] is None for row in narrow)


def test_live_analysis_exposes_tone_tracks_when_classifier_disabled(tmp_path, monkeypatch):
    rate = 12000
    pcm = np.round(np.clip(keyed_tone_audio(rate), -1, 1) * 32767).astype("<i2")
    app = create_app(tmp_path / "data")
    snapshot = {
        "pcm": pcm.tobytes(), "sample_rate": rate, "channels": 1,
        "sample_offset": 0, "sample_count": len(pcm),
        "window_sample_count": len(pcm), "session_id": "cw-regression",
        "active": True,
    }
    monkeypatch.setattr(app.state.live_capture, "analysis_audio", lambda: snapshot)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/live/analysis", params={
                "fft_size": 4096, "hop_size": 512, "max_rows": 320,
                "include_spectrum": False,
                "include_tracks": True,
                "include_classifier": False,
                "auto_trial": False,
            })
            assert response.status_code == 200, response.text
            tracks = response.json()["tracks"]
            assert any(row["tone_count_candidate"] == 1
                       and abs(row["center_hz"] - 1100) <= 25 for row in tracks)
            assert all("model_proposal" not in row for row in tracks)
            assert response.json().get("automatic_trial") is None

    asyncio.run(exercise())
