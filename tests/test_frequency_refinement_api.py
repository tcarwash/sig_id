"""API exposure of opt-in frequency refinement; no decoder side effects."""

from __future__ import annotations

import asyncio
import wave

import httpx
import numpy as np

from sigid.api import create_app


def _wav_fixture(path, rate=12000):
    rng = np.random.default_rng(22)
    n = 14 * rate
    t = np.arange(n)

    def hopping(low, seed):
        tones = np.random.default_rng(seed).integers(0, 8, 100)
        freq = low + tones[(t // round(rate * 0.16)).clip(max=99)] * 6.25
        return np.sin(np.cumsum(freq) * 2 * np.pi / rate)

    audio = 0.13 * hopping(1000, 1) + 0.10 * hopping(1075, 2) + rng.normal(0, 0.003, n)
    pcm = np.round(audio * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    return pcm


class FakeCV:
    model_sha256 = "c" * 64
    model_version = "sha256:" + model_sha256
    confidence_threshold = 0.5

    @staticmethod
    def detection(n):
        return {
            "sample_start": 0, "sample_count": n,
            "frequency_low_hz": 980.0, "frequency_high_hz": 1140.0,
            "confidence": 0.95,
            "model_sha256": FakeCV.model_sha256,
            "model_version": FakeCV.model_version,
            "window_sample_start": 0, "window_sample_count": n,
        }

    def detect_capture(self, path, *, confidence=None):
        with wave.open(str(path), "rb") as wav:
            return [self.detection(wav.getnframes())]

    def detect_live_partial(self, audio, *, sample_start, total_sample_count, sample_rate, confidence=None):
        return [self.detection(total_sample_count)]


def test_opt_in_refinement_for_live_and_saved_cv_boxes(tmp_path, monkeypatch):
    wav_file = tmp_path / "two-stations.wav"
    pcm = _wav_fixture(wav_file)
    app = create_app(tmp_path / "data")
    app.state.cv_detector = FakeCV()
    app.state.classifier = None
    snapshot = {
        "pcm": pcm.tobytes(), "sample_rate": 12000, "channels": 1,
        "sample_offset": 0, "sample_count": len(pcm),
        "window_sample_count": len(pcm), "session_id": "crowded-live",
        "active": True,
    }
    app.state.live_capture.analysis_audio = lambda: snapshot
    monkeypatch.setattr(app.state.live_capture, "status", lambda: {
        "active": True, "session_id": "crowded-live",
    })

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            live = await client.get("/api/live/cv-detections", params={
                "session_id": "crowded-live", "include_classifier": False,
            })
            assert live.status_code == 200, live.text
            assert "frequency_refinement" not in live.json()["detections"][0]
            assert live.json()["automatic_trial"] is None

            live_refined = await client.get("/api/live/cv-detections", params={
                "session_id": "crowded-live", "include_classifier": False,
                "include_frequency_refinement": True,
            })
            assert live_refined.status_code == 200, live_refined.text
            assert live_refined.json()["frequency_refinement_enabled"] is True
            assert live_refined.json()["frequency_refinement_suggestions"] == 1
            refined = live_refined.json()["detections"][0]["frequency_refinement"]
            assert refined["split_suggested"] is True
            assert len(refined["subbands"]) == 2
            assert live_refined.json()["automatic_trial"] is None
            assert app.state.capture_store.decoder_runs(session_id="crowded-live") == []

            uploaded = await client.post("/api/captures", files={
                "file": ("two-stations.wav", wav_file.read_bytes(), "audio/wav"),
            })
            assert uploaded.status_code == 200
            capture_id = uploaded.json()["id"]
            saved = await client.post(f"/api/captures/{capture_id}/cv-detections", params={
                "include_frequency_refinement": True,
            })
            assert saved.status_code == 200, saved.text
            assert saved.json()["frequency_refinement_suggestions"] == 1
            assert saved.json()["detections"][0]["frequency_refinement"]["split_suggested"]
            assert len(app.state.capture_store.regions(capture_id)) == 0

    asyncio.run(check())
