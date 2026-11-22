"""Classifier cache must be bounded, sample-provenance-aware, and non-authoritative."""
from __future__ import annotations

import asyncio
import importlib
from copy import deepcopy

import httpx
import numpy as np
import pytest

from sigid.api import create_app
from sigid.live_classifier_cache import LiveClassifierEvidenceCache

api_module = importlib.import_module("sigid.api")


def proposal(source_sample=1000):
    return {
        "available": True,
        "abstained": False,
        "method": "BPSK31",
        "family": "PSK",
        "confidence": 0.71,
        "sample_start": source_sample,
        "sample_count": 1000,
        "dsp_evidence": {"available": True, "source": "mock-signal"},
    }


def test_cache_reuse_is_explicit_and_keeps_original_source_coordinates():
    cache = LiveClassifierEvidenceCache()
    args = dict(session_id="s", track_id="s:cv:a", model_id="model",
                sample_rate=1000, low_hz=940.0, high_hz=1070.0)
    first = proposal()
    cache.remember(**args, sample_position=2000, proposal=first)
    first["sample_start"] = -8
    reused = cache.lookup(**args, sample_position=4000)
    assert reused["sample_start"] == 1000
    assert reused["sample_count"] == 1000
    assert reused["reuse_info"]["reused"]
    assert reused["reuse_info"]["evidence_age_seconds"] == 2.0
    assert reused["reuse_info"]["original_sample_start"] == 1000
    reused["method"] = "nonsense"
    assert cache.lookup(**args, sample_position=4000)["method"] == "BPSK31"
    assert cache.status("s")["reused"] == 2
    assert cache.status("wrong-session")["entries"] == 0
    assert cache.status("s")["entries"] == 1  # status did not reset live cache


def test_cache_rejects_stale_changed_model_band_rate_or_track():
    cache = LiveClassifierEvidenceCache(refresh_seconds=6)
    args = dict(session_id="s", track_id="s:cv:a", model_id="sha256-1",
                sample_rate=1000, low_hz=900, high_hz=1100)
    cache.remember(**args, sample_position=10000, proposal=proposal())
    assert cache.lookup(**args, sample_position=10001)
    for override in [
        {"track_id": "s:cv:b"},
        {"model_id": "sha256-2"},
        {"sample_rate": 2000},
        {"low_hz": 990, "high_hz": 1190},
        {"low_hz": 900, "high_hz": 1220},
        {"sample_position": 15999 + 2},
        {"sample_position": 9999},
    ]:
        other = dict(args, sample_position=11000)
        other.update(override)
        assert cache.lookup(**other) is None, override
    assert cache.lookup(**args, sample_position=15999)
    assert cache.lookup(**args, sample_position=16000) is None
    assert cache.status("other")["entries"] == 0


def test_cache_lru_and_unavailable_outcomes_are_bounded():
    cache = LiveClassifierEvidenceCache(max_tracks=2)
    for i in range(5):
        cache.remember(session_id="s", track_id=f"s:cv:{i}", model_id="m",
                       sample_rate=1000, sample_position=i*100,
                       low_hz=400, high_hz=500, proposal=proposal())
    assert cache.status("s")["entries"] == 2
    assert cache.lookup(session_id="s", track_id="s:cv:0", model_id="m",
                        sample_rate=1000, sample_position=500,
                        low_hz=400, high_hz=500) is None
    cache.remember(session_id="s", track_id="s:cv:broken", model_id="m",
                   sample_rate=1000, sample_position=500,
                   low_hz=400, high_hz=500, proposal={"available": False})
    assert cache.status("s")["entries"] == 2
    assert cache.status("s")["scored"] == 6


def test_live_cv_api_reuses_classification_without_creating_verified_labels(tmp_path, monkeypatch):
    app = create_app(tmp_path / "state")
    app.state.classifier = {"metadata": {
        "classifier_window_seconds": 1.0,
        "model_sha256": "d" * 64,
    }}

    class FakeCv:
        model_sha256 = "e" * 64

        def detect_live_partial(self, samples, *, sample_start,
                                total_sample_count, sample_rate, confidence=None):
            return [{"sample_start": total_sample_count - 2 * sample_rate,
                     "sample_count": 2 * sample_rate,
                     "frequency_low_hz": 950.0,
                     "frequency_high_hz": 1050.0,
                     "confidence": 0.8,
                     "model_sha256": self.model_sha256}]

    app.state.cv_detector = FakeCv()
    rate = 12000
    t = np.arange(rate * 6) / rate
    pcm = (np.sin(2 * np.pi * 1000 * t) * 10000).astype("<i2")
    snapshot = {"pcm": pcm[:rate * 2].tobytes(), "sample_rate": rate,
                "channels": 1, "sample_offset": 0,
                "sample_count": rate * 2, "window_sample_count": rate * 2,
                "session_id": "cv-cache", "active": True}
    monkeypatch.setattr(app.state.live_capture, "analysis_audio", lambda: snapshot)
    scoring = []

    def fake_classify(classifier, mono, sample_rate, low_hz, high_hz):
        scoring.append((len(mono), sample_rate, low_hz, high_hz))
        return {**proposal(), "sample_start": -100}

    monkeypatch.setattr(api_module, "_classify_signal_crop", fake_classify)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://test") as client:
            response = await client.get("/api/live/cv-detections",
                params={"session_id": "cv-cache", "include_classifier": "true"})
            assert response.status_code == 200, response.text
            first = response.json()
            assert first["classifier_scored_this_scan"] == 1
            assert first["classifier_reused_this_scan"] == 0
            [box] = first["detections"]
            first_source = box["classifier_proposal"]["sample_start"]
            assert box["classifier_proposal"]["reuse_info"]["reused"] is False
            assert box["track_state"] == "provisional"

            snapshot["sample_count"] += rate
            snapshot["window_sample_count"] = snapshot["sample_count"]
            snapshot["pcm"] = pcm[:snapshot["sample_count"]].tobytes()
            second = (await client.get("/api/live/cv-detections",
                params={"session_id": "cv-cache", "include_classifier": "true"})).json()
            [box] = second["detections"]
            assert box["track_state"] == "persistent"
            assert box["classifier_proposal"]["reuse_info"]["reused"] is True
            assert box["classifier_proposal"]["sample_start"] == first_source
            assert box["classifier_proposal"]["reuse_info"]["evidence_age_seconds"] == 1.0
            assert second["classifier_scored_this_scan"] == 0
            assert second["classifier_reused_this_scan"] == 1
            assert len(scoring) == 1
            assert second["automatic_trial"] is None

            app.state.classifier = {**app.state.classifier, "metadata":
                {**app.state.classifier["metadata"], "model_sha256": "f" * 64}}
            third = (await client.get("/api/live/cv-detections",
                params={"session_id": "cv-cache", "include_classifier": "true"})).json()
            assert third["classifier_scored_this_scan"] == 1
            assert len(scoring) == 2

            paused = (await client.get("/api/live/cv-detections",
                params={"session_id": "cv-cache", "include_classifier": "false"})).json()
            assert "classifier_proposal" not in paused["detections"][0]
            assert paused["classifier_scored_this_scan"] == 0
            status = (await client.get("/api/live/classifier-cache/status",
                params={"session_id": "cv-cache"})).json()
            assert status["reused"] == 1
            assert status["scored"] == 2
            assert status["refresh_seconds"] == 6.0
    try:
        asyncio.run(exercise())
    finally:
        app.state.native_live_runner.close()


def test_crowded_live_scans_limit_new_scores_without_starving_deferred_tracks(tmp_path, monkeypatch):
    app = create_app(tmp_path / "crowded")
    app.state.classifier = {"metadata": {
        "classifier_window_seconds": 1.0, "model_sha256": "a" * 64}}

    class ManyCv:
        model_sha256 = "c" * 64

        def detect_live_partial(self, audio, *, sample_start, total_sample_count,
                                sample_rate, confidence=None):
            return [{"sample_start": total_sample_count - 2 * sample_rate,
                     "sample_count": 2 * sample_rate,
                     "frequency_low_hz": 200 + i * 180,
                     "frequency_high_hz": 270 + i * 180,
                     "confidence": round(0.99 - .01 * i, 3)}
                    for i in range(12)]

    app.state.cv_detector = ManyCv()
    rate = 12000
    pcm = (np.sin(np.arange(rate * 4) / rate * 2 * np.pi * 650) * 12000).astype("<i2")
    snapshot = dict(pcm=pcm[:rate * 2].tobytes(), sample_rate=rate, channels=1,
                    sample_offset=0, sample_count=rate * 2,
                    window_sample_count=rate * 2,
                    session_id="crowded-cache", active=True)
    monkeypatch.setattr(app.state.live_capture, "analysis_audio", lambda: snapshot)
    calls = []

    def scored(classifier, mono, sample_rate, low, high):
        calls.append((low, high))
        return {"available": True, "abstained": False, "family": "PSK",
                "method": "BPSK31", "confidence": 0.8}

    monkeypatch.setattr(api_module, "_classify_signal_crop", scored)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://test") as client:
            params = {"session_id": "crowded-cache", "include_classifier": "true"}
            first = (await client.get("/api/live/cv-detections", params=params)).json()
            assert len(first["detections"]) == 12
            assert first["classifier_attempts_this_scan"] == 8
            assert first["classifier_scored_this_scan"] == 8
            assert first["classifier_deferred_this_scan"] == 4
            assert len(calls) == 8
            snapshot.update(pcm=pcm[:rate * 3].tobytes(), sample_count=rate * 3,
                            window_sample_count=rate * 3)
            second = (await client.get("/api/live/cv-detections", params=params)).json()
            assert second["classifier_reused_this_scan"] == 8
            assert second["classifier_scored_this_scan"] == 4
            assert second["classifier_attempts_this_scan"] == 4
            assert second["classifier_deferred_this_scan"] == 0
            assert len(calls) == 12
            assert all(d["classifier_proposal"].get("available") for d in second["detections"])
            assert second["classifier_cache"]["entries"] == 12
    try:
        asyncio.run(run())
    finally:
        app.state.native_live_runner.close()
