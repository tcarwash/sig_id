"""Live CV integration: async native decoding assists but never rewrites ML labels."""
from __future__ import annotations

import asyncio
import json
import time
import wave
from pathlib import Path

import httpx

from sigid.api import create_app


def test_live_cv_auto_native_decode_runs_while_classifier_abstains(tmp_path, monkeypatch):
    manifest = json.loads(
        (Path(__file__).parent / "fixtures/decoder_conformance/rtty/manifest.json").read_text())
    case = next(item for item in manifest["fixtures"] if item["expected"]["kind"] == "positive")
    with wave.open(str(Path(__file__).parent / "fixtures/decoder_conformance/rtty"
                       / case["audio"]["path"]), "rb") as f:
        pcm = f.readframes(f.getnframes())
        rate = f.getframerate()
    count = len(pcm) // 2
    assert count >= rate

    app = create_app(tmp_path / "state")
    app.state.classifier = None  # No accepted model is needed for native evidence.
    class FakeCv:
        model_sha256 = "a" * 64
        model_version = "local-test"

        def detect_live_partial(self, samples, *, sample_start, total_sample_count,
                                sample_rate, confidence=None):
            return [{
                "sample_start": 0, "sample_count": total_sample_count,
                "frequency_low_hz": case["carrier_hz"] - 110,
                "frequency_high_hz": case["carrier_hz"] + 110,
                "confidence": 0.87, "model_sha256": self.model_sha256,
                "model_version": self.model_version,
            }]

    app.state.cv_detector = FakeCv()
    snapshot = {
        "active": True, "session_id": "test-live-native",
        "sample_rate": rate, "channels": 1, "sample_offset": 0,
        "pcm": pcm[:rate * 2], "sample_count": rate,
        "window_sample_count": rate,
    }
    monkeypatch.setattr(app.state.live_capture, "analysis_audio", lambda: snapshot)

    async def check():
        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test",
                timeout=25) as client:
            params = {"session_id": "test-live-native",
                      "include_classifier": "false",
                      "auto_native_decode": "true"}
            first = await client.get("/api/live/cv-detections", params=params)
            assert first.status_code == 200, first.text
            assert first.json()["native_decode_trial"] is None
            [box] = first.json()["detections"]
            assert box["track_state"] == "provisional"

            snapshot.update({"sample_count": count,
                             "window_sample_count": count, "pcm": pcm})
            second = await client.get("/api/live/cv-detections", params=params)
            assert second.status_code == 200, second.text
            [box] = second.json()["detections"]
            assert box["track_state"] == "persistent"
            trial = second.json()["native_decode_trial"]
            assert trial and trial["status"] == "queued"
            assert trial["modes"][0] == "rtty"
            assert box.get("classifier_proposal") is None

            end = time.monotonic() + 15
            stats = {}
            while time.monotonic() < end:
                r = await client.get("/api/live/native-decode/status",
                                     params={"session_id": "test-live-native"})
                assert r.status_code == 200, r.text
                stats = r.json()
                if not stats["busy"]:
                    break
                await asyncio.sleep(.04)
            assert not stats["busy"], stats
            assert stats["recent"][0]["text"] == "CQ TEST"
            assert not stats["recent"][0]["confirmed_identity"]
            assert stats["cpu_ms_last_minute"] > 0
            assert "rtty" in stats["latency_by_mode"]
            third = await client.get("/api/live/cv-detections", params=params)
            assert third.status_code == 200, third.text
            [box] = third.json()["detections"]
            assert box["native_decode_evidence"]["text"] == "CQ TEST"
            assert box["native_decode_evidence"]["grade"] == "coherent-unverified"
            assert third.json()["native_decode_trial"] is None
            runs = app.state.capture_store.decoder_runs(session_id="test-live-native")
            assert len(runs) == 1
            assert runs[0]["output_text"] == "CQ TEST"
            assert runs[0]["decoder_result"]["live_assistance"]["confirmed_identity"] is False
    try:
        asyncio.run(check())
    finally:
        app.state.native_live_runner.close()


def test_native_live_switch_is_explicit_and_off_means_no_cpu(tmp_path, monkeypatch):
    app = create_app(tmp_path / "state")
    class FakeCv:
        model_sha256 = "b" * 64
        def detect_live_partial(self, samples, *, sample_start, total_sample_count,
                                sample_rate, confidence=None):
            return [{"sample_start": 0, "sample_count": len(samples),
                     "frequency_low_hz": 800, "frequency_high_hz": 1000,
                     "confidence": 0.8}]
    app.state.cv_detector = FakeCv()
    pcm = b"\0\0" * 24_000
    snapshot = {"active": True, "session_id": "off-session",
                "sample_rate": 12000, "channels": 1,
                "sample_offset": 0, "sample_count": 24000,
                "window_sample_count": 24000, "pcm": pcm}
    monkeypatch.setattr(app.state.live_capture, "analysis_audio", lambda: snapshot)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://test") as client:
            for _ in range(3):
                r=await client.get("/api/live/cv-detections",
                    params={"session_id":"off-session", "include_classifier": "false"})
                assert r.status_code==200
                assert r.json()["native_decode_trial"] is None
            s=await client.get("/api/live/native-decode/status",
                               params={"session_id":"off-session"})
            assert s.status_code==200
            assert s.json()["attempts_last_minute"]==0
            assert app.state.capture_store.decoder_runs(session_id="off-session")==[]
    try:
        asyncio.run(check())
    finally:
        app.state.native_live_runner.close()
