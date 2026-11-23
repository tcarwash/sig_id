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


def test_provisional_cw_signal_triggers_budgeted_native_trial_before_three_polls(tmp_path, monkeypatch):
    """Actual FastAPI -> CV -> uncertain ML ranking -> isolated worker."""
    from sigid.signal_lab import render
    from tools.bench_signal_chain_snr import read_mono
    import sigid.api as sigid_api
    manifest=render(modes=["cw"], output=tmp_path/"probe-cw.wav",
                    duration=30,sample_rate=12000,seed=71062,
                    traffic="sparse",bursts_per_mode=1,burst_seconds=16,
                    snr_db=25,snr_reference_hz=2500,stereo=False,qrm=False)
    pcm,rate,_=read_mono(manifest["mono_wav"])
    carrier=manifest["transmissions"][0]["carrier_hz"]
    app=create_app(tmp_path/"cw-probe-state")
    app.state.classifier={"metadata":{"classifier_window_seconds":2.0,
      "model_sha256":"a"*64,"feature_version":"passband-log-spectrum-envelope-roi-fixed-window-v4"},
      "parameters":{}}
    def uncertain(*args,**kwargs):
        return {"available":True,"abstained":True,"method":None,
                "top_candidate":"Morse → CW","confidence":.65,
                "ranked_candidates":[
                  {"mode_id":"fldigi-cw","relative_model_score":.72},
                  {"mode_id":"bpsk31","relative_model_score":.24}]}
    monkeypatch.setattr(sigid_api,"_classify_signal_crop", uncertain)
    class FakeCv:
        model_sha256="b"*64
        model_version="fixture"
        def detect_live_partial(self, audio, *, sample_start, total_sample_count,
                                sample_rate, confidence=None):
            return [{"sample_start":0,"sample_count":8*rate,
                     "frequency_low_hz":carrier-10,"frequency_high_hz":carrier+10,
                     "confidence":.91,"model_sha256":self.model_sha256,
                     "model_version":self.model_version}]
    app.state.cv_detector=FakeCv()
    snapshot={"active":True,"session_id":"early-cw","sample_rate":rate,
              "channels":1,"sample_offset":0,"sample_count":8*rate,
              "window_sample_count":8*rate,"pcm":pcm[:8*rate].tobytes()}
    monkeypatch.setattr(app.state.live_capture,"analysis_audio",lambda: snapshot)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                      base_url="http://test",timeout=25) as client:
            r=await client.get("/api/live/cv-detections",params={
                "session_id":"early-cw","auto_native_decode":"true"})
            assert r.status_code==200,r.text
            body=r.json()
            [box]=body["detections"]
            assert box["track_state"]=="provisional"
            assert box["track_observations"]==1
            assert box["classifier_proposal"]["abstained"] is True
            assert box["classifier_proposal"]["method"] is None
            assert box["native_decode_admission"]["status"]=="queued"
            assert body["native_decode_trial"]["admission"]=="provisional_cw"
            finish=time.monotonic()+15
            while time.monotonic()<finish:
                stats=(await client.get("/api/live/native-decode/status",
                      params={"session_id":"early-cw"})).json()
                if not stats["busy"]:break
                await asyncio.sleep(.04)
            assert not stats["busy"]
            assert stats["provisional_probes_last_minute"]==1
            assert stats["counts"]["admitted_provisional"]==1
            assert stats["recent"]
            event=stats["recent"][0]
            assert event["confirmed_identity"] is False
            assert "CQ" in event["cw_stream_cumulative_text"] or "TEST" in event["cw_stream_cumulative_text"]
            runs=app.state.capture_store.decoder_runs(session_id="early-cw")
            assert len(runs)==1
            assert runs[0]["decoder_result"]["cw_stream"]["confirmed_identity"] is False
    try:
        asyncio.run(check())
    finally:
        app.state.native_live_runner.close()


def test_live_cv_carrier_drift_preserves_native_worker_keying_state(tmp_path, monkeypatch):
    """Real FastAPI/ProcessPool path; successive CV carriers drift by 13.5 Hz."""
    import sigid.api as sigid_api
    import sigid.native_live_decode as native_decode
    from sigid.signal_lab import render
    from tools.bench_cw_carrier_tracking import drift_audio
    from tools.bench_signal_chain_snr import read_mono

    manifest=render(modes=["cw"],output=tmp_path/"cv-drift.wav",
                    duration=30,sample_rate=12000,seed=71062,traffic="sparse",
                    bursts_per_mode=1,burst_seconds=16,snr_db=0,
                    snr_reference_hz=2500,stereo=False,qrm=False)
    pcm,rate,_=read_mono(manifest["mono_wav"])
    pcm=drift_audio(pcm,rate,90)
    nominal=manifest["transmissions"][0]["carrier_hz"]
    app=create_app(tmp_path/"cv-drift-app")
    app.state.classifier={"metadata":{"classifier_window_seconds":2.0,
         "model_sha256":"a"*64,"feature_version":"passband-log-spectrum-envelope-roi-fixed-window-v4"},
         "parameters":{}}
    def uncertain(*args,**kwargs):
        return {"available":True,"abstained":True,"method":None,
                "top_candidate":"Morse → CW","confidence":.67,
                "ranked_candidates":[
                  {"mode_id":"fldigi-cw","relative_model_score":.75},
                  {"mode_id":"bpsk31","relative_model_score":.22}]}
    monkeypatch.setattr(sigid_api,"_classify_signal_crop",uncertain)
    monkeypatch.setattr(native_decode,"MIN_ATTEMPT_GAP_SECONDS",0.0)
    monkeypatch.setattr(native_decode,"CW_STREAM_TRACK_COOLDOWN_SECONDS",0.0)
    class FakeCv:
        model_sha256="b"*64
        model_version="synthetic"
        def detect_live_partial(self,audio,*,sample_start,total_sample_count,
                                sample_rate,confidence=None):
            elapsed=total_sample_count/sample_rate
            center=nominal+90*min(elapsed/20,1)
            return [{"sample_start":0,"sample_count":total_sample_count,
                     "frequency_low_hz":center-10,"frequency_high_hz":center+10,
                     "confidence":.91,"model_sha256":self.model_sha256,
                     "model_version":self.model_version}]
    app.state.cv_detector=FakeCv()
    snapshot={"active":True,"session_id":"cv-retune", "sample_rate":rate,
              "channels":1,"sample_offset":0,"sample_count":8*rate,
              "window_sample_count":8*rate,"pcm":pcm[:8*rate].tobytes()}
    monkeypatch.setattr(app.state.live_capture,"analysis_audio",lambda:snapshot)

    async def poll_finished(client):
        finish=time.monotonic()+15
        while time.monotonic()<finish:
            stats=(await client.get("/api/live/native-decode/status",
                     params={"session_id":"cv-retune"})).json()
            if not stats["busy"]:
                return stats
            await asyncio.sleep(.04)
        raise AssertionError("worker did not finish")

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                   base_url="http://test",timeout=25) as client:
            params={"session_id":"cv-retune","auto_native_decode":"true"}
            first=(await client.get("/api/live/cv-detections",params=params)).json()
            assert first["detections"][0]["track_state"]=="provisional"
            assert first["native_decode_trial"]["admission"]=="provisional_cw"
            assert first["native_decode_trial"]["modes"][0]=="cw"
            await poll_finished(client)
            snapshot.update({"sample_count":11*rate,"window_sample_count":11*rate,
                             "pcm":pcm[:11*rate].tobytes()})
            second=(await client.get("/api/live/cv-detections",params=params)).json()
            assert second["detections"][0]["track_state"]=="persistent"
            assert second["native_decode_trial"]["status"]=="queued"
            assert second["native_decode_trial"]["admission"]=="persistent"
            await poll_finished(client)
            runs=app.state.capture_store.decoder_runs(session_id="cv-retune")
            assert len(runs)==2
            first_stream=runs[0]["decoder_result"]["cw_stream"]
            last_stream=runs[1]["decoder_result"]["cw_stream"]
            assert first_stream["processed_new_samples"]==8*rate
            assert first_stream["state_reset_reason"]=="new-stream"
            assert last_stream["processed_new_samples"]==3*rate
            assert last_stream["state_reset_reason"] is None
            assert last_stream["carrier_retune_count"]==1
            assert last_stream["last_carrier_step_hz"]==13.5
            assert last_stream["confirmed_identity"] is False
            assert abs(last_stream["carrier_hz"]-(nominal+49.5))<1e-6
    try:
        asyncio.run(check())
    finally:
        app.state.native_live_runner.close()


def test_short_ax25_packet_reaches_native_decoder_through_real_api(tmp_path, monkeypatch):
    """A CRC-passed ~0.31s Bell 202 frame must not hit the old 0.4s gate."""
    manifest=json.loads((Path(__file__).parent/
        "fixtures/decoder_conformance/ax25/manifest.json").read_text())
    case=next(x for x in manifest["fixtures"] if x["expected"]["kind"]=="positive")
    with wave.open(str(Path(__file__).parent / "fixtures/decoder_conformance/ax25"
                       / case["audio"]["path"]),"rb") as f:
        frame=f.readframes(f.getnframes())
        rate=f.getframerate()
    frame_samples=len(frame)//2
    assert .125 < frame_samples/rate < .4
    app=create_app(tmp_path/"short-ax25")
    app.state.classifier=None
    class FakeCV:
        model_sha256="d"*64
        model_version="test"
        def detect_live_partial(self,samples,*,sample_start,total_sample_count,
                                sample_rate,confidence=None):
            return [{"sample_start":0,"sample_count":frame_samples,
                     "frequency_low_hz":650,"frequency_high_hz":1750,
                     "confidence":.93,"model_sha256":self.model_sha256,
                     "model_version":self.model_version}]
    app.state.cv_detector=FakeCV()
    pcm=frame+b"\0\0"*(2*rate-frame_samples)
    snapshot={"active":True,"session_id":"short-ax25", "sample_rate":rate,
              "channels":1,"sample_offset":0,"sample_count":rate,
              "window_sample_count":rate,"pcm":pcm[:2*rate]}
    monkeypatch.setattr(app.state.live_capture,"analysis_audio",lambda:snapshot)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                       base_url="http://test",timeout=25) as client:
            params={"session_id":"short-ax25","auto_native_decode":"true",
                    "include_classifier":"false"}
            one=await client.get("/api/live/cv-detections",params=params)
            assert one.status_code==200,one.text
            assert one.json()["detections"][0]["track_state"]=="provisional"
            assert one.json()["native_decode_trial"] is None
            snapshot.update({"sample_count":2*rate,"window_sample_count":2*rate})
            two=await client.get("/api/live/cv-detections",params=params)
            assert two.status_code==200,two.text
            trial=two.json()["native_decode_trial"]
            assert trial and trial["status"]=="queued"
            assert trial["modes"][0]=="ax25"
            finish=time.monotonic()+15
            while time.monotonic()<finish:
                stats=(await client.get("/api/live/native-decode/status",
                             params={"session_id":"short-ax25"})).json()
                if not stats["busy"]:break
                await asyncio.sleep(.04)
            assert not stats["busy"]
            assert stats["recent"][0]["grade"]=="frame-integrity-passed"
            assert stats["recent"][0]["text"]=="HELLO WORLD"
            assert stats["modem_stream_frames"]==1
            runs=app.state.capture_store.decoder_runs(session_id="short-ax25")
            assert len(runs)==1
            assert runs[0]["decoder_result"]["modem_stream"]["new_items"]==1
            assert runs[0]["decoder_result"]["live_assistance"]["confirmed_identity"] is False
    try:
        asyncio.run(check())
    finally:
        app.state.native_live_runner.close()
