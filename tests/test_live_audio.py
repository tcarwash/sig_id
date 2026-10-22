from __future__ import annotations

import asyncio
import time
import wave
import struct

import httpx

from sigid.api import create_app
from sigid.dataset import CaptureStore
from sigid.live_audio import LiveCaptureManager
from sigid.tracking import LiveTrackRegistry


class FakeAudio:
    CallbackStop = type("CallbackStop", (Exception,), {})
    CallbackAbort = type("CallbackAbort", (Exception,), {})

    def __init__(self):
        self.devices_list = [
            {"name": "Output only", "max_input_channels": 0, "default_samplerate": 48000},
            {"name": "Test input", "max_input_channels": 2, "default_samplerate": 48000},
        ]

    def query_devices(self, device=None, kind=None):
        if kind == "input":
            return self.devices_list[1]
        if device is not None:
            return self.devices_list[device]
        return self.devices_list

    def InputStream(self, **kwargs):
        return FakeStream(self, kwargs)


class FakeStream:
    def __init__(self, backend, kwargs):
        self.backend = backend
        self.kwargs = kwargs

    def start(self):
        # Four little-endian int16 mono frames.
        self.kwargs["callback"](struct.pack("<h", 1234) * 1024, 1024, None, None)

    def stop(self):
        pass

    def close(self):
        pass


class FakeFldigi:
    def __init__(self):
        self.mode = "BPSK31"
        self.text = b""

    def modem_names(self):
        return ["BPSK31", "RTTY", "THOR16"]

    def mode_names(self):
        return self.modem_names()

    def select_modem(self, name):
        self.mode = name
        return name

    def select_mode(self, name):
        return self.select_modem(name)

    def receive_text_length(self):
        return len(self.text)

    def receive_text(self, *, start=0, length=65536):
        return self.text[start : start + length]


class FakeReplay:
    def __init__(self):
        self.frames = None

    def decode(self, wav_path, *, mode, carrier_hz=None):
        with wave.open(str(wav_path), "rb") as wav:
            self.frames = wav.getnframes()
            assert wav.getnchannels() == 1
            assert wav.getframerate() == 8000
        return {"status": "completed", "text": "LIVE TRACK TEXT", "note": f"Decoded {mode}."}


class FakeSweepReplay:
    def __init__(self):
        self.modes = []
        self.frames = None

    def decode(self, wav_path, *, mode, carrier_hz=None):
        self.modes.append(mode)
        with wave.open(str(wav_path), "rb") as wav:
            self.frames = wav.getnframes()
        if mode == "RTTY":
            return {"status": "completed", "text": "SWEEP FOUND TEXT", "note": "Mode decoded."}
        return {"status": "completed_no_text", "text": "", "note": "No text."}


def test_live_capture_writes_wav_and_registers_capture(tmp_path):
    store = CaptureStore(tmp_path / "data.sqlite3")
    backend = FakeAudio()
    manager = LiveCaptureManager(store, tmp_path / "captures", backend)
    assert manager.devices() == [{
        "id": 1, "name": "Test input", "max_input_channels": 2, "default_sample_rate": 48000
    }]
    started = manager.start(device_id=1, sample_rate=8000, max_duration_seconds=5)
    assert started["state"] == "recording"
    result = manager.stop()
    capture = result["capture"]
    assert capture["source_kind"] == "live"
    assert capture["sample_rate"] == 8000
    assert capture["sample_count"] == 1024
    assert capture["source_metadata"]["device_name"] == "Test input"
    with wave.open(capture["audio_path"], "rb") as wav:
        assert wav.readframes(4) == struct.pack("<h", 1234) * 4
    assert manager.status() == {"active": False}


def test_live_capture_rejects_invalid_or_duplicate_start(tmp_path):
    manager = LiveCaptureManager(CaptureStore(tmp_path / "data.sqlite3"), tmp_path / "captures", FakeAudio())
    try:
        manager.start(sample_rate=1000)
    except ValueError as exc:
        assert "sample_rate" in str(exc)
    else:
        raise AssertionError("invalid sample rate was accepted")
    manager.start(max_duration_seconds=5)
    try:
        manager.start(max_duration_seconds=5)
    except RuntimeError as exc:
        assert "already active" in str(exc)
    else:
        raise AssertionError("duplicate capture was accepted")
    manager.stop()


def test_live_capture_api_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sigid.api.detect_tracks",
        lambda spectrum, threshold_db: [{
            "id": "track-001",
            "sample_start": 0,
            "sample_count": spectrum["sample_count"],
            "center_hz": 1000.0,
            "bandwidth_hz": 200.0,
            "peak_snr_db": 20.0,
            "protocol": None,
        }],
    )
    app = create_app(
        tmp_path / "app-data", audio_backend=FakeAudio(), fldigi_client=FakeFldigi()
    )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            devices = await client.get("/api/audio/devices")
            assert devices.status_code == 200
            assert devices.json()[0]["name"] == "Test input"
            bad = await client.post("/api/live/start", json={"sample_rate": 1000})
            assert bad.status_code == 422
            unavailable = await client.post(
                "/api/fldigi/trials/start",
                json={"track_id": "live:0001", "modulation_hypothesis": "FSK-like"},
            )
            assert unavailable.status_code == 409
            started = await client.post("/api/live/start", json={"device_id": 1, "sample_rate": 8000})
            assert started.status_code == 200, started.text
            assert (await client.get("/api/live")).json()["active"] is True
            analysis = await client.get("/api/live/analysis", params={"fft_size": 64, "hop_size": 16})
            assert analysis.status_code == 200, analysis.text
            live_result = analysis.json()
            assert live_result["active"] is True
            assert live_result["spectrum"]["sample_starts"][0] == 0
            assert live_result["spectrum"]["sample_count"] == 1024
            assert live_result["spectrum"]["sample_rate"] == 8000
            track = live_result["tracks"][0]
            assert track["track_id"].startswith(started.json()["session_id"] + ":")
            assert track["track_state"] == "provisional"
            again = (await client.get("/api/live/analysis", params={"fft_size": 64, "hop_size": 16})).json()
            assert again["tracks"][0]["track_id"] == track["track_id"]
            assert again["tracks"][0]["track_observations"] == 1
            trial = await client.post(
                "/api/fldigi/trials/start",
                json={"track_id": track["track_id"], "sample_start": 0, "sample_count": 1024, "modulation_hypothesis": "FSK-like"},
            )
            assert trial.status_code == 200, trial.text
            assert trial.json()["current_mode"] == "RTTY"
            manual = await client.post("/api/fldigi/mode", json={"name": "BPSK31"})
            assert manual.status_code == 200
            stopped_trial = await client.get("/api/fldigi/trials/current")
            assert stopped_trial.json()["status"] == "cancelled"
            stopped = await client.post("/api/live/stop")
            assert stopped.status_code == 200, stopped.text
            capture = stopped.json()["capture"]
            assert capture["source_kind"] == "live"
            assert capture["sample_count"] == 1024
            saved_runs = await client.get(f"/api/captures/{capture['id']}/decode-runs")
            assert saved_runs.status_code == 200, saved_runs.text
            assert len(saved_runs.json()) == 1
            saved_run = saved_runs.json()[0]
            assert saved_run["capture_id"] == capture["id"]
            assert saved_run["track_id"] == track["track_id"]
            assert saved_run["sample_start"] == 0
            assert saved_run["sample_count"] == 1024
            assert saved_run["sample_rate"] == 8000
            assert saved_run["status"] == "cancelled"
            assert saved_run["output_scope"] == "shared_live_input"
            assert saved_run["association"] == "track_window_tentative"
            assert (await client.get(f"/api/captures/missing/decode-runs")).status_code == 404
            assert (await client.get(f"/api/captures/{capture['id']}/waterfall", params={"fft_size": 64, "hop_size": 16})).status_code == 200
            assert (await client.get("/api/live/analysis")).status_code == 409

    asyncio.run(exercise())


def test_live_track_replay_uses_exact_buffered_samples_and_links_when_stopped(tmp_path):
    replay = FakeReplay()
    app = create_app(
        tmp_path / "app-data", audio_backend=FakeAudio(), fldigi_client=FakeFldigi(),
        fldigi_replay_backend=replay,
    )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            started = await client.post("/api/live/start", json={"device_id": 1, "sample_rate": 8000})
            assert started.status_code == 200, started.text
            session_id = started.json()["session_id"]
            settings = {
                "mode": "RTTY", "sample_start": 128, "sample_count": 512,
                "track_id": f"{session_id}:0001", "carrier_hz": 1200,
            }
            response = await client.post(f"/api/live/sessions/{session_id}/decode", json=settings)
            assert response.status_code == 202, response.text
            run_id = response.json()["run_id"]
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                runs = (await client.get(f"/api/live/sessions/{session_id}/decode-runs")).json()
                if runs and runs[0]["status"] not in {"queued", "running"}:
                    break
                await asyncio.sleep(0.01)
            assert runs[0]["id"] == run_id
            assert runs[0]["status"] == "completed"
            assert runs[0]["output_text"] == "LIVE TRACK TEXT"
            assert runs[0]["output_scope"] == "isolated_live_window_replay"
            assert runs[0]["association"] == "exact_live_track_window"
            assert runs[0]["sample_start"] == settings["sample_start"]
            assert runs[0]["sample_count"] == settings["sample_count"]
            assert replay.frames == settings["sample_count"]

            stale = await client.post(f"/api/live/sessions/{session_id}/decode", json={
                **settings, "sample_start": 2000,
            })
            assert stale.status_code == 422
            stopped = await client.post("/api/live/stop")
            assert stopped.status_code == 200
            [saved_run] = (await client.get(
                f"/api/captures/{stopped.json()['capture']['id']}/decode-runs"
            )).json()
            assert saved_run["id"] == run_id
            assert saved_run["association"] == "exact_live_track_window"

    asyncio.run(exercise())


def test_live_mode_sweep_uses_isolated_audio_in_candidate_order(tmp_path):
    replay = FakeSweepReplay()
    app = create_app(
        tmp_path / "app-data", audio_backend=FakeAudio(), fldigi_client=FakeFldigi(),
        fldigi_replay_backend=replay,
    )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            started = await client.post("/api/live/start", json={"device_id": 1, "sample_rate": 8000})
            session_id = started.json()["session_id"]
            response = await client.post(
                f"/api/live/sessions/{session_id}/decode-trials",
                json={"track_id": f"{session_id}:0001", "modulation_hypothesis": "FSK-like",
                      "sample_start": 32, "sample_count": 640, "carrier_hz": 1100},
            )
            assert response.status_code == 202, response.text
            run_id = response.json()["run_id"]
            duplicate = await client.post(
                f"/api/live/sessions/{session_id}/decode-trials",
                json={"track_id": f"{session_id}:0001", "modulation_hypothesis": "FSK-like",
                      "sample_start": 32, "sample_count": 640, "carrier_hz": 1100},
            )
            assert duplicate.status_code == 409
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                [run] = (await client.get(f"/api/live/sessions/{session_id}/decode-runs")).json()
                if run["status"] not in {"queued", "running"}:
                    break
                await asyncio.sleep(0.01)
            assert run["id"] == run_id
            assert run["status"] == "tentative_output"
            assert run["output_text"] == "SWEEP FOUND TEXT"
            assert run["modes"] == ["RTTY", "THOR16", "BPSK31"]
            assert run["attempted_modes"] == ["RTTY"]
            assert replay.modes == ["RTTY"]
            assert replay.frames == 640
            assert run["output_scope"] == "isolated_live_mode_sweep"
            assert run["association"] == "exact_live_track_window"
            stopped = await client.post("/api/live/stop")
            assert stopped.status_code == 200
            [saved] = (await client.get(
                f"/api/captures/{stopped.json()['capture']['id']}/decode-runs"
            )).json()
            assert saved["id"] == run_id

    asyncio.run(exercise())


def test_accepted_persistent_model_candidate_starts_isolated_sweep(tmp_path, monkeypatch):
    replay = FakeSweepReplay()
    monkeypatch.setattr("sigid.api.detect_tracks", lambda spectrum, threshold_db: [{
        "id": "track-001", "sample_start": 0, "sample_count": spectrum["sample_count"],
        "center_hz": 1000.0, "bandwidth_hz": 200.0, "peak_snr_db": 20.0, "protocol": None,
    }])
    monkeypatch.setattr("sigid.api.extract_features", lambda *args, **kwargs: {})
    monkeypatch.setattr("sigid.api.propose", lambda *args, **kwargs: {
        "abstained": False, "family": "FSK", "method": "FSK", "confidence": 0.99,
    })
    app = create_app(
        tmp_path / "app-data", audio_backend=FakeAudio(), fldigi_client=FakeFldigi(),
        fldigi_replay_backend=replay,
    )
    app.state.classifier = {
        "metadata": {
            "feature_version": "test", "classes": ["FSK"],
            "abstention_threshold": 0.5, "family_abstention_threshold": 0.5,
            "model_sha256": "test-model",
        },
        "parameters": {},
    }

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            started = await client.post("/api/live/start", json={"device_id": 1, "sample_rate": 8000})
            session_id = started.json()["session_id"]
            app.state.live_track_registry = LiveTrackRegistry(session_id, confirmation_observations=1)
            analysis = await client.get("/api/live/analysis", params={
                "fft_size": 64, "hop_size": 16, "auto_trial": "true", "include_spectrum": "false",
            })
            assert analysis.status_code == 200, analysis.text
            trial = analysis.json()["automatic_trial"]
            assert trial["trigger"] == "accepted_model_proposal"
            assert trial["run_id"]
            assert trial["output_scope"] == "isolated_live_mode_sweep"
            [run] = (await client.get(f"/api/live/sessions/{session_id}/decode-runs")).json()
            assert run["track_id"] == analysis.json()["tracks"][0]["track_id"]
            assert run["sample_start"] == 0
            assert run["sample_count"] == 1024
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and run["status"] in {"queued", "running"}:
                await asyncio.sleep(0.01)
                [run] = (await client.get(f"/api/live/sessions/{session_id}/decode-runs")).json()
            assert run["status"] == "tentative_output"
            assert run["output_text"] == "SWEEP FOUND TEXT"
            await client.post("/api/live/stop")

    asyncio.run(exercise())


def test_application_shutdown_finalizes_an_active_recording(tmp_path):
    app = create_app(tmp_path / "data", audio_backend=FakeAudio(), fldigi_client=FakeFldigi())

    async def exercise():
        async with app.router.lifespan_context(app):
            app.state.live_capture.start(sample_rate=8000)
            assert app.state.live_capture.status()["active"]
        captures = app.state.capture_store.list_captures()
        assert len(captures) == 1
        assert captures[0]["sample_count"] == 1024
        assert app.state.live_capture.status() == {"active": False}
        assert app.state.live_capture.shutdown() is None

    asyncio.run(exercise())


def test_writer_failure_preserves_completed_blocks_as_a_partial_capture(tmp_path, monkeypatch):
    import time

    class TwoBlockStream(FakeStream):
        def start(self):
            for _ in range(2):
                self.kwargs["callback"](struct.pack("<h", 2345) * 1024, 1024, None, None)

    class TwoBlockAudio(FakeAudio):
        def InputStream(self, **kwargs):
            return TwoBlockStream(self, kwargs)

    original = wave.Wave_write.writeframesraw
    writes = 0

    def fail_second_write(writer, data):
        nonlocal writes
        writes += 1
        if writes > 1:
            raise OSError("simulated full disk")
        return original(writer, data)

    monkeypatch.setattr(wave.Wave_write, "writeframesraw", fail_second_write)
    store = CaptureStore(tmp_path / "data.sqlite3")
    manager = LiveCaptureManager(store, tmp_path / "captures", TwoBlockAudio())
    manager.start(sample_rate=8000)
    deadline = time.monotonic() + 2
    while manager.status()["state"] != "error" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.status()["state"] == "error"
    result = manager.shutdown()
    assert result["truncated_reason"] == "audio_writer_error"
    capture = result["capture"]
    assert capture["sample_count"] == 1024
    assert "simulated full disk" in " ".join(capture["source_metadata"]["status_messages"])
    with wave.open(capture["audio_path"], "rb") as wav:
        assert wav.readframes(1024) == struct.pack("<h", 2345) * 1024


def test_index_failure_keeps_a_readable_file_and_releases_the_capture_slot(tmp_path, monkeypatch):
    import pytest

    store = CaptureStore(tmp_path / "data.sqlite3")
    manager = LiveCaptureManager(store, tmp_path / "captures", FakeAudio())
    started = manager.start(sample_rate=8000)

    def fail_index(*args, **kwargs):
        raise OSError("simulated metadata storage failure")

    monkeypatch.setattr(store, "add_wav_capture", fail_index)
    with pytest.raises(RuntimeError, match="audio retained at"):
        manager.stop()
    assert manager.status() == {"active": False}
    files = list((tmp_path / "captures").glob("*.wav"))
    assert len(files) == 1
    with wave.open(str(files[0]), "rb") as wav:
        assert wav.getnframes() == 1024
