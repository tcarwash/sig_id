from __future__ import annotations

import asyncio
import wave
import struct

import httpx

from sigid.api import create_app
from sigid.dataset import CaptureStore
from sigid.live_audio import LiveCaptureManager


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
