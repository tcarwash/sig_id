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

    def select_modem(self, name):
        self.mode = name
        return name

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
                json={"track_id": track["track_id"], "modulation_hypothesis": "FSK-like"},
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
            assert (await client.get(f"/api/captures/{capture['id']}/waterfall", params={"fft_size": 64, "hop_size": 16})).status_code == 200
            assert (await client.get("/api/live/analysis")).status_code == 409

    asyncio.run(exercise())
