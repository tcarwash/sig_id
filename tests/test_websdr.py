from __future__ import annotations

import asyncio
import struct
import time

import httpx
import pytest

from sigid.api import create_app
from sigid.websdr import WebSDRSessionManager, validate_receiver_url


def test_receiver_url_validation_and_lan_opt_in():
    assert validate_receiver_url("https://www.sdrutah.org/") == "https://www.sdrutah.org/"
    assert validate_receiver_url("http://192.168.1.20:8073/", allow_private=True) == "http://192.168.1.20:8073/"
    for url in (
        "file:///etc/passwd",
        "javascript:alert(1)",
        "http://user:password@example.com/",
        "http://localhost:8073/",
        "http://192.168.1.20:8073/",
        "http://127.0.0.1:8073/",
    ):
        with pytest.raises(ValueError):
            validate_receiver_url(url)


class FakeLoopback:
    instances = []

    def __init__(self, *, profile_dir, browser=None):
        self.profile_dir = profile_dir
        self.browser = browser
        self.device_name = "sigid_websdr_fake_rx"
        self.device_label = "Signal-ID-WebSDR-RX-fake"
        self.audio_source_name = self.device_name
        self.running = False
        self.closed = False
        self.url = None
        self.__class__.instances.append(self)

    def launch(self, url):
        self.url = url
        self.running = True

    def status(self):
        return {
            "browser_running": self.running,
            "state": "connected" if self.running else "browser_closed",
            "device_name": self.device_name,
            "device_label": self.device_label,
            "audio_source_name": self.audio_source_name,
            "routing_errors": 0,
        }

    def close(self):
        self.closed = True
        self.running = False


def test_session_manager_lifecycle_and_reconnect_after_browser_exit(tmp_path):
    FakeLoopback.instances.clear()
    manager = WebSDRSessionManager(tmp_path, loopback_factory=FakeLoopback)
    connected = manager.connect("https://www.sdrutah.org/", "Northern Utah")
    assert connected["active"] is True
    assert connected["browser_running"] is True
    assert connected["name"] == "Northern Utah"
    assert connected["device_name"] == "sigid_websdr_fake_rx"
    with pytest.raises(RuntimeError):
        manager.connect("https://example.org/", "Other")
    first = FakeLoopback.instances[0]
    first.running = False
    replacement = manager.connect("https://example.org/", "Other")
    assert replacement["url"] == "https://example.org/"
    assert first.closed is True
    assert len(FakeLoopback.instances) == 2
    assert manager.disconnect() == {"active": False, "state": "disconnected"}
    assert FakeLoopback.instances[1].closed is True
    assert manager.status()["active"] is False


class FakeAudio:
    CallbackStop = type("CallbackStop", (Exception,), {})
    CallbackAbort = type("CallbackAbort", (Exception,), {})

    def __init__(self):
        self.devices_list = [
            {"name": "Output only", "max_input_channels": 0, "default_samplerate": 48000},
            {"name": "sigid_websdr_test_rx", "max_input_channels": 1, "default_samplerate": 48000},
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
        self.kwargs["callback"](struct.pack("<h", 1234) * 1024, 1024, None, None)

    def stop(self):
        pass

    def close(self):
        pass


class FakeWebSDRManager:
    def __init__(self):
        self.active = False
        self.name = None
        self.url = None
        self.device_name = "sigid_websdr_test_rx"

    def connect(self, url, name):
        self.url = validate_receiver_url(url)
        self.name = name
        self.active = True
        return self.status()

    def status(self):
        if not self.active:
            return {"active": False, "state": "disconnected"}
        return {
            "active": True,
            "state": "connected",
            "browser_running": True,
            "name": self.name,
            "url": self.url,
            "device_name": self.device_name,
            "audio_source_name": self.device_name,
            "routing_errors": 0,
        }

    def disconnect(self):
        self.active = False
        return {"active": False, "state": "disconnected"}

    def close(self):
        self.disconnect()


def test_websdr_api_routes_capture_metadata_and_protect_active_input(tmp_path):
    manager = FakeWebSDRManager()
    app = create_app(tmp_path / "data", audio_backend=FakeAudio(), websdr_manager=manager)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            receivers = await client.get("/api/websdr/receivers")
            assert receivers.status_code == 200
            assert any(item["id"] == "northern-utah" for item in receivers.json())

            invalid = await client.post("/api/websdr/connect", json={
                "url": "file:///etc/passwd", "name": "Invalid",
            })
            assert invalid.status_code == 422

            connected = await client.post("/api/websdr/connect", json={
                "url": "https://www.sdrutah.org/", "name": "Northern Utah WebSDR",
            })
            assert connected.status_code == 200, connected.text
            assert connected.json()["device_id"] == 1
            assert (await client.get("/api/websdr/status")).json()["device_id"] == 1

            app.state.cv_detector = object()
            band_hop = await client.post("/api/live/start", json={
                "device_id": 1,
                "capture_mode": "unattended",
                "band_hop_frequencies_hz": [7_000_000, 14_000_000],
            })
            assert band_hop.status_code == 422
            assert "rigctl band hopping is disabled" in band_hop.json()["detail"]

            started = await client.post("/api/live/start", json={
                "device_id": 1,
                "receiver_frequency_hz": 7_074_000,
                "sample_rate": 8000,
                "max_duration_seconds": 60,
                "capture_mode": "record",
            })
            assert started.status_code == 200, started.text

            blocked = await client.post("/api/websdr/disconnect")
            assert blocked.status_code == 409

            await asyncio.sleep(0.05)
            stopped = await client.post("/api/live/stop")
            assert stopped.status_code == 200, stopped.text
            receiver = stopped.json()["capture"]["source_metadata"]["receiver"]
            assert receiver == {
                "source_type": "websdr",
                "receiver_name": "Northern Utah WebSDR",
                "receiver_url": "https://www.sdrutah.org/",
                "frequency_hz": 7_074_000,
            }

            disconnected = await client.post("/api/websdr/disconnect")
            assert disconnected.status_code == 200
            assert (await client.get("/api/websdr/status")).json()["active"] is False

    asyncio.run(exercise())


def test_settings_panel_exposes_one_click_receiver_workflow():
    from pathlib import Path

    web_root = Path(__file__).parents[1] / "sigid" / "web"
    html = (web_root / "index.html").read_text()
    js = (web_root / "app.js").read_text()
    element_ids = ("websdr-panel", "websdr-preset", "websdr-url", "websdr-name",
                   "websdr-frequency", "websdr-connect", "websdr-disconnect", "websdr-status")
    for element_id in element_ids:
        assert f'id="{element_id}"' in html
    for element_id in element_ids[1:]:
        assert f'el("{element_id}")' in js
    assert "Connect &amp; start live" in html
    assert "/api/websdr/connect" in js
    assert "/api/websdr/disconnect" in js
    assert "isWebSDRInput" in js
