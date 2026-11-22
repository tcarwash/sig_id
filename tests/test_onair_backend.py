"""Contract tests for OnAir's documented read-only /api/v1/events API."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from sigid.api import create_app
from sigid.onair_backend import OnAirEventBackend, _normalise_event


class FakeOnAirServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, token: str | None = "test-secret"):
        super().__init__(("127.0.0.1", 0), FakeOnAirHandler)
        self.token = token
        self.counter = 0
        self.history = []
        self.requests = []
        self.gap = False
        self.redirect = False

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_port}"

    def emit(self, event):
        self.counter += 1
        self.history.append((self.counter, event))


class FakeOnAirHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        base = urlsplit(self.path)
        state = self.server
        state.requests.append((self.command, base.path, parse_qs(base.query)))
        if base.path != "/api/v1/events":
            self.send_error(404)
            return
        if state.redirect:
            self.send_response(302)
            self.send_header("Location", "http://example.com/private")
            self.end_headers()
            return
        if state.token and self.headers.get("Authorization") != "Bearer " + state.token:
            self.send_error(401)
            return
        import json
        params = parse_qs(base.query)
        if "since" in params:
            cursor = int(params["since"][0])
            events = [item for seq, item in state.history if seq > cursor]
        else:
            # OnAir says an omitted cursor starts from 'now'.
            events = []
        payload = json.dumps({
            "gap": state.gap,
            "next": state.counter,
            "events": events[: int(params.get("max", [100])[0])],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextmanager
def onair_server(token: str | None = "test-secret"):
    server = FakeOnAirServer(token)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def decoded_event(mode="CW", *, message="CQ TEST", freq_hz=7_074_000):
    return {
        "v": 1, "type": "ham.cw", "time": datetime.now(timezone.utc).timestamp(),
        "mode": mode, "freqHz": freq_hz,
        "fields": {"text": message, "snr_db": -12},
    }


def test_event_poller_is_read_only_authenticated_cursor_aware_and_deduplicated():
    with onair_server() as server:
        backend = OnAirEventBackend(server.url, token="test-secret", start_worker=False)
        try:
            assert backend.info().connected is False
            assert backend.poll_once() == 0
            assert backend.info().connected is True
            assert server.requests[0][2] == {"max": ["100"]}
            server.emit(decoded_event("CW", message="TEST ONE"))
            server.emit(decoded_event("FT8", message="CQ AG7SU CN85"))
            assert backend.poll_once() == 2
            assert backend.poll_once() == 0
            output = backend.drain_events()
            assert [item.native_mode for item in output] == ["CW", "FT8"]
            assert output[0].mode_id == "onair-cw"
            assert output[1].mode_id == "ft8"
            assert output[0].products[0].content == "TEST ONE"
            assert output[0].metadata["frequency_reference"] == "rf"
            assert output[0].metadata["event_type"] == "ham.cw"
            assert output[0].frequency_hz == 7_074_000
            assert output[0].snr_db == -12
            assert backend.drain_events() == []
            assert server.requests[1][2]["since"] == ["0"]
            assert server.requests[2][2]["since"] == ["2"]
            assert all(method == "GET" and path == "/api/v1/events" for method, path, _ in server.requests)

            # Repeated provider message with a NEW sequence is not displayed twice.
            server.emit(decoded_event("CW", message="TEST ONE"))
            # Normalizer hashes the exact source event, so preserve the original
            # timestamp when re-emitting the exact same message.
            server.history[-1] = (server.counter, server.history[0][1])
            assert backend.poll_once() == 0
            assert backend.drain_events() == []

            # OnAir restart resets its cursor; new events must be readable.
            server.counter = 0
            server.history.clear()
            assert backend.poll_once() == 0
            server.emit(decoded_event("CW", message="AFTER RESTART"))
            server.gap = True
            assert backend.poll_once() == 1
            assert backend.drain_events()[0].products[0].content == "AFTER RESTART"
            assert "source gaps" in backend.info().note
            assert backend.info().connected
        finally:
            backend.close()


def test_offline_authentication_failure_and_redirect_never_trigger_remote_fetch():
    backend = OnAirEventBackend(start_worker=False)
    assert not backend.info().connected
    assert backend.poll_once() == 0
    backend.close()
    for url in (
        "https://127.0.0.1:8765", "http://192.168.1.5:8765",
        "http://evil.example:8765", "http://127.0.0.1:8765/api/v1",
        "http://user:password@127.0.0.1:8765", "http://localhost",
        "http://127.0.0.1:99999",
    ):
        with pytest.raises(ValueError):
            OnAirEventBackend(url, start_worker=False)

    with onair_server() as server:
        backend = OnAirEventBackend(server.url, token="wrong", start_worker=False)
        assert backend.poll_once() == 0
        assert backend.info().connected is False
        assert "HTTP 401" in backend.info().note
        assert "wrong" not in backend.info().note
        backend.close()

        server.redirect = True
        backend = OnAirEventBackend(server.url, token="test-secret", start_worker=False)
        assert backend.poll_once() == 0
        assert not backend.info().connected
        assert "HTTP 302" in backend.info().note
        backend.close()


def test_malformed_events_abstain_instead_of_inventing_decoder_results():
    assert _normalise_event({"v": 1, "mode": "CW", "type": "ham.cw"}) is None
    assert _normalise_event({**decoded_event(), "v": 2}) is None
    assert _normalise_event({**decoded_event(), "time": float("nan")}) is None
    assert _normalise_event({**decoded_event(), "mode": ""}) is None
    event = decoded_event("POCSAG", freq_hz=153_350_000)
    event["fields"] = {"address": 1234, "payload": "test"}
    result = _normalise_event(event)
    assert result.mode_id == "onair-pocsag"
    assert result.native_mode == "POCSAG"
    assert result.products[0].kind == "metadata"
    assert '"address": 1234' in result.products[0].content
    assert result.frequency_hz == 153_350_000
    assert result.metadata["fields"]["address"] == 1234


def test_normal_decoder_backend_api_exposes_onair_events(tmp_path):
    with onair_server() as server:
        backend = OnAirEventBackend(server.url, token="test-secret", start_worker=False)
        app = create_app(tmp_path / "state", decoder_backends=(backend,))
        backend.poll_once()
        server.emit(decoded_event())
        backend.poll_once()

        async def exercise():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                status = await client.get("/api/decoder-backends")
                assert status.status_code == 200
                listed = next(x for x in status.json()["backends"] if x["backend_id"] == "onair")
                assert listed["connected"] is True
                assert listed["modes"] == []  # Live-only, not a selectable clip decoder.
                assert "event feed" in listed["note"]
                first = await client.get("/api/decoder-backends/onair/events")
                assert first.status_code == 200
                assert len(first.json()) == 1
                record = first.json()[0]
                assert record["backend_id"] == "onair"
                assert record["native_mode"] == "CW"
                assert record["frequency_hz"] == 7_074_000
                assert record["metadata"]["source"] == "onair-http-events-v1"
                assert record["products"][0]["content"] == "CQ TEST"
                second = await client.get("/api/decoder-backends/onair/events")
                assert second.json() == []
        try:
            asyncio.run(exercise())
        finally:
            backend.close()
