from __future__ import annotations

import asyncio

import httpx
import pytest

from sigid.api import create_app
from sigid.synthetic import generate_2fsk_wav


def make_fixture(path):
    generate_2fsk_wav(
        path,
        "001101" * 10,
        sample_rate=8000,
        symbol_rate=100,
        tone_low_hz=1000,
        tone_high_hz=1500,
    )
    return path.read_bytes()


def test_upload_waterfall_manual_label_audio_and_jsonl_export(tmp_path):
    audio = make_fixture(tmp_path / "known-fsk.wav")
    app = create_app(tmp_path / "app-data")

    async def exercise():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/api/health")).json()["status"] == "ok"
            assert "Signal ID Workbench" in (await client.get("/")).text
            assert "drawWaterfall" in (await client.get("/app.js")).text
            assert "waterfall-layout" in (await client.get("/style.css")).text
            uploaded = await client.post(
                "/api/captures",
                files={"file": ("radio-passband.wav", audio, "audio/wav")},
            )
            assert uploaded.status_code == 200, uploaded.text
            capture = uploaded.json()
            assert capture["sample_rate"] == 8000
            assert capture["source_metadata"]["uploaded_filename"] == "radio-passband.wav"

            spectrum = await client.get(
                f"/api/captures/{capture['id']}/waterfall",
                params={"fft_size": 512, "hop_size": 128, "max_rows": 24},
            )
            assert spectrum.status_code == 200, spectrum.text
            data = spectrum.json()
            assert len(data["db"]) <= 24
            assert len(data["db"][0]) == 257
            assert data["sample_starts"] == sorted(data["sample_starts"])

            saved = await client.post(
                f"/api/captures/{capture['id']}/regions",
                json={
                    "sample_start": 100,
                    "sample_count": 500,
                    "frequency_low_hz": 900,
                    "frequency_high_hz": 1600,
                    "family": "FSK",
                    "method": "binary frequency shift keying",
                },
            )
            assert saved.status_code == 201, saved.text
            region = saved.json()
            assert region["family"] == "FSK"
            assert region["protocol"] is None
            assert region["review_state"] == "accepted"
            assert region["source"] == "human"

            playback = await client.get(f"/api/captures/{capture['id']}/audio")
            assert playback.content == audio
            regions = await client.get(f"/api/captures/{capture['id']}/regions")
            assert len(regions.json()) == 1
            exported = await client.get("/api/exports/regions.jsonl")
            assert exported.status_code == 200
            assert '"family": "FSK"' in exported.text
            assert exported.headers["content-type"].startswith("application/x-ndjson")

            export_path = tmp_path / "app-data" / "exports" / "dataset.jsonl"
            assert app.state.capture_store.export_jsonl(export_path) == 1
            assert '"protocol": null' in export_path.read_text()

    asyncio.run(exercise())


def test_rejects_non_wav_uploads_and_region_outside_capture(tmp_path):
    app = create_app(tmp_path / "app-data")
    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            bad = await client.post(
                "/api/captures", files={"file": ("audio.mp3", b"noise", "audio/mpeg")}
            )
            assert bad.status_code == 415
            content = make_fixture(tmp_path / "short.wav")
            uploaded = await client.post(
                "/api/captures", files={"file": ("short.wav", content, "audio/wav")}
            )
            capture_id = uploaded.json()["id"]
            bad_region = await client.post(
                f"/api/captures/{capture_id}/regions",
                json={"sample_start": 999999, "sample_count": 1},
            )
            assert bad_region.status_code == 422
            missing = await client.get("/api/captures/not-a-capture/waterfall")
            assert missing.status_code == 404
    asyncio.run(exercise())


def test_fldigi_manual_selection_status_and_incremental_receive_output(tmp_path):
    class FakeFldigi:
        def __init__(self):
            self.mode = "PSK31"
            self.output = "CQ DE TEST"

        def version(self):
            from sigid.fldigi import FldigiVersion
            return FldigiVersion("fldigi", "test")

        def current_modem(self):
            return self.mode

        def modem_names(self):
            return ["PSK31", "THOR16"]

        def select_modem(self, name):
            if name not in self.modem_names():
                raise ValueError("mode is not present in the Fldigi catalog")
            self.mode = name
            return self.mode

        def receive_text_length(self):
            return len(self.output)

        def receive_text(self, start=0, length=65536):
            return self.output[start : start + length].encode()

    app = create_app(tmp_path / "app-data", fldigi_client=FakeFldigi())

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            status = (await client.get("/api/fldigi/status")).json()
            assert status["connected"] is True
            assert status["audio_routed"] is None
            assert status["audio_route_managed"] is False
            assert (await client.get("/api/fldigi/modes")).json() == ["PSK31", "THOR16"]
            selected = await client.post("/api/fldigi/mode", json={"name": "THOR16"})
            assert selected.json()["current_mode"] == "THOR16"
            unknown = await client.post("/api/fldigi/mode", json={"name": "MYSTERY"})
            assert unknown.status_code == 422
            output = (await client.get("/api/fldigi/output", params={"start": 7})).json()
            assert output == {"current_mode": "THOR16", "start": 7, "end": 10, "text": "EST", "truncated": False}

    asyncio.run(exercise())
