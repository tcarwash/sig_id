import asyncio

import httpx
import pytest

from sigid.api import create_app
from sigid.synthetic import generate_bpsk_wav


def test_mode_search_api_and_user_added_bpsk31_label_resolve_consistently(tmp_path):
    wav = tmp_path / "source.wav"
    generate_bpsk_wav(wav, "00110110" * 16)
    app = create_app(tmp_path / "data")

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            search = (await client.get("/api/modes", params={"q": "psk31"})).json()
            assert [item["id"] for item in search["modes"][:2]] == ["bpsk31", "qpsk31"]
            capture = (await client.post("/api/captures", files={"file": ("source.wav", wav.read_bytes(), "audio/wav")})).json()
            region = await client.post(f"/api/captures/{capture['id']}/regions", json={
                "sample_start": 0, "sample_count": 1200, "origin": "user_added_missed_detection", "mode_id": "bpsk31",
            })
            assert region.status_code == 201, region.text
            body = region.json()
            assert (body["family"], body["method"], body["protocol"], body["variant"]) == ("PSK", "BPSK", "PSK31", "BPSK31")
            assert body["mode_id"] == "bpsk31"
            assert body["origin"] == "user_added_missed_detection"
            bad = await client.post(f"/api/captures/{capture['id']}/regions", json={
                "sample_start": 100, "sample_count": 1200, "mode_id": "bpsk31", "family": "FSK",
            })
            assert bad.status_code == 422

    asyncio.run(run())


def test_label_revision_keeps_canonical_mode_id(tmp_path):
    wav = tmp_path / "source.wav"
    generate_bpsk_wav(wav, "00110110" * 16)
    app = create_app(tmp_path / "data")

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            capture = (await client.post("/api/captures", files={"file": ("source.wav", wav.read_bytes(), "audio/wav")})).json()
            region = (await client.post(f"/api/captures/{capture['id']}/regions", json={"sample_start": 0, "sample_count": 1200})).json()
            revised = await client.post(f"/api/captures/{capture['id']}/regions/{region['id']}/labels", json={
                "expected_revision": 1, "mode_id": "bpsk31",
            })
            assert revised.status_code == 201, revised.text
            assert revised.json()["mode_id"] == "bpsk31"
            history = (await client.get(f"/api/captures/{capture['id']}/regions/{region['id']}/history")).json()
            assert history[-1]["mode_id"] == "bpsk31"

    asyncio.run(run())
