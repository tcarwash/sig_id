from __future__ import annotations

import asyncio

import httpx

from sigid.api import create_app
from sigid.synthetic import generate_mfsk_wav


def test_import_synthetic_wav_manifest_and_auto_label(tmp_path):
    source = tmp_path / "mfsk.wav"
    generate_mfsk_wav(source, [0, 1, 3, 2], tone_count=4)
    manifest = source.with_suffix(".wav.json").read_bytes()
    audio = source.read_bytes()
    app = create_app(tmp_path / "app-data")

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            page = await client.get("/")
            assert "SYNTHETIC DATASET" in page.text
            script = await client.get("/app.js")
            assert "/api/synthetic/import" in script.text
            result = await client.post(
                "/api/synthetic/import",
                files={
                    "wav_file": ("mfsk.wav", audio, "audio/wav"),
                    "manifest_file": ("mfsk.wav.json", manifest, "application/json"),
                },
            )
            assert result.status_code == 200, result.text
            data = result.json()
            capture = data["capture"]
            region = data["region"]
            assert capture["source_kind"] == "synthetic"
            assert capture["source_metadata"]["generator"] == "sig-id-mfsk"
            assert region["family"] == "MFSK"
            assert region["method"] == "4-ary frequency shift keying"
            assert region["protocol"] is None
            assert region["source"] == "synthetic"
            assert region["review_state"] == "accepted"
            assert region["evidence"]["manifest_sha256"]
            assert (await client.get(f"/api/captures/{capture['id']}/regions")).json()[0]["source"] == "synthetic"
            downloaded = await client.get(f"/api/captures/{capture['id']}/manifest")
            assert downloaded.status_code == 200
            assert downloaded.content == manifest

    asyncio.run(exercise())


def test_synthetic_api_rejects_mismatched_wav_and_manifest(tmp_path):
    source = tmp_path / "wrong.wav"
    generate_mfsk_wav(source, [0, 1, 2, 3], tone_count=4)
    audio = bytearray(source.read_bytes())
    audio[-2] ^= 0x01
    app = create_app(tmp_path / "app-data")

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/synthetic/import",
                files={
                    "wav_file": ("wrong.wav", bytes(audio), "audio/wav"),
                    "manifest_file": ("wrong.wav.json", source.with_suffix(".wav.json").read_bytes(), "application/json"),
                },
            )
            assert response.status_code == 422
            assert "do not match" in response.json()["detail"]
            assert app.state.capture_store.list_captures() == []

    asyncio.run(exercise())
