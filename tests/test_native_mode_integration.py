"""API integration smoke tests for first-party RTTY, PSK31, and AX.25.

Pinned waveforms are synthetic, not validated on-air receptions.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import wave

import httpx
import pytest

from sigid.api import create_app


FIXTURES = Path(__file__).parent / "fixtures" / "decoder_conformance"


@pytest.mark.parametrize("mode", ["rtty", "bpsk31", "ax25"])
def test_saved_capture_native_decoder_is_selectable_and_persists_evidence(mode, tmp_path):
    manifest = json.loads((FIXTURES / mode / "manifest.json").read_text())
    fixture = next(item for item in manifest["fixtures"] if item["expected"]["kind"] == "positive")
    audio = FIXTURES / mode / fixture["audio"]["path"]
    with wave.open(str(audio), "rb") as wav:
        count = wav.getnframes()
        assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
    app = create_app(tmp_path / "state")

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            backends = (await client.get("/api/decoder-backends")).json()["backends"]
            native = next(item for item in backends if item["backend_id"] == "sigdecode")
            assert any(item["mode_id"] == mode and item["selectable"]
                       for item in native["modes"])
            onair = next(item for item in backends if item["backend_id"] == "onair")
            assert onair["modes"] == []  # Event-only; no fake saved-audio support.

            uploaded = await client.post(
                "/api/captures",
                files={"file": (audio.name, audio.read_bytes(), "audio/wav")},
            )
            assert uploaded.status_code == 200, uploaded.text
            capture_id = uploaded.json()["id"]
            response = await client.post(
                f"/api/captures/{capture_id}/decoder-backends/sigdecode/decode",
                json={"mode_id": mode, "sample_start": 0, "sample_count": count,
                      "carrier_hz": fixture["carrier_hz"]},
            )
            assert response.status_code == 200, response.text
            value = response.json()
            assert value["backend_id"] == "sigdecode"
            assert value["status"] == "completed"
            runs = (await client.get(
                f"/api/captures/{capture_id}/decode-runs"
            )).json()
            saved = next(item for item in runs if item["id"] == value["run_id"])
            evidence = saved["decoder_result"]
            assert evidence["decoder_id"] == mode
            assert evidence["source"] == "sigid-independent-implementation"
            assert evidence["status"] == "decoded"
            assert evidence["text"] == fixture["expected"]["text"]
            if mode == "ax25":
                assert evidence["frames"]
                assert evidence["frames"][0]["integrity"][0]["outcome"] == "passed"
                assert any(item["kind"] == "binary" for item in value["products"])

    asyncio.run(exercise())
