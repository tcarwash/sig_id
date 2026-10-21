from __future__ import annotations

import asyncio

import httpx

from sigid.api import create_app
from sigid.synthetic import generate_2fsk_wav, generate_mfsk_wav


def test_api_demodulates_saved_region_to_raw_bits_without_protocol_label(tmp_path):
    bits = "001101011001" * 8
    wav_path = tmp_path / "known-fsk.wav"
    generate_2fsk_wav(
        wav_path,
        bits,
        sample_rate=12000,
        symbol_rate=100,
        tone_low_hz=1100,
        tone_high_hz=1400,
    )
    app = create_app(tmp_path / "app-data")

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            uploaded = await client.post(
                "/api/captures",
                files={"file": ("known-fsk.wav", wav_path.read_bytes(), "audio/wav")},
            )
            capture_id = uploaded.json()["id"]
            decoded = await client.post(
                f"/api/captures/{capture_id}/demod/2fsk",
                json={
                    "sample_count": len(bits) * 120,
                    "symbol_rate": 100,
                    "tone_0_hz": 1100,
                    "tone_1_hz": 1400,
                },
            )
            assert decoded.status_code == 200, decoded.text
            result = decoded.json()
            assert result["bits"] == bits
            assert result["protocol"] is None
            assert result["symbol_count"] == len(bits)
            assert result["sample_start"] == 0
            assert result["symbol_sample_bounds"][0] == [0, 120]

            outside = await client.post(
                f"/api/captures/{capture_id}/demod/2fsk",
                json={"sample_start": len(bits) * 120 + 1},
            )
            assert outside.status_code == 422

    asyncio.run(exercise())


def test_api_demodulates_mfsk_symbols_and_power_of_two_bits(tmp_path):
    symbols = [0, 3, 1, 2] * 12
    wav_path = tmp_path / "known-mfsk.wav"
    generate_mfsk_wav(
        wav_path,
        symbols,
        tone_count=4,
        sample_rate=12000,
        symbol_rate=50,
        base_tone_hz=1000,
        tone_spacing_hz=200,
    )
    app = create_app(tmp_path / "app-data")

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            uploaded = await client.post(
                "/api/captures",
                files={"file": ("known-mfsk.wav", wav_path.read_bytes(), "audio/wav")},
            )
            capture_id = uploaded.json()["id"]
            decoded = await client.post(
                f"/api/captures/{capture_id}/demod/mfsk",
                json={"sample_count": len(symbols) * 240},
            )
            assert decoded.status_code == 200, decoded.text
            result = decoded.json()
            assert result["symbols"] == symbols
            assert result["bits"] == "".join(f"{symbol:02b}" for symbol in symbols)
            assert result["tone_frequencies_hz"] == [1000, 1200, 1400, 1600]
            assert result["protocol"] is None

            unsupported_mapping = await client.post(
                f"/api/captures/{capture_id}/demod/mfsk",
                json={"tone_count": 3, "tone_spacing_hz": 150},
            )
            assert unsupported_mapping.status_code == 200
            assert unsupported_mapping.json()["bits"] is None

    asyncio.run(exercise())
