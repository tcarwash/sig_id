from __future__ import annotations

import asyncio

import httpx

from sigid.api import create_app
from sigid.synthetic import generate_2fsk_wav, generate_bpsk_wav, generate_mfsk_wav, generate_qpsk_wav


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

            acquired = await client.post(
                f"/api/captures/{capture_id}/demod/2fsk",
                json={
                    "sample_start": 8,
                    "sample_count": len(bits) * 120 - 8,
                    "symbol_rate": 100,
                    "tone_0_hz": 1100,
                    "tone_1_hz": 1400,
                    "auto_timing": True,
                },
            )
            assert acquired.status_code == 200, acquired.text
            acquired_result = acquired.json()
            assert acquired_result["bits"] == bits[: acquired_result["symbol_count"]]
            assert acquired_result["timing_offset_estimated"] is True
            assert acquired_result["timing_search"]["phase_count"] == 32

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

            acquired = await client.post(
                f"/api/captures/{capture_id}/demod/mfsk",
                json={"sample_count": len(symbols) * 240, "auto_timing": True},
            )
            assert acquired.status_code == 200, acquired.text
            acquired_result = acquired.json()
            assert acquired_result["symbols"] == symbols
            assert acquired_result["timing_offset_estimated"] is True
            assert acquired_result["timing_search"]["phase_count"] == 32

            rate_searched = await client.post(
                f"/api/captures/{capture_id}/demod/mfsk",
                json={
                    "sample_count": len(symbols) * 240,
                    "auto_rate_search": True,
                    "rate_search_min": 12.5,
                    "rate_search_max": 200,
                    "rate_search_steps": 9,
                },
            )
            assert rate_searched.status_code == 200, rate_searched.text
            rate_result = rate_searched.json()
            assert abs(rate_result["symbol_rate"] - 50) < 1e-9
            assert rate_result["symbols"] == symbols
            assert rate_result["symbol_rate_estimated"] is True

            bounded = await client.post(
                f"/api/captures/{capture_id}/demod/mfsk",
                json={"tone_count": 64, "rate_search_steps": 24, "auto_rate_search": True},
            )
            assert bounded.status_code == 422
            assert "shorter region" in bounded.json()["detail"]

            unsupported_mapping = await client.post(
                f"/api/captures/{capture_id}/demod/mfsk",
                json={"tone_count": 3, "tone_spacing_hz": 150},
            )
            assert unsupported_mapping.status_code == 200
            assert unsupported_mapping.json()["bits"] is None

    asyncio.run(exercise())


def test_api_bpsk_preserves_both_global_polarity_hypotheses(tmp_path):
    bits = "010010111001" * 8
    wav_path = tmp_path / "known-bpsk.wav"
    generate_bpsk_wav(
        wav_path,
        bits,
        sample_rate=12000,
        symbol_rate=100,
        carrier_hz=1500,
        initial_phase_radians=0.5,
    )
    app = create_app(tmp_path / "app-data")

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            uploaded = await client.post(
                "/api/captures",
                files={"file": ("known-bpsk.wav", wav_path.read_bytes(), "audio/wav")},
            )
            capture_id = uploaded.json()["id"]
            decoded = await client.post(
                f"/api/captures/{capture_id}/demod/bpsk",
                json={"sample_count": len(bits) * 120, "carrier_hz": 1500},
            )
            assert decoded.status_code == 200, decoded.text
            result = decoded.json()
            assert result["bits"] == bits or result["inverted_bits"] == bits
            assert result["polarity_ambiguous"] is True
            assert result["protocol"] is None

    asyncio.run(exercise())


def test_api_qpsk_preserves_all_rotation_hypotheses(tmp_path):
    bits = "000111100110" * 8
    wav_path = tmp_path / "known-qpsk.wav"
    generate_qpsk_wav(
        wav_path,
        bits,
        sample_rate=12000,
        symbol_rate=100,
        carrier_hz=1500,
        initial_phase_radians=0.3,
    )
    app = create_app(tmp_path / "app-data")

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            uploaded = await client.post(
                "/api/captures",
                files={"file": ("known-qpsk.wav", wav_path.read_bytes(), "audio/wav")},
            )
            capture_id = uploaded.json()["id"]
            decoded = await client.post(
                f"/api/captures/{capture_id}/demod/qpsk",
                json={"sample_count": len(bits) // 2 * 120, "carrier_hz": 1500},
            )
            assert decoded.status_code == 200, decoded.text
            result = decoded.json()
            assert bits in [item["bits"] for item in result["symbol_hypotheses"]]
            assert len(result["symbol_hypotheses"]) == 4
            assert result["rotation_ambiguous"] is True
            assert result["protocol"] is None

    asyncio.run(exercise())
