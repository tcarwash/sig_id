"""Contract, validation and HTTP bridge tests for future packet decoders.

The fixture simulates a decoder result. It does not claim an actual AX.25,
FT8, or other over-the-air waveform decoder exists.
"""
from __future__ import annotations

import asyncio
import base64
import json
import wave

import httpx
import numpy as np
import pytest

from sigid.libsigdecode import (
    DecodeWindow, DecoderLibrary, DecoderSpec, LocalDecodeResult,
    DecodedFrame, IntegrityCheck, SoftBitSpan,
)
from sigid.local_decoder_backend import LocalDecoderBackend
from sigid.decoder_backend import DecoderRequest


class FrameFixtureDecoder:
    spec = DecoderSpec(id="packet_fixture", name="Packet fixture", family="test")

    def __init__(self, *, abstain: bool = False) -> None:
        self.abstain = abstain

    def decode(self, window: DecodeWindow) -> LocalDecodeResult:
        origin = window.sample_start
        check = IntegrityCheck(
            "crc", "test-only CRC16", "failed" if self.abstain else "passed",
        )
        return LocalDecodeResult(
            "packet_fixture", "fixture-v1",
            status="no_decode" if self.abstain else "partial",
            note="Synthetic artifact fixture; no real protocol decoder.",
            frames=(DecodedFrame(
                origin + 100, origin + 500, b"\x01\x02\x03",
                bit_length=20, protocol_hint="synthetic-packet",
                integrity=(check,),
            ),),
            soft_bits=(SoftBitSpan(
                origin + 150, origin + 450,
                (0.10, 0.94, 0.77, 0.32), "probability_one",
            ),),
        )


def test_frame_and_soft_bits_are_typed_bounded_and_source_indexed():
    frame = DecodedFrame(1100, 1600, b"\x01\x02\x03", bit_length=20,
                         integrity=(IntegrityCheck("fec", "example code", "passed", corrected_bits=2),))
    details = frame.to_dict()
    assert details["payload_base64"] == "AQID"
    assert details["encoding"] == "base64"
    assert details["bit_length"] == 20
    assert details["integrity"][0]["corrected_bits"] == 2
    assert SoftBitSpan(1200, 1500, (-3.0, 0.0, 2.0), "llr_positive_is_one").to_dict()["bit_count"] == 3
    with pytest.raises(ValueError, match="bit_length"):
        DecodedFrame(100, 200, b"\xff", bit_length=9)
    with pytest.raises(ValueError, match="payload"):
        DecodedFrame(100, 200, bytes(8193))
    with pytest.raises(ValueError, match="span"):
        DecodedFrame(300, 200, b"\x00")
    with pytest.raises(ValueError, match="FEC"):
        IntegrityCheck("crc", "test", "passed", corrected_bits=3)
    with pytest.raises(ValueError, match="outcome"):
        IntegrityCheck("crc", "test", "verified")
    with pytest.raises(ValueError, match="not_checked"):
        IntegrityCheck("fec", "test", "not_checked", corrected_bits=1)
    with pytest.raises(ValueError, match="probability_one"):
        SoftBitSpan(100, 200, (0.3, 1.2), "probability_one")
    with pytest.raises(ValueError, match="finite"):
        SoftBitSpan(100, 200, (float("nan"),), "llr_positive_is_one")


def test_aggregate_evidence_limits_are_bounded():
    with pytest.raises(ValueError, match="too many or invalid frames"):
        LocalDecodeResult(
            "packet_fixture", "fixture-v1", "partial",
            frames=tuple(DecodedFrame(i, i + 1, b"x") for i in range(33)),
        )
    with pytest.raises(ValueError, match="too much or invalid soft-bit"):
        LocalDecodeResult(
            "packet_fixture", "fixture-v1", "partial",
            soft_bits=(SoftBitSpan(1, 200, (0.5,) * 8192, "probability_one"),
                       SoftBitSpan(201, 210, (0.5,), "probability_one")),
        )


def test_old_decoder_constructor_remains_compatible_and_schema_tracks_new_fields():
    from pathlib import Path

    result = LocalDecodeResult("cw", "v1", "no_decode", note="Not enough marks")
    assert result.to_dict()["frames"] == []
    assert result.to_dict()["soft_bits"] == []
    schema = json.loads((Path(__file__).parents[1] /
                         "docs/schemas/local_decode_result.schema.json").read_text())
    assert set(schema["required"]) == set(result.to_dict())
    assert schema["properties"]["frames"]["items"]["properties"]["encoding"]["const"] == "base64"
    assert schema["properties"]["soft_bits"]["items"]["properties"]["encoding"]["enum"] == [
        "llr_positive_is_one", "probability_one",
    ]
    with pytest.raises(ValueError, match="no_decode"):
        LocalDecodeResult("cw", "v1", "no_decode", text="fake")


def test_decoder_library_rejects_out_of_window_frame_and_soft_bit_evidence():
    window = DecodeWindow(np.zeros(12000), 12000, sample_start=4000)
    good = DecoderLibrary((FrameFixtureDecoder(),)).decode("packet_fixture", window)
    assert good.frames[0].start_sample == 4100
    assert good.soft_bits[0].end_sample == 4450

    class EscapingDecoder(FrameFixtureDecoder):
        def decode(self, window):
            result = super().decode(window)
            return LocalDecodeResult("packet_fixture", "fixture", "partial",
                frames=(DecodedFrame(window.sample_start - 1, window.sample_start + 50, b"x"),))

    class SoftEscapeDecoder(FrameFixtureDecoder):
        def decode(self, window):
            return LocalDecodeResult("packet_fixture", "fixture", "partial",
                soft_bits=(SoftBitSpan(window.sample_start, window.sample_start + len(window.samples) + 1,
                                       (0.4,), "probability_one"),))

    with pytest.raises(ValueError, match="frame extends"):
        DecoderLibrary((EscapingDecoder(),)).decode("packet_fixture", window)
    with pytest.raises(ValueError, match="soft-bit evidence extends"):
        DecoderLibrary((SoftEscapeDecoder(),)).decode("packet_fixture", window)


def test_binary_adapter_reports_frames_and_keeps_abstentions_in_metadata(tmp_path):
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(12000)
        wav.writeframes(np.zeros(12000, dtype="<i2").tobytes())
    request = DecoderRequest(
        audio_path=path, sample_rate=12000, mode_id="packet_fixture",
        native_mode="packet_fixture", sample_start=2500, sample_count=12000,
    )
    backend = LocalDecoderBackend(DecoderLibrary((FrameFixtureDecoder(),)))
    assert "binary" in backend.info().product_kinds
    decoded = backend.decode(request)
    assert decoded.status == "completed"
    assert [p.kind for p in decoded.products] == ["binary", "metadata"]
    assert decoded.products[0].content == b"\x01\x02\x03"
    assert decoded.products[0].metadata["bit_length"] == 20
    assert decoded.products[0].metadata["verified"] is False
    metadata = json.loads(decoded.products[-1].content)
    assert metadata["frames"][0]["start_sample"] == 2600
    assert metadata["soft_bits"][0]["encoding"] == "probability_one"

    abstained = LocalDecoderBackend(DecoderLibrary((FrameFixtureDecoder(abstain=True),))).decode(request)
    assert abstained.status == "completed_empty"
    assert [p.kind for p in abstained.products] == ["metadata"]
    metadata = json.loads(abstained.products[0].content)
    assert metadata["status"] == "no_decode"
    assert metadata["frames"][0]["integrity"][0]["outcome"] == "failed"


def test_http_saved_capture_persists_binary_and_soft_bit_evidence(tmp_path, monkeypatch):
    import sigid.api as api

    monkeypatch.setattr(
        api, "LocalDecoderBackend",
        lambda: LocalDecoderBackend(DecoderLibrary((FrameFixtureDecoder(),))),
    )
    path = tmp_path / "input.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(12000)
        wav.writeframes(np.zeros(18000, dtype="<i2").tobytes())
    app = api.create_app(tmp_path / "data")

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            upload = await client.post(
                "/api/captures",
                files={"file": ("test.wav", path.read_bytes(), "audio/wav")},
            )
            assert upload.status_code == 200, upload.text
            capture_id = upload.json()["id"]
            response = await client.post(
                f"/api/captures/{capture_id}/decoder-backends/sigdecode/decode",
                json={"mode_id": "packet_fixture", "sample_start": 2000,
                      "sample_count": 12000},
            )
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["status"] == "completed"
            binary = next(item for item in result["products"] if item["kind"] == "binary")
            assert binary["encoding"] == "base64"
            assert base64.b64decode(binary["content"]) == b"\x01\x02\x03"
            assert binary["metadata"]["verified"] is False
            runs = (await client.get(
                f"/api/captures/{capture_id}/decode-runs"
            )).json()
            run = next(item for item in runs if item["id"] == result["run_id"])
            assert run["decoder_result"]["frames"][0]["start_sample"] == 2100
            assert run["decoder_result"]["frames"][0]["payload_base64"] == "AQID"
            assert run["decoder_result"]["soft_bits"][0]["bit_count"] == 4
            assert run["output_text"] == ""
    asyncio.run(exercise())
