"""Independent test vectors for the in-project Morse decoder."""
from __future__ import annotations

import json
import math
import wave

import httpx
import numpy as np
import pytest

from sigid.libsigdecode import (
    CWDecoder, CWSettings, DecodeWindow, DecoderLibrary, create_library,
)
from sigid.libsigdecode.cw import MORSE
from sigid.local_decoder_backend import LocalDecoderBackend

REVERSE = {letter: dots for dots, letter in MORSE.items() if len(letter) == 1}


def cw_fixture(
    text: str, rate: int = 12000, wpm: float = 20.0, carrier: float = 1100.0,
    amplitude: float = 0.5, noise: float = 0, adjacent: bool = False,
) -> np.ndarray:
    """Independent textbook Morse timing generator, not copied from OnAir."""
    dot = 1.2 / wpm
    mark: list[tuple[bool, float]] = [(False, 0.25)]
    words = text.upper().split(" ")
    for wi, word in enumerate(words):
        for ci, letter in enumerate(word):
            for j, symbol in enumerate(REVERSE[letter]):
                mark.append((True, dot * (3 if symbol == "-" else 1)))
                if j < len(REVERSE[letter]) - 1:
                    mark.append((False, dot))
            if ci < len(word) - 1:
                mark.append((False, 3 * dot))
        if wi < len(words) - 1:
            mark.append((False, 7 * dot))
    mark.append((False, 0.3))
    gate = np.concatenate([
        np.full(round(seconds * rate), 1.0 if pressed else 0.0, dtype=np.float64)
        for pressed, seconds in mark
    ])
    t = np.arange(len(gate), dtype=float) / rate
    # Ten millisecond ramp mitigates click energy independently of the CW detector.
    edge = min(1, round(0.007 * rate))
    if edge > 1:
        from scipy.ndimage import uniform_filter1d
        gate = uniform_filter1d(gate, size=edge)
    out = amplitude * gate * np.sin(2 * np.pi * carrier * t)
    if adjacent:
        out += 0.18 * np.sin(2 * np.pi * (carrier + 210) * t)
    if noise:
        out += np.random.default_rng(12).normal(0, noise, len(out))
    return np.asarray(out, dtype=np.float64)


@pytest.mark.parametrize("text,wpm", [
    ("CQ TEST", 20.0), ("SOS", 30.0), ("DE AG7SU", 15.0),
    ("CQ CQ", 24.0),
])
def test_textbook_cw_audio_round_trip(text, wpm):
    x = cw_fixture(text, wpm=wpm, noise=0.003, adjacent=True)
    result = create_library().decode("cw", DecodeWindow(x, 12000, 17_000, 1100.0))
    assert result.text == text, result.to_dict()
    assert result.status == "decoded"
    assert all(token.start_sample >= 17_000 and token.end_sample > token.start_sample for token in result.tokens)
    assert result.evidence["estimated_wpm"] == pytest.approx(wpm, abs=4.5)
    assert result.evidence["confidence_kind"] == "timing_fit_heuristic_not_probability"
    assert result.evidence["mean_symbol_timing_fit"] >= 0.65


def test_uncertain_and_unkeyed_sources_are_not_verified_decodes():
    t = np.arange(12000 * 8) / 12000
    sine = np.sin(2 * np.pi * 1100 * t) * 0.4
    for audio in [sine, np.zeros_like(sine), np.random.default_rng(8).normal(0, 0.001, len(sine))]:
        result = CWDecoder().decode(DecodeWindow(audio, 12000, carrier_hz=1100))
        assert result.status == "no_decode", result.to_dict()
        assert result.text == ""
    missing_carrier = CWDecoder().decode(DecodeWindow(cw_fixture("CQ"), 12000))
    assert missing_carrier.status == "no_decode"
    assert "carrier" in missing_carrier.note.lower()


def test_registry_and_configuration_are_explicit():
    lib = create_library()
    assert [s.id for s in lib.modes()] == ["cw", "rtty", "bpsk31", "ax25"]
    with pytest.raises(KeyError):
        lib.decode("ft8", DecodeWindow(cw_fixture("CQ"), 12000, carrier_hz=1100))
    with pytest.raises(ValueError):
        lib.register(CWDecoder())
    with pytest.raises(ValueError):
        CWSettings(wpm=200)
    with pytest.raises(ValueError):
        DecodeWindow(np.zeros((2, 4)), 12000, carrier_hz=1100)
    with pytest.raises(ValueError):
        DecodeWindow(np.zeros(1024), 12000, carrier_hz=6500)


def test_file_adapter_does_not_need_external_application(tmp_path):
    output = tmp_path / "cw.wav"
    audio = cw_fixture("CQ TEST")
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(12000)
        wav.writeframes(np.clip(np.rint(audio * 32767), -32768, 32767).astype("<i2").tobytes())
    from sigid.decoder_backend import DecoderRequest, DecoderBackendRegistry
    external = DecoderBackendRegistry((LocalDecoderBackend(),))
    modes = external.info()[0]
    assert modes["connected"] and modes["backend_id"] == "sigdecode"
    assert modes["modes"][0]["mode_id"] == "cw"
    result = external.decode("sigdecode", DecoderRequest(
        audio_path=output, sample_rate=12000, mode_id="cw", native_mode="cw",
        carrier_hz=1100, sample_start=4000, sample_count=len(audio),
    ))
    assert result.backend_id == "sigdecode"
    assert result.status == "completed"
    assert result.products[0].content == "CQ TEST"
    parsed = json.loads(result.products[-1].content)
    assert parsed["external_decoder_used"] is False
    assert parsed["tokens"][0]["start_sample"] >= 4000


def test_saved_capture_decodes_with_our_library_without_external_apps(tmp_path):
    import asyncio
    from sigid.api import create_app

    data = cw_fixture("CQ TEST", noise=0.002, adjacent=True)
    clip = tmp_path / "clean-owned-cw.wav"
    with wave.open(str(clip), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(12000)
        wav.writeframes(np.rint(np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())
    app = create_app(tmp_path / "data")

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            backends = (await client.get("/api/decoder-backends")).json()["backends"]
            internal = next(item for item in backends if item["backend_id"] == "sigdecode")
            assert internal["connected"]
            assert any(mode["mode_id"] == "cw" and mode["selectable"] for mode in internal["modes"])
            assert any(item["backend_id"] == "fldigi" for item in backends)
            assert any(item["backend_id"] == "wsjtx" for item in backends)
            assert any(item["backend_id"] == "onair" for item in backends)
            upload = await client.post(
                "/api/captures",
                files={"file": ("cw.wav", clip.read_bytes(), "audio/wav")},
            )
            assert upload.status_code == 200, upload.text
            capture_id = upload.json()["id"]
            detector = {
                "detector": "yolo",
                "model_sha256": "a" * 64,
                "model_version": "sha256:" + "a" * 64,
                "confidence": 0.72,
                "box_xyxy_normalized": [0.1, 0.2, 0.4, 0.7],
                "image_width": 640, "image_height": 360,
                "preprocessing_version": "test-roi-v1",
                "window_sample_start": 0,
                "window_sample_count": len(data),
                "sample_rate": 12000,
            }
            decoded = await client.post(
                f"/api/captures/{capture_id}/decoder-backends/sigdecode/decode",
                json={
                    "mode_id": "cw", "sample_start": 0,
                    "sample_count": len(data), "carrier_hz": 1100.0,
                    "frequency_low_hz": 1050.0, "frequency_high_hz": 1150.0,
                    "detector_evidence": detector,
                },
            )
            assert decoded.status_code == 200, decoded.text
            result = decoded.json()
            assert result["backend_id"] == "sigdecode"
            assert result["status"] == "completed"
            assert result["products"][0]["content"] == "CQ TEST"
            evidence = json.loads(result["products"][1]["content"])
            assert evidence["source"] == "sigid-independent-implementation"
            assert evidence["tokens"]
            assert evidence["external_decoder_used"] is False
            assert result["run_id"]
            runs = (await client.get(f"/api/captures/{capture_id}/decode-runs")).json()
            native_run = next(run for run in runs if run["id"] == result["run_id"])
            assert native_run["backend"] == "sigdecode"
            assert native_run["decoder_result"]["decoder_id"] == "cw"
            assert native_run["decoder_result"]["status"] == "decoded"
            assert native_run["decoder_result"]["tokens"]
            assert native_run["decoder_result"]["evidence"]["estimated_wpm"] > 0
            assert native_run["decoder_result"]["source"] == "sigid-independent-implementation"
            assert native_run["detector_evidence"]["native_trial_detector_source"] == "client_submitted"
            assert native_run["detector_evidence"]["selection_bounds"] == {
                "frequency_low_hz": 1050.0, "frequency_high_hz": 1150.0,
            }
            assert native_run["detector_evidence"]["model_sha256"] == "a" * 64
            assert native_run["classifier_evidence"] == {}

            from sigid.dataset import SignalLabel
            region_id = app.state.capture_store.add_region(
                capture_id, sample_start=0, sample_count=len(data),
                frequency_low_hz=1050.0, frequency_high_hz=1150.0,
                label=SignalLabel(evidence={
                    "classifier_proposal": {
                        "method": "2-FSK", "confidence": 0.51,
                        "model_sha256": "b" * 64,
                    },
                    "detector_proposal": detector,
                }),
            )
            region_decoded = await client.post(
                f"/api/captures/{capture_id}/decoder-backends/sigdecode/decode",
                json={"mode_id": "cw", "sample_start": 0,
                      "sample_count": len(data), "carrier_hz": 1100.0,
                      "region_id": region_id},
            )
            assert region_decoded.status_code == 200, region_decoded.text
            saved_runs = (await client.get(
                f"/api/captures/{capture_id}/decode-runs"
            )).json()
            saved_run = next(run for run in saved_runs
                             if run["id"] == region_decoded.json()["run_id"])
            assert saved_run["region_id"] == region_id
            assert saved_run["classifier_evidence"]["model_sha256"] == "b" * 64
            assert saved_run["classifier_evidence"]["method"] == "2-FSK"
            assert saved_run["detector_evidence"]["native_trial_detector_source"] == "linked_region"
            assert saved_run["detector_evidence"]["model_sha256"] == "a" * 64
            wrong_link = await client.post(
                f"/api/captures/{capture_id}/decoder-backends/sigdecode/decode",
                json={"mode_id": "cw", "sample_start": 1,
                      "sample_count": len(data) - 1, "region_id": region_id},
            )
            assert wrong_link.status_code == 422, wrong_link.text

            wrong_rate = await client.post(
                f"/api/captures/{capture_id}/decoder-backends/sigdecode/decode",
                json={"mode_id": "cw", "sample_start": 0, "sample_count": len(data),
                      "detector_evidence": {**detector, "sample_rate": 8000}},
            )
            assert wrong_rate.status_code == 422, wrong_rate.text
            wrong_window = await client.post(
                f"/api/captures/{capture_id}/decoder-backends/sigdecode/decode",
                json={"mode_id": "cw", "sample_start": 0, "sample_count": len(data),
                      "detector_evidence": {**detector, "window_sample_count": len(data) // 2}},
            )
            assert wrong_window.status_code == 422, wrong_window.text
            wrong_support = await client.post(
                f"/api/captures/{capture_id}/decoder-backends/sigdecode/decode",
                json={"mode_id": "cw", "sample_start": 0, "sample_count": len(data),
                      "detector_evidence": {**detector, "supporting_windows": [
                          {"sample_start": len(data) - 10, "sample_count": 100},
                      ]}},
            )
            assert wrong_support.status_code == 422, wrong_support.text
            wrong_band = await client.post(
                f"/api/captures/{capture_id}/decoder-backends/sigdecode/decode",
                json={"mode_id": "cw", "sample_start": 0, "sample_count": len(data),
                      "frequency_low_hz": 1050.0, "frequency_high_hz": 8000.0},
            )
            assert wrong_band.status_code == 422, wrong_band.text

            abstained = await client.post(
                f"/api/captures/{capture_id}/decoder-backends/sigdecode/decode",
                json={"mode_id": "cw", "sample_start": 0,
                      "sample_count": len(data), "carrier_hz": None},
            )
            assert abstained.status_code == 200, abstained.text
            abstained_runs = (await client.get(
                f"/api/captures/{capture_id}/decode-runs"
            )).json()
            abstained_run = next(run for run in abstained_runs
                                 if run["id"] == abstained.json()["run_id"])
            assert abstained_run["decoder_result"]["status"] == "no_decode"
            assert abstained_run["decoder_result"]["text"] == ""
            assert abstained_run["decoder_result"]["note"]

            invalid = await client.post(
                f"/api/captures/{capture_id}/decoder-backends/sigdecode/decode",
                json={
                    "mode_id": "ft8", "sample_start": 0,
                    "sample_count": len(data), "carrier_hz": 1100.0,
                },
            )
            assert invalid.status_code == 422

    asyncio.run(exercise())



def test_frequency_hopping_false_cw_candidates_abstain():
    rate = 12000
    count = rate * 12
    t = np.arange(count) / rate
    rng = np.random.default_rng(15)
    tones = rng.integers(0, 8, 95)
    freq = 1100 + 6.25 * tones[np.minimum((t / 0.16).astype(int), 94)]
    phase = 2 * np.pi * np.cumsum(freq) / rate
    audio = 0.3 * np.sin(phase) + np.random.default_rng(16).normal(0, 0.001, count)
    # A tone that happens to recur in FSK is not Morse just because
    # filtering it creates an apparently on/off-keyed envelope.
    for carrier in (1100, 1112.5, 1125):
        result = create_library().decode("cw", DecodeWindow(audio, rate, carrier_hz=carrier))
        assert result.status == "no_decode", result.to_dict()
        assert not result.text


def test_iq_only_mode_is_not_advertised_as_audio_selectable(tmp_path):
    from sigid.libsigdecode.core import DecoderSpec
    from sigid.decoder_backend import DecoderRequest

    class IQOnlyDecoder:
        spec = DecoderSpec(id="iq_test", name="IQ test", family="test",
                           input_kind="complex-iq")

        def decode(self, window):
            raise AssertionError("IQ-only decoder must never receive mono audio")

    backend = LocalDecoderBackend(DecoderLibrary((IQOnlyDecoder(),)))
    mode = backend.info().modes[0]
    assert not mode.selectable
    assert "complex-iq" in mode.note
    with pytest.raises(ValueError, match="audio decoding unsupported"):
        backend.decode(DecoderRequest(
            audio_path=tmp_path / "not-read.wav", sample_rate=12000,
            mode_id="iq_test", native_mode="iq_test",
        ))
