"""Synthetic acceptance vectors for the independently implemented ITA2 path."""
from __future__ import annotations

import numpy as np
import pytest
import wave

from sigid.libsigdecode import DecodeWindow
from sigid.libsigdecode.rtty import RTTYDecoder, RTTYSettings
from sigid.decoder_backend import DecoderRequest
from sigid.local_decoder_backend import LocalDecoderBackend


LETTERS = {
    "A": 3, "C": 14, "D": 9, "E": 1, "G": 26, "I": 6, "J": 11,
    "L": 18, "N": 12, "O": 24, "Q": 23, "R": 10, "S": 5,
    "T": 16, "U": 7, "W": 19, "X": 29,
}
FIGURES = {"1": 23, "2": 19, "3": 1, "4": 10, "5": 16}
FIGS = 27
LTRS = 31
SPACE = 4


def _ita2_codes(text: str) -> list[int]:
    codes = [LTRS]
    figures = False
    for char in text:
        if char == " ":
            codes.append(SPACE)
            continue
        if char.isdigit():
            if not figures:
                codes.append(FIGS)
                figures = True
            codes.append(FIGURES[char])
        else:
            if figures:
                codes.append(LTRS)
                figures = False
            codes.append(LETTERS[char])
    return codes


def rtty_fixture(
    text: str, *, rate: int = 12000, baud: float = 45.45,
    shift: float = 170.0, carrier: float = 1500.0,
    stop_bits: float = 1.5, inverted: bool = False,
    noise: float = 0.0,
) -> np.ndarray:
    """Generate ideal continuous-phase 2-FSK from ITA2 code points."""
    unit = 1.0 / baud
    symbols: list[tuple[int, float]] = [(1, 4 * unit)]
    for code in _ita2_codes(text):
        symbols.append((0, unit))  # start
        symbols.extend(((code >> bit) & 1, unit) for bit in range(5))
        symbols.append((1, stop_bits * unit))
    symbols.append((1, 3 * unit))
    gates = np.concatenate([
        np.full(round(duration * rate), bit, dtype=np.float64)
        for bit, duration in symbols
    ])
    if inverted:
        gates = 1.0 - gates
    mark, space = carrier + shift / 2, carrier - shift / 2
    frequency = np.where(gates > 0.5, mark, space)
    phase = 2 * np.pi * np.cumsum(frequency) / rate
    audio = 0.65 * np.sin(phase)
    if noise:
        audio += np.random.default_rng(812).normal(0, noise, audio.size)
    return audio.astype(np.float64)


@pytest.mark.parametrize("inverted", [False, True])
def test_rtty_decodes_letters_figures_and_offsets(inverted):
    audio = rtty_fixture("CQ TEST 12345", inverted=inverted, noise=0.003)
    result = RTTYDecoder().decode(DecodeWindow(audio, 12000, 23_000, 1500.0))
    assert result.status == "decoded", result.to_dict()
    assert result.text == "CQ TEST 12345"
    assert result.evidence["winner_polarity"] == ("inverted" if inverted else "normal")
    assert result.evidence["winner_valid_stop_fraction"] >= 0.85
    assert all(token.start_sample >= 23_000 for token in result.tokens)
    assert all(token.end_sample > token.start_sample for token in result.tokens)


@pytest.mark.parametrize("stop_bits", [1.0, 1.4, 1.5, 2.0])
def test_rtty_framing_search_accepts_supported_stop_lengths(stop_bits):
    audio = rtty_fixture("CQ TEST", stop_bits=stop_bits)
    decoder = RTTYDecoder(RTTYSettings(stop_bits=1.5))
    result = decoder.decode(DecodeWindow(audio, 12000, carrier_hz=1500.0))
    assert result.status == "decoded", result.to_dict()
    assert result.text == "CQ TEST"
    assert result.evidence["winner_stop_bits"] == stop_bits


@pytest.mark.parametrize("baud,shift", [
    (baud, shift)
    for baud in (45.45, 50.0, 57.0, 75.0, 100.0)
    for shift in (170.0, 200.0, 425.0, 850.0)
])
def test_rtty_supported_baud_and_shift_profiles(baud, shift):
    audio = rtty_fixture("CQ TEST", baud=baud, shift=shift)
    result = RTTYDecoder(RTTYSettings(baud=baud, shift_hz=shift)).decode(
        DecodeWindow(audio, 12000, carrier_hz=1500.0),
    )
    assert result.status == "decoded", result.to_dict()
    assert result.text == "CQ TEST"


def test_rtty_j_is_letters_character_and_bell_only_in_figures():
    audio = rtty_fixture("J TEST")
    result = RTTYDecoder().decode(DecodeWindow(audio, 12000, carrier_hz=1500.0))
    assert result.status == "decoded", result.to_dict()
    assert result.text == "J TEST"
    assert result.evidence["bell_count"] == 0


def test_rtty_decimated_rate_keeps_tokens_on_original_sample_timeline():
    start = 123_456
    audio = rtty_fixture("J CQ", rate=48000)
    result = RTTYDecoder().decode(DecodeWindow(audio, 48000, start, 1500.0))
    assert result.status == "partial", result.to_dict()
    assert result.text == "J CQ"
    assert result.evidence["decimation_factor"] == 4
    assert result.tokens[0].start_sample >= start + 3 * round(48000 / 45.45)
    assert result.tokens[-1].end_sample <= start + len(audio)
    assert all(a.end_sample <= b.start_sample for a, b in zip(result.tokens, result.tokens[1:]))


def test_rtty_abstains_on_silence_noise_single_tone_and_wrong_frequency():
    rate = 12000
    rng = np.random.default_rng(81)
    silence = np.zeros(rate * 8)
    noise = rng.normal(0, 0.001, rate * 8)
    t = np.arange(rate * 8) / rate
    tone = 0.4 * np.sin(2 * np.pi * 1500 * t)
    valid = rtty_fixture("CQ TEST", rate=rate, carrier=2400.0)
    cases = [silence, noise, tone, valid]
    carriers = [1500.0, 1500.0, 1500.0, 1500.0]
    for samples, carrier in zip(cases, carriers):
        result = RTTYDecoder().decode(DecodeWindow(samples, rate, carrier_hz=carrier))
        assert result.status == "no_decode", result.to_dict()
        assert result.text == ""


def test_rtty_configuration_rejects_unsupported_profiles():
    with pytest.raises(ValueError):
        RTTYSettings(baud=300.0)
    with pytest.raises(ValueError):
        RTTYSettings(shift_hz=100.0)
    with pytest.raises(ValueError):
        RTTYSettings(stop_bits=1.25)


def test_rtty_enforces_fixed_memory_input_bound():
    audio = np.zeros(8_000_001, dtype=np.float64)
    result = RTTYDecoder().decode(DecodeWindow(audio, 384000, carrier_hz=1500.0))
    assert result.status == "no_decode"
    assert result.evidence["max_samples"] == 8_000_000


def test_rtty_is_available_through_the_local_decoder_backend(tmp_path):
    audio = rtty_fixture("CQ TEST 123", noise=0.002)
    path = tmp_path / "rtty.wav"
    pcm = np.clip(np.rint(audio * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(12000)
        wav.writeframes(pcm.tobytes())

    backend = LocalDecoderBackend()
    assert any(mode.mode_id == "rtty" and mode.selectable for mode in backend.info().modes)
    result = backend.decode(DecoderRequest(
        audio_path=path, sample_rate=12000, mode_id="rtty", native_mode="rtty",
        sample_count=len(audio), carrier_hz=1500.0,
    ))
    assert result.status == "completed"
    assert result.products[0].content == "CQ TEST 123"
    assert result.products[0].metadata["status"] == "decoded"
