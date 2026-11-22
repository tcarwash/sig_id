"""Independent deterministic synthetic-audio tests for the BPSK31 decoder.

The waveform synthesizer below is authored for these tests only: it maps
text through the decoder's published varicode table into a differential
phase track, shapes symbol boundaries with a raised-cosine complex blend
(spec-plausible cosine shaping, amplitude dips on reversals only), and
modulates one audio carrier. Decoding runs end to end through
``DecodeWindow`` DSP; no test asserts on bare codeword parsing alone.
"""
from __future__ import annotations

import hashlib

import numpy as np

from sigid.libsigdecode.bpsk31 import (
    _CODE_TO_CHAR,
    _CHAR_TO_CODE,
    _split_codewords,
    BPSK31Decoder,
    MAX_CODEWORD_BITS,
    SYMBOL_RATE_BAUD,
)
from sigid.libsigdecode.core import DecodeWindow

SAMPLE_RATE = 8000
CARRIER_HZ = 1000.0
ROUND_TRIP_TEXT = "cq cq de ab1cd 73 "
PHASE0 = 0.7  # non-trivial absolute phase for the invariance control


def encode_bits(text: str) -> str:
    return "".join(_CHAR_TO_CODE[ch] + "00" for ch in text)


def synth_bpsk31(
    text: str,
    *,
    sample_rate: int = SAMPLE_RATE,
    carrier_hz: float = CARRIER_HZ,
    phase0: float = 0.0,
    negate: bool = False,
    leading_silence_s: float = 0.0,
    prefix_bits: str = "",
    noise_std: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic shaped-BPSK31 waveform for ``text`` (prefix_bits first)."""
    sps = int(round(sample_rate / SYMBOL_RATE_BAUD))
    assert abs(sps * SYMBOL_RATE_BAUD - sample_rate) < 1e-6
    bits = prefix_bits + encode_bits(text)
    phases = [phase0]
    for bit in bits:
        phases.append(phases[-1] + (np.pi if bit == "0" else 0.0))
    symbols = np.exp(1j * np.array(phases))
    base = np.repeat(symbols, sps).astype(np.complex128)
    # Raised-cosine blend of the complex amplitude across boundaries where
    # the phase changes; untouched mid-symbols keep full amplitude.
    width = max(4, sps // 16)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(width * 2) / (width * 2))
    out = base.copy()
    for k in range(1, len(symbols)):
        if abs(symbols[k] - symbols[k - 1]) > 1e-9:
            center = k * sps
            lo, hi = center - width, center + width
            out[lo:hi] = (1 - ramp) * base[lo - 1:hi - 1] + ramp * base[lo:hi]
    t = np.arange(out.size, dtype=np.float64) / sample_rate
    audio = np.real(out * np.exp(2j * np.pi * carrier_hz * t))
    if negate:
        audio = -audio
    if noise_std:
        rng = np.random.default_rng(seed)
        audio = audio + rng.normal(0.0, noise_std, audio.size)
    if leading_silence_s:
        pad = np.zeros(int(round(leading_silence_s * sample_rate)))
        audio = np.concatenate((pad, audio))
    return audio.astype(np.float64)


def decode(audio: np.ndarray, carrier_hz: float | None = CARRIER_HZ,
           sample_start: int = 0) -> object:
    return BPSK31Decoder().decode(DecodeWindow(
        audio, SAMPLE_RATE, sample_start=sample_start, carrier_hz=carrier_hz))


def check_token_timeline(result, sample_start: int, total: int) -> None:
    assert "".join(t.text for t in result.tokens) == result.text
    prev = sample_start
    for token in result.tokens:
        assert token.start_sample >= prev
        assert token.end_sample > token.start_sample
        assert sample_start <= token.start_sample
        assert token.end_sample <= sample_start + total
        assert 0.0 <= token.confidence_hint <= 1.0
        assert token.pattern and "00" not in token.pattern
        prev = token.start_sample


def test_round_trip_letters_digits_spaces():
    audio = synth_bpsk31(ROUND_TRIP_TEXT)
    result = decode(audio)
    assert result.status == "decoded", result.note
    assert result.text == ROUND_TRIP_TEXT
    assert result.decoder_id == "bpsk31"
    check_token_timeline(result, 0, audio.size)
    assert result.evidence["unmapped_codeword_count"] == 0
    assert result.evidence["overlong_runs_skipped"] == 0
    assert "no_fec_crc" in result.evidence["bit_and_frame_integrity"]


def test_polarity_inversion_decodes_identically():
    plain = decode(synth_bpsk31(ROUND_TRIP_TEXT))
    flipped = decode(synth_bpsk31(ROUND_TRIP_TEXT, negate=True))
    assert flipped.status == "decoded", flipped.note
    assert flipped.text == plain.text == ROUND_TRIP_TEXT


def test_absolute_phase_offset_decodes_identically():
    result = decode(synth_bpsk31(ROUND_TRIP_TEXT, phase0=PHASE0))
    assert result.status == "decoded", result.note
    assert result.text == ROUND_TRIP_TEXT


def test_fractional_symbol_timing_offset_decodes():
    # Leading silence shifts the symbol grid by 0.37 of a symbol.
    shift_s = 0.37 / SYMBOL_RATE_BAUD
    result = decode(synth_bpsk31(ROUND_TRIP_TEXT, leading_silence_s=shift_s))
    assert result.status == "decoded", result.note
    assert result.text == ROUND_TRIP_TEXT


def test_sample_start_offsets_are_absolute():
    audio = synth_bpsk31(ROUND_TRIP_TEXT)
    start = 48000
    result = decode(audio, sample_start=start)
    assert result.status == "decoded", result.note
    assert result.text == ROUND_TRIP_TEXT
    check_token_timeline(result, start, audio.size)
    assert result.tokens[0].start_sample >= start
    assert result.evidence["source_window_start_sample"] == start


def test_mid_stream_garbage_resyncs_to_partial():
    # 34 alternating bits hit the overlong bound exactly twice, leaving no
    # residue, so the trailing message still parses while status is partial.
    garbage = "01" * 17
    assert len(garbage) == 2 * (MAX_CODEWORD_BITS + 1)
    audio = synth_bpsk31("cq ", prefix_bits=encode_bits("de ") + garbage)
    result = decode(audio)
    assert result.status == "partial", result.note
    assert result.text == "de cq "
    assert result.evidence["overlong_runs_skipped"] == 2


def test_abstain_silence():
    result = decode(np.zeros(2 * SAMPLE_RATE))
    assert result.status == "no_decode"
    assert result.text == ""


def test_abstain_noise():
    rng = np.random.default_rng(7)
    result = decode(rng.normal(0.0, 0.3, 2 * SAMPLE_RATE))
    assert result.status == "no_decode"
    assert result.text == ""


def test_abstain_steady_carrier():
    t = np.arange(2 * SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    result = decode(0.8 * np.cos(2 * np.pi * CARRIER_HZ * t))
    assert result.status == "no_decode"
    assert result.text == ""


def test_abstain_carrier_outside_acquisition_bound():
    audio = synth_bpsk31(ROUND_TRIP_TEXT)
    result = decode(audio, carrier_hz=CARRIER_HZ + 6.0)
    assert result.status == "no_decode", result.note
    assert result.text == ""
    cfo = result.evidence.get("residual_cfo_hz")
    assert cfo is not None and 5.5 <= cfo <= 6.5, result.evidence


def test_abstain_far_wrong_carrier():
    audio = synth_bpsk31(ROUND_TRIP_TEXT)
    result = decode(audio, carrier_hz=CARRIER_HZ + 200.0)
    assert result.status == "no_decode"
    assert result.text == ""


def test_abstain_overlong_framing_only():
    audio = synth_bpsk31("", prefix_bits="01" * 100)
    result = decode(audio)
    assert result.status == "no_decode"
    assert result.text == ""
    assert result.evidence["overlong_runs_skipped"] > 0


def test_abstain_unrelated_fsk():
    # 45-baud two-tone FSK around the carrier must not read as BPSK31 text.
    sps = SAMPLE_RATE // 45
    rng = np.random.default_rng(3)
    tone_select = rng.integers(0, 2, 90)
    freqs = np.repeat(np.where(tone_select, CARRIER_HZ + 180.0, CARRIER_HZ), sps)
    phase = 2 * np.pi * np.cumsum(freqs) / SAMPLE_RATE
    audio = 0.8 * np.cos(phase)
    result = decode(audio[: 2 * SAMPLE_RATE])
    assert result.status == "no_decode"
    assert result.text == ""


def test_abstain_without_carrier_and_short_window():
    audio = synth_bpsk31(ROUND_TRIP_TEXT)
    no_carrier = decode(audio, carrier_hz=None)
    assert no_carrier.status == "no_decode"
    assert no_carrier.text == ""
    short = BPSK31Decoder().decode(DecodeWindow(
        audio[: SAMPLE_RATE // 4], SAMPLE_RATE, carrier_hz=CARRIER_HZ))
    assert short.status == "no_decode"
    assert short.text == ""


def test_codeword_splitter_resync_and_trailing():
    codewords, overlong, trailing = _split_codewords(
        [c == "1" for c in "1" + "00" + "01" * 10])
    assert [p for p, _, _ in codewords] == ["1"]
    assert overlong == 1
    assert trailing == 3


def test_unmapped_codeword_fails_closed():
    assert "010101" not in _CODE_TO_CHAR  # well-framed but undefined
    assert _CODE_TO_CHAR.get("010101") is None


def test_varicode_table_framing_properties():
    codes = list(_CODE_TO_CHAR)
    assert len(codes) == len(set(codes))
    assert all("00" not in code for code in codes)
    assert all(code.endswith("1") for code in codes)
    assert all(set(code) <= {"0", "1"} for code in codes)
    assert _CODE_TO_CHAR["1"] == " "
    assert len(_CODE_TO_CHAR) == 95  # full printable ASCII 32..126
    assert sorted(_CODE_TO_CHAR.values()) == [chr(c) for c in range(32, 127)]
    fingerprint = hashlib.sha256("\n".join(
        f"{char}:{code}" for char, code in sorted(_CHAR_TO_CODE.items())
    ).encode()).hexdigest()
    assert fingerprint == "565485c1e8daa9d7c3305867cef81659c3edf59a356e1fb5ba9d030bce691fac"
    assert {char: _CHAR_TO_CODE[char] for char in "pQZ$"} == {
        "p": "111111", "Q": "111011101", "Z": "1010101101", "$": "111011011",
    }
