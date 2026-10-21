from __future__ import annotations

import numpy as np
import pytest

from sigid.demod import demodulate_2fsk, demodulate_bpsk, demodulate_mfsk, demodulate_qpsk
from sigid.synthetic import generate_2fsk_wav, generate_bpsk_wav, generate_mfsk_wav, generate_qpsk_wav
from sigid.wav import WavSource


def read_samples(path):
    return np.asarray([sample for block in WavSource(path).blocks() for sample in block.samples])


def test_2fsk_demodulator_recovers_uncoded_bits_and_soft_metrics(tmp_path):
    bits = "00110101100101101010" * 5
    path = tmp_path / "known-fsk.wav"
    generate_2fsk_wav(
        path,
        bits,
        sample_rate=12000,
        symbol_rate=100,
        tone_low_hz=1100,
        tone_high_hz=1400,
    )
    result = demodulate_2fsk(
        read_samples(path),
        12000,
        symbol_rate=100,
        tone_0_hz=1100,
        tone_1_hz=1400,
        sample_offset=2400,
    )
    assert result["demodulator"] == "noncoherent-2fsk-tone-correlator"
    assert result["bits"] == bits
    assert result["symbols"] == [int(bit) for bit in bits]
    assert result["symbol_count"] == len(bits)
    assert all(abs(metric) > 0.8 for metric in result["soft_metrics"])
    assert result["symbol_sample_bounds"][0][0] == 2400
    assert result["symbol_sample_bounds"][-1][1] <= 2400 + len(bits) * 120


def test_2fsk_demodulator_handles_timing_offset_and_awgn(tmp_path):
    bits = "0100111010011010" * 8
    path = tmp_path / "fsk.wav"
    generate_2fsk_wav(
        path,
        bits,
        sample_rate=12000,
        symbol_rate=100,
        tone_low_hz=900,
        tone_high_hz=1500,
    )
    samples = read_samples(path)
    rng = np.random.default_rng(42)
    noisy = samples + rng.normal(0, 0.035, size=samples.size)
    noisy = np.concatenate((np.zeros(8), noisy, np.zeros(8)))
    result = demodulate_2fsk(
        noisy,
        12000,
        symbol_rate=100,
        tone_0_hz=900,
        tone_1_hz=1500,
        timing_offset_samples=7.5,
        integration_fraction=0.7,
    )
    assert result["bits"] == bits

    automatic = demodulate_2fsk(
        noisy,
        12000,
        symbol_rate=100,
        tone_0_hz=900,
        tone_1_hz=1500,
        auto_timing=True,
        integration_fraction=0.7,
    )
    assert automatic["bits"] == bits, (automatic["timing_offset_samples"], automatic["symbol_count"])
    assert automatic["timing_offset_estimated"] is True
    assert automatic["timing_offset_samples"] == pytest.approx(8.0, abs=8.0)
    assert automatic["timing_search"]["phase_count"] == 32


def test_mfsk_demodulator_recovers_tone_symbols_bits_and_confidence(tmp_path):
    symbols = [0, 3, 1, 2, 3, 0, 2, 1] * 8
    path = tmp_path / "mfsk4.wav"
    generate_mfsk_wav(
        path,
        symbols,
        tone_count=4,
        sample_rate=12000,
        symbol_rate=50,
        base_tone_hz=1000,
        tone_spacing_hz=200,
    )
    result = demodulate_mfsk(
        read_samples(path),
        12000,
        symbol_rate=50,
        tone_frequencies_hz=[1000, 1200, 1400, 1600],
    )
    expected_bits = "".join(f"{symbol:02b}" for symbol in symbols)
    assert result["symbols"] == symbols
    assert result["bits"] == expected_bits
    assert result["tone_count"] == 4
    assert min(result["soft_confidence"]) > 0.8
    assert result["protocol"] is None


def test_mfsk_non_power_of_two_returns_symbols_without_invented_bits():
    samples = np.tile(np.sin(2 * np.pi * 1000 * np.arange(120) / 12000), 12)
    result = demodulate_mfsk(
        samples,
        12000,
        symbol_rate=100,
        tone_frequencies_hz=[1000, 1500, 2000],
    )
    assert result["symbols"] == [0] * 12
    assert result["bits"] is None
    assert result["bits_per_symbol"] is None


def test_bpsk_demodulator_recovers_bits_with_explicit_polarity_hypothesis(tmp_path):
    bits = "00110101100101101010" * 5
    path = tmp_path / "known-bpsk.wav"
    manifest = generate_bpsk_wav(
        path,
        bits,
        sample_rate=12000,
        symbol_rate=100,
        carrier_hz=1500,
        initial_phase_radians=0.73,
    )
    samples = read_samples(path)
    rng = np.random.default_rng(100)
    samples = samples + rng.normal(0, 0.04, size=samples.size)
    result = demodulate_bpsk(
        samples,
        12000,
        symbol_rate=100,
        carrier_hz=1500,
        sample_offset=2400,
    )
    assert result["bits"] == bits or result["inverted_bits"] == bits
    assert result["inverted_bits"] == "".join("1" if bit == "0" else "0" for bit in result["bits"])
    assert result["polarity_ambiguous"] is True
    assert len(result["soft_metrics"]) == len(bits)
    assert manifest["labels"]["protocol"] is None


def test_qpsk_demodulator_returns_four_rotation_candidates(tmp_path):
    bits = "000111100110" * 8
    path = tmp_path / "known-qpsk.wav"
    manifest = generate_qpsk_wav(
        path,
        bits,
        sample_rate=12000,
        symbol_rate=100,
        carrier_hz=1500,
        initial_phase_radians=0.91,
    )
    samples = read_samples(path)
    rng = np.random.default_rng(101)
    samples = samples + rng.normal(0, 0.025, size=samples.size)
    result = demodulate_qpsk(
        samples,
        12000,
        symbol_rate=100,
        carrier_hz=1500,
        sample_offset=1200,
    )
    candidates = [hypothesis["bits"] for hypothesis in result["symbol_hypotheses"]]
    assert bits in candidates
    assert len(result["symbol_hypotheses"]) == 4
    assert result["rotation_ambiguous"] is True
    assert result["symbol_count"] == len(bits) // 2
    assert min(result["soft_confidence"]) > 0.7
    assert manifest["labels"]["protocol"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"symbol_rate": 0},
        {"symbol_rate": 2000},
        {"tone_0_hz": 0},
        {"tone_1_hz": 6000},
        {"tone_0_hz": 1000, "tone_1_hz": 1000},
        {"timing_offset_samples": 120},
    ],
)
def test_2fsk_demodulator_rejects_invalid_parameters(kwargs):
    options = {"symbol_rate": 100, "tone_0_hz": 1000, "tone_1_hz": 1500}
    options.update(kwargs)
    with pytest.raises(ValueError):
        demodulate_2fsk(np.ones(240), 12000, **options)
