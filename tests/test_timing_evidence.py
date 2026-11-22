"""Timing-evidence regression tests for uncoded FSK-like audio."""

from __future__ import annotations

import numpy as np
import pytest

from sigid.timing_evidence import measure_timing_evidence


def hopping_fixture(
    rate: int = 12000,
    seconds: float = 12.8,
    baud: float = 6.25,
    base_hz: float = 1100.0,
    seed: int = 5,
    amplitude: float = 0.14,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = round(rate * seconds)
    t = np.arange(count) / rate
    symbol_ids = rng.integers(0, 8, size=int(np.ceil(seconds * baud)) + 2)
    tones = base_hz + 6.25 * symbol_ids[np.minimum((t * baud).astype(int), len(symbol_ids) - 1)]
    phase = 2 * np.pi * np.cumsum(tones) / rate
    return amplitude * np.sin(phase) + 0.001 * rng.normal(size=count)


def test_recovers_six_point_two_five_baud_hopping_candidate():
    sample_rate = 12000
    samples = hopping_fixture()
    result = measure_timing_evidence(samples, sample_rate, 1085, 1160)
    assert result["available"], result
    assert result["transition_count"] >= 20, result
    assert result["symbol_rate_baud_candidate"] == pytest.approx(6.25, abs=0.2), result
    assert result["symbol_period_seconds_candidate"] == pytest.approx(0.16, abs=0.008), result
    assert result["tone_spacing_hz_candidate"] == pytest.approx(6.25, abs=1), result
    assert result["timing_coherence"] > 0.65
    assert "not protocol sync" in result["reason"]
    assert "not protocol" in result["limitation"] or "verifies a protocol" in result["limitation"]


def test_distinguishes_faster_hopping_clock():
    sample_rate = 12000
    result = measure_timing_evidence(
        hopping_fixture(baud=12.5, seed=8),
        sample_rate, 1085, 1160,
    )
    assert result["symbol_rate_baud_candidate"] == pytest.approx(12.5, abs=0.35), result


def test_silence_stationary_tone_and_wide_band_abstain():
    rate = 12000
    silence = measure_timing_evidence(np.zeros(rate * 8), rate, 1085, 1160)
    assert silence["available"] is False
    samples = 0.2 * np.sin(2 * np.pi * 1120 * np.arange(rate * 10) / rate)
    carrier = measure_timing_evidence(samples, rate, 1090, 1150)
    assert carrier["symbol_rate_baud_candidate"] is None
    wide = measure_timing_evidence(samples, rate, 900, 1600)
    assert wide["available"] is False
    assert "wide" in wide["reason"]


def test_unusable_inputs_are_rejected():
    with pytest.raises(ValueError, match="in band"):
        measure_timing_evidence(np.zeros(12000), 12000, 900, 7000)
    with pytest.raises(ValueError, match="finite"):
        measure_timing_evidence(np.full(12000 * 4, np.nan), 12000, 1000, 1100)
    short = measure_timing_evidence(np.zeros(12000), 12000, 1000, 1100)
    assert short["available"] is False
