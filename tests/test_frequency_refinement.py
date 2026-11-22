"""Synthetic crowded-audio checks for conservative frequency refinement."""

from __future__ import annotations

import numpy as np
import pytest

from sigid.frequency_refinement import refine_frequency_region


RATE = 12000
DURATION = 14.0


def hopping_tones(
    low: float, *, seed: int, amplitude: float = 0.13, tone_spacing: float = 6.25,
    duration: float = DURATION,
) -> np.ndarray:
    """Uncoded frequency-hopping fixture; intentionally not a valid FT8 frame."""
    rng = np.random.default_rng(seed)
    n = round(duration * RATE)
    symbols = rng.integers(0, 8, size=round(duration / 0.16) + 1)
    tone = low + symbols[np.minimum((np.arange(n) / (RATE * 0.16)).astype(int), len(symbols) - 1)] * tone_spacing
    phase = 2 * np.pi * np.cumsum(tone) / RATE
    envelope = ((np.arange(n) / RATE >= 0.4) & (np.arange(n) / RATE < 13.3)).astype(float)
    return amplitude * envelope * np.sin(phase)


def test_two_nearby_hopping_bands_are_suggested_not_confirmed():
    rng = np.random.default_rng(4)
    samples = (
        hopping_tones(1000, seed=1)
        + hopping_tones(1075, seed=2, amplitude=0.10)
        + rng.normal(0, 0.003, round(DURATION * RATE))
    )
    result = refine_frequency_region(samples, RATE, 980, 1140)

    assert result["split_suggested"] is True, result
    assert result["status"] == "split_suggested"
    assert result["resolution_hz"] < 2.0
    assert len(result["subbands"]) == 2
    assert result["subbands"][0]["frequency_low_hz"] <= 1005
    assert 1030 <= result["subbands"][0]["frequency_high_hz"] <= 1060
    assert result["subbands"][1]["frequency_low_hz"] >= 1060
    assert result["subbands"][1]["frequency_high_hz"] >= 1107
    assert "not independent signals" in result["limitation"]


def test_single_hopping_signal_does_not_split_on_its_eight_tones():
    result = refine_frequency_region(hopping_tones(1050, seed=42), RATE, 1020, 1130)
    assert result["split_suggested"] is False
    assert result["subbands"] == []


def test_sparse_two_fsk_tones_are_not_mistaken_for_two_stations():
    n = round(DURATION * RATE)
    tones = np.where((np.arange(n) // 1600) % 2 == 0, 940.0, 1080.0)
    phase = 2 * np.pi * np.cumsum(tones) / RATE
    samples = 0.2 * np.sin(phase)
    result = refine_frequency_region(samples, RATE, 910, 1110)
    assert result["split_suggested"] is False


def test_noise_and_narrow_boxes_abstain():
    noise = np.random.default_rng(23).normal(0, 0.01, round(DURATION * RATE))
    assert refine_frequency_region(noise, RATE, 930, 1200)["split_suggested"] is False
    result = refine_frequency_region(hopping_tones(900, seed=3), RATE, 890, 945)
    assert result["split_suggested"] is False
    assert "narrow" in result["reason"]


def test_short_invalid_audio_is_handled_without_guessing():
    short = np.zeros(1200)
    assert refine_frequency_region(short, RATE, 900, 1200)["split_suggested"] is False
    with pytest.raises(ValueError, match="passband"):
        refine_frequency_region(np.zeros(24000), RATE, 2100, 3500)
    with pytest.raises(ValueError, match="finite"):
        refine_frequency_region(np.full(30000, np.nan), RATE, 900, 1200)
