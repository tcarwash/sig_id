"""Reproducible chain benchmark uses truth only for scoring, never acquisition."""
from __future__ import annotations

import numpy as np

from tools.bench_signal_chain_snr import (
    detection_matches, narrow_tone_metrics, score_case, summary,
)


def test_geometric_matching_needs_time_and_frequency_overlap():
    truth = {"start_sample": 12000, "sample_count": 12000,
             "occupied_low_hz": 610.0, "occupied_high_hz": 870.0}
    assert detection_matches({
        "sample_start": 14000, "sample_count": 9000,
        "frequency_low_hz": 650, "frequency_high_hz": 900,
    }, truth, 12000)
    assert not detection_matches({
        "sample_start": 34000, "sample_count": 9000,
        "frequency_low_hz": 650, "frequency_high_hz": 900,
    }, truth, 12000)
    assert not detection_matches({
        "sample_start": 14000, "sample_count": 9000,
        "frequency_low_hz": 1800, "frequency_high_hz": 1900,
    }, truth, 12000)


def test_snr_chain_strong_rtty_has_correct_midburst_recovery_and_noise_abstention(tmp_path):
    row = score_case(mode="rtty", snr_db=25.0, seed=71062, output_dir=tmp_path)
    assert row["oracle"]["correct"] is True
    assert row["midburst_four_second"]["fully_embedded_messages"] >= 1
    assert row["midburst_four_second"]["correct_payload"] is True
    assert row["midburst_four_second"]["sample_count"] == 4*12000
    assert row["noise_only"]["false_output"] is False
    assert row["cv"] is None
    assert row["mono_sha256"] != row["fixture_sha256"]
    s = summary([row])["by_mode"]["rtty"]
    assert s["count"] == 1
    assert s["four_sec_contains_payload"] == 1
    assert s["four_sec_evaluable_complete_message"] == 1
    assert s["noise_false_outputs"] == 0


def test_noise_only_narrow_tone_benchmark_remains_unidentified():
    rate = 12000
    samples = np.rint(np.random.default_rng(42).normal(0, 0.02, rate*10)
                       * 32767).clip(-32768,32767).astype("<i2")
    result = narrow_tone_metrics(samples, rate, start=0, stop=len(samples), truth=None)
    assert result["matched_truth"] is False
    assert result["narrow_candidates"] >= 0
    assert result["wall_ms"] >= 0
