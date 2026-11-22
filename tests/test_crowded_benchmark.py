"""Reproducible crowded-WAV metrics; no real audio is committed with tests."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from sigid.crowded_benchmark import evaluate_crowded_wav


def _hopping_audio(rate: int, center: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rate * 14
    symbols = rng.integers(0, 8, size=100)
    frequencies = center + symbols[(np.arange(n) // round(rate * 0.16)).clip(0, 99)] * 6.25
    phase = 2 * np.pi * np.cumsum(frequencies) / rate
    return np.sin(phase)


def _make_wav(path, rate=12000):
    rng = np.random.default_rng(7)
    samples = (
        0.13 * _hopping_audio(rate, 1000, 3)
        + 0.10 * _hopping_audio(rate, 1075, 5)
        + rng.normal(0, 0.003, rate * 14)
    )
    pcm = np.round(np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())


def test_benchmark_reports_improvement_only_with_reviewed_reference(tmp_path):
    path = tmp_path / "two-hopping-bands.wav"
    _make_wav(path)
    parent = [{"sample_start": 0, "sample_count": 168000,
               "frequency_low_hz": 980, "frequency_high_hz": 1140}]
    references = [
        {"sample_start": 0, "sample_count": 168000,
         "frequency_low_hz": 1000, "frequency_high_hz": 1044},
        {"sample_start": 0, "sample_count": 168000,
         "frequency_low_hz": 1075, "frequency_high_hz": 1119},
    ]
    result = evaluate_crowded_wav(path, {"detections": parent}, references=references)
    assert result["source_name"] == path.name
    assert result["wav_sha256"]
    assert result["split_suggestions"] == 1
    assert result["proposed_boxes"] == 2
    assert result["metrics"]["original"]["matched"] == 1
    assert result["metrics"]["refined"]["matched"] == 2
    assert result["metrics"]["refined"]["recall"] == 1.0
    no_reference = evaluate_crowded_wav(path, parent)
    assert no_reference["reference_provided"] is False
    assert no_reference["metrics"] is None


def test_benchmark_does_not_infer_truth_or_read_out_of_bounds(tmp_path):
    path = tmp_path / "audio.wav"
    _make_wav(path)
    box = {
        "start_seconds": 0, "end_seconds": 14,
        "frequency_low_hz": 980, "frequency_high_hz": 1140,
    }
    result = evaluate_crowded_wav(path, [box])
    assert result["regions_checked"] == 1
    assert result["results"][0]["original"]["sample_count"] == 168000

    invalid = {**box, "end_seconds": 15}
    with pytest.raises(ValueError, match="exceeds WAV"):
        evaluate_crowded_wav(path, [invalid])
