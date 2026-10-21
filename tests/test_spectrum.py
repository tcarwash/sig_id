from __future__ import annotations

import pytest
import numpy as np

from sigid.spectrum import waterfall, waterfall_samples
from sigid.synthetic import generate_2fsk_wav
from sigid.wav import WavSource


def test_waterfall_locates_synthetic_tone_and_preserves_sample_positions(tmp_path):
    path = tmp_path / "tone.wav"
    generate_2fsk_wav(
        path,
        "0" * 100,
        sample_rate=8000,
        symbol_rate=100,
        tone_low_hz=1000,
        tone_high_hz=1500,
    )
    result = waterfall(path, fft_size=512, hop_size=128, max_rows=32)
    assert len(result["db"]) <= 32
    assert result["sample_starts"] == sorted(set(result["sample_starts"]))
    assert all(0 <= start <= result["sample_count"] - 512 for start in result["sample_starts"])
    peak = max(range(len(result["frequency_hz"])), key=lambda i: result["db"][5][i])
    assert result["frequency_hz"][peak] == pytest.approx(1000, abs=8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fft_size": 500},
        {"fft_size": 16},
        {"hop_size": 0},
        {"max_rows": 0},
    ],
)
def test_waterfall_rejects_invalid_analysis_settings(tmp_path, kwargs):
    path = tmp_path / "short.wav"
    generate_2fsk_wav(path, "0", sample_rate=8000, symbol_rate=100)
    with pytest.raises(ValueError):
        waterfall(path, **kwargs)


def test_waterfall_rejects_recording_shorter_than_fft(tmp_path):
    path = tmp_path / "short.wav"
    generate_2fsk_wav(path, "0", sample_rate=8000, symbol_rate=100)
    with pytest.raises(ValueError, match="at least"):
        waterfall(path, fft_size=256)


def test_in_memory_waterfall_matches_wav_with_absolute_sample_offset(tmp_path):
    path = tmp_path / "tone.wav"
    generate_2fsk_wav(path, "01" * 20, sample_rate=8000, symbol_rate=100)
    recorded = waterfall(path, fft_size=256, hop_size=64, max_rows=64)
    source = WavSource(path)
    samples = [sample for block in source.blocks() for sample in block.samples]
    live = waterfall_samples(
        samples,
        8000,
        fft_size=256,
        hop_size=64,
        max_rows=64,
        sample_offset=5000,
        total_sample_count=5000 + len(samples),
    )
    assert live["sample_starts"] == [5000 + start for start in recorded["sample_starts"]]
    assert live["sample_count"] == 5000 + recorded["sample_count"]
    np.testing.assert_allclose(live["db"], recorded["db"], atol=0.0001)
