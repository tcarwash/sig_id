from __future__ import annotations

import wave

import numpy as np
import pytest

from sigid.detection import detect_tracks
from sigid.spectrum import waterfall
from sigid.synthetic import generate_2fsk_wav, generate_mfsk_wav


def test_detector_reports_provisional_fsk_and_mfsk_tone_evidence(tmp_path):
    mfsk_path = tmp_path / "mfsk4.wav"
    generate_mfsk_wav(
        mfsk_path,
        [0, 1, 2, 3] * 16,
        tone_count=4,
        sample_rate=12000,
        symbol_rate=50,
        base_tone_hz=1000,
        tone_spacing_hz=200,
    )
    mfsk_result = detect_tracks(waterfall(mfsk_path, fft_size=256, hop_size=64, max_rows=2048))
    assert len(mfsk_result) == 1
    candidate = mfsk_result[0]
    assert candidate["modulation_hypothesis"] == "MFSK-like"
    assert candidate["tone_count_candidate"] == 4
    assert candidate["tone_centers_hz"] == pytest.approx([1000, 1200, 1400, 1600], abs=30)
    assert candidate["identification_state"] == "provisional"
    assert candidate["protocol"] is None
    assert candidate["sample_start"] == 0
    assert candidate["sample_count"] == 16 * 4 * 240
    assert 0 < candidate["occupancy"] <= 1
    assert candidate["peak_snr_db"] > 20

    fsk_path = tmp_path / "fsk.wav"
    generate_2fsk_wav(
        fsk_path,
        "01" * 30,
        sample_rate=8000,
        symbol_rate=50,
        tone_low_hz=1000,
        tone_high_hz=1500,
    )
    fsk_result = detect_tracks(waterfall(fsk_path, fft_size=128, hop_size=32, max_rows=2048))
    assert len(fsk_result) == 1
    assert fsk_result[0]["modulation_hypothesis"] == "FSK-like"
    assert fsk_result[0]["tone_count_candidate"] == 2
    assert fsk_result[0]["tone_centers_hz"] == pytest.approx([1000, 1500], abs=1)


def test_detector_rejects_quiet_noise_as_a_signal(tmp_path):
    path = tmp_path / "quiet-noise.wav"
    rng = np.random.default_rng(20261021)
    noise = np.rint(rng.normal(0, 180, 16_000)).clip(-32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(noise.tobytes())
    assert detect_tracks(waterfall(path, fft_size=256, hop_size=64, max_rows=256)) == []


def test_detector_rejects_invalid_parameters_or_spectrum_shape():
    spectrum = {
        "db": [[-80.0, -20.0]],
        "frequency_hz": [0.0, 100.0],
        "sample_starts": [0],
        "sample_rate": 8000,
        "fft_size": 8,
        "sample_count": 8,
    }
    with pytest.raises(ValueError, match="threshold_db"):
        detect_tracks(spectrum, threshold_db=0)
    with pytest.raises(ValueError, match="do not agree"):
        detect_tracks({**spectrum, "sample_starts": []})
