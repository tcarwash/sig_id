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
    assert fsk_result[0]["drift_hz_per_second"] == pytest.approx(0.0, abs=1.0)


def test_detector_estimates_linear_carrier_drift(tmp_path):
    sample_rate = 8000
    time_s = np.arange(16_000) / sample_rate
    phase = 2 * np.pi * (900 * time_s + 0.5 * 25 * time_s**2)
    pcm = np.rint(0.8 * np.sin(phase) * 32767).astype("<i2")
    path = tmp_path / "drifting-carrier.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())

    candidates = detect_tracks(waterfall(path, fft_size=512, hop_size=128, max_rows=2048))
    assert len(candidates) == 1
    assert candidates[0]["drift_hz_per_second"] == pytest.approx(25.0, abs=8.0)


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


def test_detector_separates_two_simultaneous_fsk_signals(tmp_path):
    sources = []
    for name, low, high, bits in (
        ("lower", 1000, 1200, "01" * 100),
        ("upper", 2600, 2800, "10" * 100),
    ):
        path = tmp_path / f"{name}.wav"
        generate_2fsk_wav(
            path,
            bits,
            sample_rate=12000,
            symbol_rate=50,
            tone_low_hz=low,
            tone_high_hz=high,
            amplitude=0.25,
        )
        with wave.open(str(path), "rb") as wav:
            sources.append(np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.int32))

    mixture = np.clip(sources[0] + sources[1], -32768, 32767).astype("<i2")
    mixed_path = tmp_path / "two-signals.wav"
    with wave.open(str(mixed_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(12000)
        wav.writeframes(mixture.tobytes())

    candidates = detect_tracks(waterfall(mixed_path, fft_size=256, hop_size=64, max_rows=2048))
    assert len(candidates) == 2
    assert [candidate["modulation_hypothesis"] for candidate in candidates] == ["FSK-like", "FSK-like"]
    assert candidates[0]["frequency_high_hz"] < candidates[1]["frequency_low_hz"]
    assert candidates[0]["frequency_low_hz"] == pytest.approx(1000, abs=100)
    assert candidates[1]["frequency_low_hz"] == pytest.approx(2600, abs=100)
    assert all(candidate["protocol"] is None for candidate in candidates)


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
