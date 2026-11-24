"""Wave A CV preprocessing and live-load regression tests."""
from __future__ import annotations

from threading import Lock
import json

import numpy as np
import pytest

from sigid.cv_detector import (
    DB_CEILING, DB_FLOOR, FFT_SIZE, HOP_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH,
    MAX_FREQUENCY_HZ, MODEL_IMAGE_SIZE, WATERFALL_COLORS, YoloSignalDetector,
    _partial_background_noise_rms, spectrogram_image,
)
from sigid.cv_embeddings import EMBEDDING_PREPROCESSING_VERSION, load_embedding_index
from sigid.spectrum import waterfall_samples


def _old_spectrogram_image(samples, rate):
    spectrum = waterfall_samples(samples, rate, fft_size=FFT_SIZE, hop_size=HOP_SIZE,
                                 max_rows=IMAGE_HEIGHT)
    db = np.asarray(spectrum["db"], dtype=np.float32)
    frequencies = np.asarray(spectrum["frequency_hz"], dtype=np.float64)
    target_frequency = np.linspace(0.0, MAX_FREQUENCY_HZ, IMAGE_WIDTH)
    frequency_scaled = np.empty((db.shape[0], IMAGE_WIDTH), dtype=np.float32)
    for row_index, row in enumerate(db):
        frequency_scaled[row_index] = np.interp(target_frequency, frequencies, row)
    source_rows = np.arange(frequency_scaled.shape[0], dtype=np.float64)
    target_rows = np.linspace(max(0, len(source_rows) - 1), 0.0, IMAGE_HEIGHT)
    time_scaled = np.empty((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    for column in range(IMAGE_WIDTH):
        time_scaled[:, column] = np.interp(target_rows, source_rows, frequency_scaled[:, column])
    pixels = np.clip((time_scaled - DB_FLOOR) * (255.0 / (DB_CEILING - DB_FLOOR)), 0, 255)
    palette_position = pixels * (len(WATERFALL_COLORS) - 1) / 255.0
    palette_index = np.floor(palette_position).astype(np.int16)
    palette_index = np.clip(palette_index, 0, len(WATERFALL_COLORS) - 1)
    next_index = np.minimum(palette_index + 1, len(WATERFALL_COLORS) - 1)
    fraction = (palette_position - palette_index)[..., None]
    rgb = WATERFALL_COLORS[palette_index] * (1.0 - fraction) + WATERFALL_COLORS[next_index] * fraction
    return np.rint(rgb).astype(np.uint8)


@pytest.mark.parametrize("sample_rate,duration", [(8000, 0.6), (12000, 4.0), (48000, 10.0)])
def test_vectorized_spectrogram_is_pixel_identical_to_v2(sample_rate, duration):
    rng = np.random.default_rng(1729 + sample_rate)
    samples = rng.normal(0, 0.015, int(sample_rate * duration))
    samples += 0.1 * np.sin(np.arange(len(samples)) * (2 * np.pi * 1132 / sample_rate))
    old = _old_spectrogram_image(samples, sample_rate)
    new = spectrogram_image(samples, sample_rate)
    assert new.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    assert new.dtype == np.uint8
    np.testing.assert_array_equal(new, old)


def test_vectorized_waterfall_matches_sequential_frame_fft():
    rng = np.random.default_rng(800)
    rate = 12000
    samples = rng.normal(0, 0.05, 9 * rate)
    fft_size, hop_size, max_rows = 1024, 256, 121
    result = waterfall_samples(samples, rate, fft_size=fft_size, hop_size=hop_size,
                               max_rows=max_rows, sample_offset=320)
    frames = 1 + (len(samples) - fft_size) // hop_size
    stride = max(1, (frames + max_rows - 1) // max_rows)
    window = np.hanning(fft_size)
    starts = []
    rows = []
    for i in range(0, frames, stride):
        offset = i * hop_size
        power = np.abs(np.fft.rfft(samples[offset : offset + fft_size] * window)) / max(window.sum(), 1.0)
        rows.append((20.0 * np.log10(np.maximum(power, 1e-10))).astype(np.float32))
        starts.append(320 + offset)
    assert result["sample_starts"] == starts
    np.testing.assert_allclose(result["db"], rows, rtol=0, atol=1e-5)


def test_embed_image_uses_bgr_and_schema_versioned_index():
    assert EMBEDDING_PREPROCESSING_VERSION == "sig-id-cv-region-embedding-v3"

    class FakeModel:
        def __init__(self):
            self.arrays = []

        def embed(self, image, **kwargs):
            assert kwargs["imgsz"] == MODEL_IMAGE_SIZE
            self.arrays.append(image)
            return np.asarray([3.0, 4.0], dtype=np.float32)

    detector = object.__new__(YoloSignalDetector)
    detector._lock = Lock()
    detector._model = FakeModel()
    image = np.asarray([[[11, 22, 33], [44, 55, 66]]], dtype=np.uint8)
    vector = detector.embed_image(image)
    np.testing.assert_array_equal(detector._model.arrays[0],
                                  np.asarray([[[33, 22, 11], [66, 55, 44]]], dtype=np.uint8))
    assert detector._model.arrays[0].flags.c_contiguous
    np.testing.assert_allclose(vector, [0.6, 0.8])


def test_partial_noise_padding_ignores_strong_tail_carrier():
    rate = 12000
    rng = np.random.default_rng(47)
    t = np.arange(2 * rate) / rate
    observed = rng.normal(0.0, 0.004, len(t)) + 0.24 * np.sin(2 * np.pi * 1300 * t)
    rms = _partial_background_noise_rms(observed, rate)
    assert 0.0015 < rms < 0.008, rms
    assert rms < np.sqrt(np.mean(observed ** 2)) / 15

    class Probe(YoloSignalDetector):
        def detect(self, image, *, sample_start, sample_count, sample_rate, confidence=None):
            self.got = (image, sample_start, sample_count, sample_rate)
            return []

    probe = object.__new__(Probe)
    assert probe.detect_live_partial(observed, sample_start=0,
                                     total_sample_count=len(observed),
                                     sample_rate=rate) == []
    image, window_start, count, actual_rate = probe.got
    assert image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    assert count == rate * 10
    assert window_start == len(observed) - count
    assert actual_rate == rate


def test_previous_embedding_index_cannot_be_silently_reused(tmp_path):
    old = tmp_path / "old.npz"
    metadata = {
        "schema_version": 2, "model_sha256": "a" * 64,
        "preprocessing_version": "sig-id-cv-region-embedding-v2",
        "entries": [],
    }
    np.savez_compressed(old, vectors=np.empty((0, 2), dtype=np.float32),
                        metadata=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError, match="unsupported CV embedding index format"):
        load_embedding_index(old, model_sha256="a" * 64)


def test_empty_noise_padding_has_zero_background():
    assert _partial_background_noise_rms(np.zeros(24000), 12000) == 0.0
