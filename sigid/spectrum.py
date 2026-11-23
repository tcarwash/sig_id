"""Bounded-size, sample-indexed spectrogram calculation for audio WAV files."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

from .wav import WavSource


def waterfall_samples(
    samples: Any,
    sample_rate: int,
    *,
    fft_size: int = 1024,
    hop_size: int = 256,
    max_rows: int = 512,
    sample_offset: int = 0,
    total_sample_count: int | None = None,
) -> dict[str, Any]:
    """Calculate a bounded waterfall from normalized mono samples in memory."""
    if fft_size < 32 or fft_size & (fft_size - 1):
        raise ValueError("fft_size must be a power of two and at least 32")
    if not (0 < hop_size <= fft_size):
        raise ValueError("hop_size must be between 1 and fft_size")
    if max_rows <= 0 or sample_rate <= 0 or sample_offset < 0:
        raise ValueError("max_rows and sample_rate must be positive and sample_offset non-negative")
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("samples must be a one-dimensional mono array")
    if values.size < fft_size:
        raise ValueError(f"recording needs at least {fft_size} samples for this FFT")
    frames = 1 + (values.size - fft_size) // hop_size
    row_stride = max(1, ceil(frames / max_rows))
    window = np.hanning(fft_size)
    rows = []
    starts = []
    for frame_index in range(0, frames, row_stride):
        offset = frame_index * hop_size
        spectrum = np.fft.rfft(values[offset : offset + fft_size] * window)
        magnitude = np.abs(spectrum) / max(window.sum(), 1.0)
        rows.append((20.0 * np.log10(np.maximum(magnitude, 1e-10))).astype(np.float32).tolist())
        starts.append(sample_offset + offset)
    capture_samples = total_sample_count if total_sample_count is not None else sample_offset + values.size
    if capture_samples < sample_offset + values.size:
        raise ValueError("total_sample_count cannot end before the supplied sample window")
    return {
        "sample_rate": sample_rate,
        "sample_count": capture_samples,
        "fft_size": fft_size,
        "hop_size": hop_size,
        "sample_starts": starts,
        "frequency_hz": (np.fft.rfftfreq(fft_size, 1 / sample_rate)).astype(np.float32).tolist(),
        "db": rows,
    }


def waterfall(
    path: str | Path,
    *,
    fft_size: int = 1024,
    hop_size: int = 256,
    max_rows: int = 512,
    sample_start: int = 0,
    window_sample_count: int | None = None,
) -> dict[str, Any]:
    """Calculate an amplitude spectrum while keeping a capped number of rows.

    Rows are decimated across time when the file is long. `sample_starts` preserves
    the exact source position of every displayed row so UI selections map back to
    the original recording rather than a resized image coordinate.
    """
    if fft_size < 32 or fft_size & (fft_size - 1):
        raise ValueError("fft_size must be a power of two and at least 32")
    if not (0 < hop_size <= fft_size):
        raise ValueError("hop_size must be between 1 and fft_size")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")

    source = WavSource(path)
    metadata = source.metadata()
    total_sample_count = int(metadata["sample_count"])
    sample_rate = int(metadata["sample_rate"])
    if total_sample_count < fft_size:
        raise ValueError(f"recording needs at least {fft_size} samples for this FFT")
    if sample_start < 0 or sample_start >= total_sample_count:
        raise ValueError("sample_start must be within the recording")
    requested_end = total_sample_count if window_sample_count is None else sample_start + window_sample_count
    if window_sample_count is not None and window_sample_count <= 0:
        raise ValueError("window_sample_count must be positive")
    if requested_end > total_sample_count:
        raise ValueError("requested waterfall window extends beyond the recording")

    total_frames = 1 + (total_sample_count - fft_size) // hop_size
    first_frame = ceil(sample_start / hop_size)
    end_frame = min(total_frames, ceil(requested_end / hop_size))
    window_frames = max(0, end_frame - first_frame)
    row_stride = max(1, ceil(window_frames / max_rows))
    window = np.hanning(fft_size)
    rows: list[list[float]] = []
    starts: list[int] = []
    # Reading every PCM sample is O(recording duration), even though only
    # max_rows FFT frames are displayed. Seek directly to the sampled frames
    # instead: cost is bounded by max_rows * fft_size regardless of WAV length.
    # This also avoids decoding all preceding samples for focused windows.
    import wave

    with wave.open(str(path), "rb") as audio:
        if audio.getcomptype() != "NONE":
            raise ValueError("compressed WAV is unsupported")
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        for frame_index in range(first_frame, end_frame, row_stride):
            absolute_start = frame_index * hop_size
            audio.setpos(absolute_start)
            raw = audio.readframes(fft_size)
            if len(raw) != fft_size * channels * sample_width:
                raise ValueError("recording contains incomplete PCM frame")
            if sample_width == 1:
                samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
            elif sample_width == 2:
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
            elif sample_width == 3:
                triples = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
                signed = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
                signed = (signed ^ 0x800000) - 0x800000
                samples = signed.astype(np.float64) / 8388608.0
            elif sample_width == 4:
                samples = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
            else:
                raise ValueError(f"unsupported PCM sample width: {sample_width} bytes")
            mono = samples.reshape(fft_size, channels).mean(axis=1)
            spectrum = np.fft.rfft(mono * window)
            magnitude = np.abs(spectrum) / max(window.sum(), 1.0)
            db = 20.0 * np.log10(np.maximum(magnitude, 1e-10))
            rows.append(db.astype(np.float32).tolist())
            starts.append(absolute_start)

    return {
        "sample_rate": sample_rate,
        "sample_count": total_sample_count,
        "fft_size": fft_size,
        "hop_size": hop_size,
        "sample_starts": starts,
        "frequency_hz": (np.fft.rfftfreq(fft_size, 1 / sample_rate)).astype(np.float32).tolist(),
        "db": rows,
    }
