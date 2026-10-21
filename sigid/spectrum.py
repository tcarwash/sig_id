"""Bounded-size, sample-indexed spectrogram calculation for audio WAV files."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

from .wav import WavSource


def waterfall(
    path: str | Path,
    *,
    fft_size: int = 1024,
    hop_size: int = 256,
    max_rows: int = 512,
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
    sample_count = int(metadata["sample_count"])
    sample_rate = int(metadata["sample_rate"])
    if sample_count < fft_size:
        raise ValueError(f"recording needs at least {fft_size} samples for this FFT")

    total_frames = 1 + (sample_count - fft_size) // hop_size
    row_stride = max(1, ceil(total_frames / max_rows))
    window = np.hanning(fft_size)
    rows: list[list[float]] = []
    starts: list[int] = []
    sample_buffer = np.empty(0, dtype=np.float64)
    frame_index = 0

    for block in source.blocks():
        joined = np.concatenate((sample_buffer, np.asarray(block.samples, dtype=np.float64)))
        offset = 0
        while offset + fft_size <= joined.size:
            if frame_index % row_stride == 0:
                spectrum = np.fft.rfft(joined[offset : offset + fft_size] * window)
                magnitude = np.abs(spectrum) / max(window.sum(), 1.0)
                db = 20.0 * np.log10(np.maximum(magnitude, 1e-10))
                rows.append(db.astype(np.float32).tolist())
                starts.append(block.first_sample + offset - sample_buffer.size)
            offset += hop_size
            frame_index += 1
        sample_buffer = joined[offset:]

    return {
        "sample_rate": sample_rate,
        "sample_count": sample_count,
        "fft_size": fft_size,
        "hop_size": hop_size,
        "sample_starts": starts,
        "frequency_hz": (np.fft.rfftfreq(fft_size, 1 / sample_rate)).astype(np.float32).tolist(),
        "db": rows,
    }
