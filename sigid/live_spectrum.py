"""Incremental, globally sample-aligned spectrum packets for the live display."""
from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np


def spectrum_packet(
    snapshot: dict[str, Any], *, after_sample: int = -1, session_id: str | None = None,
    fft_size: int = 4096, hop_size: int = 512, max_rows: int = 512,
) -> bytes:
    if fft_size < 32 or fft_size > 8192 or fft_size & (fft_size - 1):
        raise ValueError("fft_size must be a power of two between 32 and 8192")
    if not 1 <= hop_size <= fft_size or not 1 <= max_rows <= 1024 or after_sample < -1:
        raise ValueError("invalid incremental spectrum bounds")
    rate = int(snapshot["sample_rate"])
    channels = int(snapshot["channels"])
    offset = int(snapshot["sample_offset"])
    total = int(snapshot["sample_count"])
    source = np.frombuffer(snapshot["pcm"], dtype="<i2").reshape(-1, channels)
    if session_id != snapshot["session_id"]:
        after_sample = -1
    first = max(((offset + hop_size - 1) // hop_size) * hop_size,
                ((after_sample // hop_size) + 1) * hop_size)
    last = ((offset + len(source) - fft_size) // hop_size) * hop_size
    first = max(first, last - (max_rows - 1) * hop_size)
    starts = list(range(first, last + 1, hop_size)) if first <= last else []
    values = (source[first-offset:last-offset+fft_size].astype(np.float64).mean(axis=1) / 32768.0
              if starts else np.empty(0, dtype=np.float64))
    frequencies = np.fft.rfftfreq(fft_size, 1 / rate)
    frequencies = frequencies[frequencies <= min(3000, rate / 2)]
    rows = np.empty((len(starts), len(frequencies)), dtype="<f4")
    window = np.hanning(fft_size)
    scale = max(float(window.sum()), 1.0)
    for index, position in enumerate(starts):
        local = position - first
        spectrum = np.abs(np.fft.rfft(values[local:local + fft_size] * window))[:len(frequencies)] / scale
        rows[index] = 20 * np.log10(np.maximum(spectrum, 1e-10))
    metadata = {
        "schema_version": 1, "session_id": snapshot["session_id"], "sample_rate": rate,
        "sample_count": total, "fft_size": fft_size, "hop_size": hop_size,
        "frequency_hz": frequencies.tolist(), "sample_starts": starts,
        "shape": [len(starts), len(frequencies)],
        "gap_before": bool(starts and after_sample >= 0 and starts[0] > after_sample + hop_size),
    }
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % 4)
    return struct.pack("<I", len(header)) + header + rows.tobytes()
