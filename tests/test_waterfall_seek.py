"""Regression checks for bounded random-access WAV waterfall rendering."""
from __future__ import annotations

import struct
import wave

import numpy as np
import pytest

from sigid.spectrum import waterfall, waterfall_samples
from sigid.wav import WavSource


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("channels", [1, 2])
def test_seek_waterfall_matches_reference_pcm(width, channels, tmp_path):
    rate = 8000
    time = np.arange(6500) / rate
    left = 0.64 * np.sin(2 * np.pi * 1040 * time)
    right = 0.43 * np.sin(2 * np.pi * 1670 * time)
    if channels == 1:
        interleaved = left
    else:
        interleaved = np.column_stack((left, right)).reshape(-1)
    if width == 1:
        raw = np.clip(np.round(interleaved * 127 + 128), 0, 255).astype("u1").tobytes()
    else:
        peak = (1 << (8 * width - 1)) - 1
        values = np.clip(np.round(interleaved * peak), -peak - 1, peak).astype(np.int64)
        if width == 2:
            raw = values.astype("<i2").tobytes()
        elif width == 4:
            raw = values.astype("<i4").tobytes()
        else:
            raw = b"".join(int(value).to_bytes(3, "little", signed=True) for value in values)
    path = tmp_path / f"pcm-{width}-{channels}.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(raw)

    reference_samples = [sample for block in WavSource(path).blocks() for sample in block.samples]
    kwargs = dict(fft_size=256, hop_size=64, max_rows=31)
    wanted = waterfall_samples(reference_samples, rate, **kwargs)
    actual = waterfall(path, **kwargs)
    assert actual["sample_starts"] == wanted["sample_starts"]
    assert actual["frequency_hz"] == wanted["frequency_hz"]
    np.testing.assert_allclose(actual["db"], wanted["db"], atol=0.0001, rtol=0)


def test_sparse_long_wav_never_scans_all_audio_blocks(tmp_path, monkeypatch):
    # 12-minute file consumes negligible real disk space.
    rate = 48000
    total = rate * 12 * 60
    size = total * 2
    path = tmp_path / "twelve-minutes.wav"
    header = (
        b"RIFF" + struct.pack("<I", size + 36)
        + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, 2 * rate, 2, 16)
        + b"data" + struct.pack("<I", size)
    )
    with path.open("wb") as f:
        f.write(header)
        f.seek(44 + size - 1)
        f.write(b"\0")

    def should_not_scan_entire_source(*args, **kwargs):
        raise AssertionError("waterfall attempted a sequential full WAV scan")

    monkeypatch.setattr(WavSource, "blocks", should_not_scan_entire_source)
    full = waterfall(path, fft_size=4096, hop_size=512, max_rows=80)
    assert len(full["db"]) <= 80
    assert full["sample_count"] == total
    focused = waterfall(path, fft_size=4096, hop_size=512, max_rows=60,
                        sample_start=total - rate * 30, window_sample_count=rate * 25)
    assert len(focused["db"]) <= 60
    assert focused["sample_starts"][0] >= total - rate * 30
    assert focused["sample_starts"][-1] <= total - rate * 5
