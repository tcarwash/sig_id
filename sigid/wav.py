"""Blockwise PCM WAV replay with sample-indexed, normalized mono blocks."""

from __future__ import annotations

from dataclasses import dataclass
import wave
from collections.abc import Iterator
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AudioBlock:
    first_sample: int
    sample_rate: int
    samples: tuple[float, ...]

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def end_sample(self) -> int:
        return self.first_sample + self.sample_count


def _decode_pcm(raw: bytes, sample_width: int) -> list[float]:
    if sample_width == 1:
        return [(value - 128) / 128.0 for value in raw]
    if sample_width == 2:
        values = []
        for i in range(0, len(raw), 2):
            value = int.from_bytes(raw[i : i + 2], "little", signed=True)
            values.append(value / 32768.0)
        return values
    if sample_width == 3:
        values = []
        for i in range(0, len(raw), 3):
            value = int.from_bytes(raw[i : i + 3], "little", signed=False)
            if value & 0x800000:
                value -= 1 << 24
            values.append(value / 8388608.0)
        return values
    if sample_width == 4:
        values = []
        for i in range(0, len(raw), 4):
            value = int.from_bytes(raw[i : i + 4], "little", signed=True)
            values.append(value / 2147483648.0)
        return values
    raise ValueError(f"unsupported PCM sample width: {sample_width} bytes")


class WavSource:
    """Read uncompressed PCM WAV as normalized mono float blocks.

    Multi-channel files are averaged to mono. The original channel count and
    sample format remain available in the source metadata for provenance.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def metadata(self) -> dict[str, int | str]:
        with wave.open(str(self.path), "rb") as wav:
            if wav.getcomptype() != "NONE":
                raise ValueError(f"compressed WAV is unsupported: {wav.getcomptype()}")
            return {
                "path": str(self.path),
                "sample_rate": wav.getframerate(),
                "channels": wav.getnchannels(),
                "sample_width_bytes": wav.getsampwidth(),
                "sample_count": wav.getnframes(),
            }

    def blocks(self, block_frames: int = 4096) -> Iterator[AudioBlock]:
        if block_frames <= 0:
            raise ValueError("block_frames must be positive")
        with wave.open(str(self.path), "rb") as wav:
            if wav.getcomptype() != "NONE":
                raise ValueError(f"compressed WAV is unsupported: {wav.getcomptype()}")
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            first_sample = 0
            while raw := wav.readframes(block_frames):
                interleaved = _decode_pcm(raw, sample_width)
                if len(interleaved) % channels:
                    raise ValueError("WAV data ends in a partial channel frame")
                mono = tuple(
                    sum(interleaved[offset : offset + channels]) / channels
                    for offset in range(0, len(interleaved), channels)
                )
                yield AudioBlock(first_sample, sample_rate, mono)
                first_sample += len(mono)
