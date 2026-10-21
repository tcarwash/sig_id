from __future__ import annotations

import wave

import pytest

from sigid.wav import WavSource


def write_pcm16(path, values, *, channels=1, sample_rate=8000):
    interleaved = b"".join(int(value).to_bytes(2, "little", signed=True) for value in values)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(interleaved)


def test_wav_blocks_cover_samples_with_stable_offsets_and_normalization(tmp_path):
    path = tmp_path / "mono.wav"
    write_pcm16(path, [-32768, 0, 16384, 32767])

    source = WavSource(path)
    blocks = list(source.blocks(block_frames=3))
    assert [(b.first_sample, b.sample_count, b.end_sample) for b in blocks] == [
        (0, 3, 3), (3, 1, 4)
    ]
    assert blocks[0].samples == (-1.0, 0.0, 0.5)
    assert blocks[1].samples[0] == pytest.approx(32767 / 32768)
    assert source.metadata()["sample_count"] == 4


def test_wav_multichannel_is_averaged_to_mono(tmp_path):
    path = tmp_path / "stereo.wav"
    write_pcm16(path, [32767, -32768, 16384, 16384], channels=2, sample_rate=16000)
    block = next(WavSource(path).blocks())
    assert block.sample_rate == 16000
    assert block.samples == pytest.approx((-1 / 65536, 0.5))


@pytest.mark.parametrize("block_frames", [0, -1])
def test_wav_rejects_invalid_block_size(tmp_path, block_frames):
    path = tmp_path / "empty.wav"
    write_pcm16(path, [])
    with pytest.raises(ValueError, match="block_frames"):
        list(WavSource(path).blocks(block_frames=block_frames))
