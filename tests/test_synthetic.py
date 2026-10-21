from __future__ import annotations

import json
import wave

import pytest

from sigid.synthetic import generate_2fsk_wav
from sigid.wav import WavSource


def test_generator_writes_repeatable_labeled_waveform_and_manifest(tmp_path):
    path = tmp_path / "source.wav"
    manifest = generate_2fsk_wav(path, "001101", sample_rate=8000, symbol_rate=100)
    loaded = json.loads(path.with_suffix(".wav.json").read_text())
    assert manifest == loaded
    assert manifest["labels"] == {
        "modulation_family": "FSK",
        "modulation_method": "binary frequency shift keying",
        "protocol": None,
        "variant": None,
    }
    assert manifest["reference_bits"] == "001101"
    assert manifest["sample_count"] == 480
    blocks = list(WavSource(path).blocks(block_frames=64))
    assert sum(block.sample_count for block in blocks) == 480
    assert all(-1.0 <= sample <= 1.0 for block in blocks for sample in block.samples)
    assert any(sample != 0.0 for block in blocks for sample in block.samples)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 8000, "symbol_rate": 300},
        {"sample_rate": 8000, "symbol_rate": 0},
        {"sample_rate": 8000, "tone_low_hz": 4000, "tone_high_hz": 5000},
        {"amplitude": 0},
    ],
)
def test_generator_rejects_invalid_parameters(tmp_path, kwargs):
    with pytest.raises(ValueError):
        generate_2fsk_wav(tmp_path / "bad.wav", "101", **kwargs)


def test_generator_rejects_invalid_bits(tmp_path):
    with pytest.raises(ValueError, match="bits"):
        generate_2fsk_wav(tmp_path / "bad.wav", "10x1")


def test_generator_emits_the_requested_tone_for_each_symbol(tmp_path):
    path = tmp_path / "tones.wav"
    generate_2fsk_wav(
        path,
        "01",
        sample_rate=8000,
        symbol_rate=100,
        tone_low_hz=1000,
        tone_high_hz=1500,
    )
    samples = [sample for block in WavSource(path).blocks() for sample in block.samples]
    symbol = len(samples) // 2

    def crossings(values):
        signs = [1 if value > 0 else -1 for value in values if value != 0]
        return sum(left != right for left, right in zip(signs, signs[1:]))

    assert crossings(samples[:symbol]) == 19
    assert crossings(samples[symbol:]) in (29, 30)
