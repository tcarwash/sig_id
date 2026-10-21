from __future__ import annotations

import json
import wave

import pytest

from sigid.synthetic import bpsk_main, generate_2fsk_wav, generate_bpsk_wav, generate_mfsk_wav, generate_ook_wav, generate_qpsk_wav, mfsk_main, ook_main, qpsk_main
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


def test_mfsk_generator_saves_exact_symbols_bits_mapping_and_method_label(tmp_path):
    import numpy as np

    path = tmp_path / "mfsk4.wav"
    manifest = generate_mfsk_wav(
        path,
        [0, 1, 2, 3],
        tone_count=4,
        sample_rate=12000,
        symbol_rate=100,
        base_tone_hz=1000,
        tone_spacing_hz=200,
    )
    saved = json.loads(path.with_suffix(".wav.json").read_text())
    assert saved == manifest
    assert manifest["labels"] == {
        "modulation_family": "MFSK",
        "modulation_method": "4-ary frequency shift keying",
        "protocol": None,
        "variant": None,
    }
    assert manifest["reference_symbols"] == [0, 1, 2, 3]
    assert manifest["reference_bits"] == "00011011"
    assert manifest["parameters"]["tone_frequencies_hz"] == [1000, 1200, 1400, 1600]
    assert manifest["parameters"]["protocol_framing"] is None

    with wave.open(str(path), "rb") as wav:
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float)
    samples_per_symbol = 120
    measured = []
    for index in range(4):
        block = samples[index * samples_per_symbol : (index + 1) * samples_per_symbol]
        spectrum = np.abs(np.fft.rfft(block))
        measured.append(int(np.argmax(spectrum)) * 100)
    assert measured == [1000, 1200, 1400, 1600]

    second = tmp_path / "same.wav"
    generate_mfsk_wav(second, [0, 1, 2, 3], sample_rate=12000, symbol_rate=100)
    assert path.read_bytes() == second.read_bytes()


def test_mfsk_non_power_of_two_uses_symbols_without_fabricating_bits(tmp_path):
    manifest = generate_mfsk_wav(tmp_path / "mfsk5.wav", [0, 4, 1], tone_count=5)
    assert manifest["labels"]["modulation_family"] == "MFSK"
    assert manifest["reference_symbols"] == [0, 4, 1]
    assert manifest["reference_bits"] is None
    assert manifest["parameters"]["bits_per_symbol"] is None


def test_mfsk_cli_generates_a_labeled_fixture(tmp_path, monkeypatch, capsys):
    path = tmp_path / "cli-mfsk.wav"
    monkeypatch.setattr("sys.argv", ["sig-id-generate-mfsk", str(path), "0,3,1"])
    mfsk_main()
    output = json.loads(capsys.readouterr().out)
    assert path.exists()
    assert output["reference_symbols"] == [0, 3, 1]
    assert output["labels"]["modulation_family"] == "MFSK"


def test_bpsk_generator_and_cli_preserve_exact_bits_and_unknown_protocol(tmp_path, monkeypatch, capsys):
    path = tmp_path / "bpsk.wav"
    manifest = generate_bpsk_wav(
        path,
        "00101101",
        sample_rate=12000,
        symbol_rate=100,
        carrier_hz=1600,
    )
    assert json.loads(path.with_suffix(".wav.json").read_text()) == manifest
    assert manifest["labels"]["modulation_method"] == "binary phase-shift keying"
    assert manifest["labels"]["protocol"] is None
    second = tmp_path / "same.wav"
    generate_bpsk_wav(second, "00101101", sample_rate=12000, symbol_rate=100, carrier_hz=1600)
    assert path.read_bytes() == second.read_bytes()

    cli_path = tmp_path / "cli-bpsk.wav"
    monkeypatch.setattr("sys.argv", ["sig-id-generate-bpsk", str(cli_path), "101001"])
    bpsk_main()
    cli_manifest = json.loads(capsys.readouterr().out)
    assert cli_manifest["reference_bits"] == "101001"
    assert cli_manifest["labels"]["protocol"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 8000, "symbol_rate": 300},
        {"sample_rate": True},
        {"symbol_rate": 0},
        {"carrier_hz": 6000},
        {"initial_phase_radians": float("nan")},
        {"amplitude": 0},
    ],
)
def test_bpsk_generator_rejects_invalid_parameters(tmp_path, kwargs):
    with pytest.raises(ValueError):
        generate_bpsk_wav(tmp_path / "invalid.wav", "101", **kwargs)


def test_qpsk_generator_and_cli_preserve_gray_mapped_bits(tmp_path, monkeypatch, capsys):
    path = tmp_path / "qpsk.wav"
    manifest = generate_qpsk_wav(path, "00110110", initial_phase_radians=0.5)
    assert json.loads(path.with_suffix(".wav.json").read_text()) == manifest
    assert manifest["reference_symbols"] == ["00", "11", "01", "10"]
    assert manifest["labels"]["modulation_family"] == "PSK"
    assert manifest["labels"]["protocol"] is None

    cli_path = tmp_path / "cli-qpsk.wav"
    monkeypatch.setattr("sys.argv", ["sig-id-generate-qpsk", str(cli_path), "0011"])
    qpsk_main()
    cli_manifest = json.loads(capsys.readouterr().out)
    assert cli_manifest["reference_bits"] == "0011"


@pytest.mark.parametrize("bits", ["", "0", "001"])
def test_qpsk_generator_rejects_empty_or_odd_bit_sequences(tmp_path, bits):
    with pytest.raises(ValueError, match="even-length"):
        generate_qpsk_wav(tmp_path / "bad-qpsk.wav", bits)


def test_ook_generator_and_cli_preserve_exact_bits_and_ask_label(tmp_path, monkeypatch, capsys):
    path = tmp_path / "ook.wav"
    manifest = generate_ook_wav(
        path,
        "010101",
        sample_rate=8000,
        symbol_rate=100,
        carrier_hz=1200,
        initial_phase_radians=0.5,
    )
    assert json.loads(path.with_suffix(".wav.json").read_text()) == manifest
    assert manifest["labels"] == {
        "modulation_family": "ASK",
        "modulation_method": "on-off keying (OOK)",
        "protocol": None,
        "variant": None,
    }
    assert manifest["reference_bits"] == "010101"
    samples = [sample for block in WavSource(path).blocks() for sample in block.samples]
    assert not any(samples[:80])
    assert any(samples[80:160])
    second = tmp_path / "same.wav"
    generate_ook_wav(
        second,
        "010101",
        sample_rate=8000,
        symbol_rate=100,
        carrier_hz=1200,
        initial_phase_radians=0.5,
    )
    assert path.read_bytes() == second.read_bytes()

    cli_path = tmp_path / "cli-ook.wav"
    monkeypatch.setattr("sys.argv", ["sig-id-generate-ook", str(cli_path), "1010"])
    ook_main()
    cli_manifest = json.loads(capsys.readouterr().out)
    assert cli_manifest["reference_bits"] == "1010"
    assert cli_manifest["labels"]["modulation_family"] == "ASK"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 8000, "symbol_rate": 300},
        {"sample_rate": True},
        {"symbol_rate": 0},
        {"carrier_hz": 6000},
        {"initial_phase_radians": float("nan")},
        {"amplitude": 0},
    ],
)
def test_ook_generator_rejects_invalid_parameters(tmp_path, kwargs):
    with pytest.raises(ValueError):
        generate_ook_wav(tmp_path / "invalid-ook.wav", "101", **kwargs)


@pytest.mark.parametrize(
    ("symbols", "kwargs", "message"),
    [
        ([], {}, "non-empty"),
        ([0, 4], {"tone_count": 4}, "tone index"),
        ([0, 1.5], {}, "tone index"),
        ([0, 1], {"tone_count": 1}, "tone_count"),
        ([0, 1], {"sample_rate": 8000, "symbol_rate": 300}, "multiple"),
        ([0, 1], {"base_tone_hz": 3000, "tone_spacing_hz": 1000}, "Nyquist"),
        ([0, 1], {"amplitude": 0}, "amplitude"),
    ],
)
def test_mfsk_generator_rejects_invalid_inputs(tmp_path, symbols, kwargs, message):
    with pytest.raises(ValueError, match=message):
        generate_mfsk_wav(tmp_path / "invalid.wav", symbols, **kwargs)
