from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigid.dataset import CaptureStore
from sigid.synthetic import generate_2fsk_wav, generate_mfsk_wav
from sigid.synthetic_dataset import import_synthetic_wav, main as import_synthetic_main


@pytest.mark.parametrize(
    "generator,kwargs,label_family,label_method,reference_key,reference_value",
    [
        (generate_2fsk_wav, {"bits": "001101"}, "FSK", "binary frequency shift keying", "reference_bits", "001101"),
        (generate_mfsk_wav, {"symbols": [0, 3, 2, 1], "tone_count": 4}, "MFSK", "4-ary frequency shift keying", "reference_symbols", [0, 3, 2, 1]),
    ],
)
def test_import_preserves_synthetic_audio_manifest_and_ground_truth(
    tmp_path, generator, kwargs, label_family, label_method, reference_key, reference_value
):
    waveform = tmp_path / "fixture.wav"
    generated = generator(waveform, **kwargs)
    store = CaptureStore(tmp_path / "dataset.sqlite3")
    imported = import_synthetic_wav(
        waveform,
        waveform.with_suffix(".wav.json"),
        store=store,
        capture_dir=tmp_path / "captures",
    )

    capture = imported["capture"]
    region = imported["region"]
    copied_manifest = json.loads(Path(imported["manifest_path"]).read_text())
    assert capture["source_kind"] == "synthetic"
    assert capture["sample_count"] == generated["sample_count"]
    assert capture["source_metadata"]["generator"] == generated["generator"]
    assert capture["source_metadata"]["manifest_sha256"]
    assert Path(capture["audio_path"]).read_bytes() == waveform.read_bytes()
    assert Path(imported["manifest_path"]).exists()
    assert region["sample_start"] == 0
    assert region["sample_count"] == capture["sample_count"]
    assert region["family"] == label_family
    assert region["method"] == label_method
    assert region["protocol"] is None
    assert region["variant"] is None
    assert region["source"] == "synthetic"
    assert region["review_state"] == "accepted"
    assert region["evidence"]["manifest_sha256"] == capture["source_metadata"]["manifest_sha256"]
    assert copied_manifest[reference_key] == reference_value


def test_import_rejects_protocol_invention_and_does_not_store_files(tmp_path):
    waveform = tmp_path / "generic-mfsk.wav"
    generate_mfsk_wav(waveform, [0, 1, 2, 3], tone_count=4)
    sidecar = waveform.with_suffix(".wav.json")
    manifest = json.loads(sidecar.read_text())
    manifest["labels"]["protocol"] = "THOR16"
    sidecar.write_text(json.dumps(manifest))
    capture_dir = tmp_path / "captures"
    store = CaptureStore(tmp_path / "dataset.sqlite3")

    with pytest.raises(ValueError, match="cannot claim a protocol"):
        import_synthetic_wav(waveform, sidecar, store=store, capture_dir=capture_dir)
    assert store.list_captures() == []
    assert not list(capture_dir.glob("*"))


def test_import_rejects_waveform_that_does_not_match_its_manifest(tmp_path):
    waveform = tmp_path / "tampered.wav"
    generate_2fsk_wav(waveform, "101010")
    audio = bytearray(waveform.read_bytes())
    audio[-2] ^= 0x01
    waveform.write_bytes(audio)
    store = CaptureStore(tmp_path / "dataset.sqlite3")

    with pytest.raises(ValueError, match="do not match"):
        import_synthetic_wav(
            waveform,
            waveform.with_suffix(".wav.json"),
            store=store,
            capture_dir=tmp_path / "captures",
        )
    assert store.list_captures() == []


def test_import_cli_registers_generated_fixture(tmp_path, monkeypatch, capsys):
    waveform = tmp_path / "cli.wav"
    generate_mfsk_wav(waveform, [0, 2, 1, 3], tone_count=4)
    data_dir = tmp_path / "app-data"
    monkeypatch.setattr(
        "sys.argv",
        ["sig-id-import-synthetic", str(waveform), "--data-dir", str(data_dir)],
    )
    import_synthetic_main()
    result = json.loads(capsys.readouterr().out)
    assert result["capture"]["source_kind"] == "synthetic"
    assert result["region"]["family"] == "MFSK"
    assert result["region"]["protocol"] is None
    assert len(CaptureStore(data_dir / "dataset.sqlite3").list_captures()) == 1
