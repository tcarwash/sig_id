from __future__ import annotations

import json

import pytest

from sigid.augmentation import generate_awgn_augmentation
from sigid.dataset import CaptureStore
from sigid.synthetic import generate_bpsk_wav
from sigid.synthetic_dataset import import_synthetic_wav


def test_awgn_augmentation_is_reproducible_and_imports_with_parent_lineage(tmp_path):
    source = tmp_path / "source.wav"
    generate_bpsk_wav(source, "00110110" * 32, initial_phase_radians=0.4)
    store = CaptureStore(tmp_path / "db.sqlite3")
    clean = import_synthetic_wav(
        source,
        source.with_suffix(".wav.json"),
        store=store,
        capture_dir=tmp_path / "captures",
    )

    child_a = tmp_path / "awgn-a.wav"
    child_b = tmp_path / "awgn-b.wav"
    manifest = generate_awgn_augmentation(
        source,
        source.with_suffix(".wav.json"),
        child_a,
        snr_db=10,
        gain=0.7,
        seed=1234,
    )
    generate_awgn_augmentation(
        source,
        source.with_suffix(".wav.json"),
        child_b,
        snr_db=10,
        gain=0.7,
        seed=1234,
    )
    assert child_a.read_bytes() == child_b.read_bytes()
    assert manifest["parameters"]["measured_snr_db"] == pytest.approx(10, abs=1)

    imported = import_synthetic_wav(
        child_a,
        child_a.with_suffix(".wav.json"),
        store=store,
        capture_dir=tmp_path / "captures",
    )
    capture = imported["capture"]
    assert capture["parent_capture_id"] == clean["capture"]["id"]
    assert capture["source_kind"] == "synthetic"
    assert imported["region"]["family"] == "PSK"
    assert imported["region"]["method"] == "binary phase-shift keying"
    assert imported["region"]["protocol"] is None
    assert imported["region"]["source"] == "synthetic"
    assert imported["region"]["evidence"]["parent_capture_id"] == clean["capture"]["id"]


def test_awgn_augmentation_rejects_unknown_parent_and_tampered_recipe(tmp_path):
    source = tmp_path / "source.wav"
    generate_bpsk_wav(source, "01010110" * 16)
    child = tmp_path / "child.wav"
    generate_awgn_augmentation(
        source,
        source.with_suffix(".wav.json"),
        child,
        snr_db=15,
        seed=5,
    )
    empty_store = CaptureStore(tmp_path / "empty.sqlite3")
    with pytest.raises(ValueError, match="already be imported"):
        import_synthetic_wav(
            child,
            child.with_suffix(".wav.json"),
            store=empty_store,
            capture_dir=tmp_path / "empty-captures",
        )

    store = CaptureStore(tmp_path / "parent.sqlite3")
    import_synthetic_wav(
        source,
        source.with_suffix(".wav.json"),
        store=store,
        capture_dir=tmp_path / "parent-captures",
    )
    manifest_path = child.with_suffix(".wav.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["parameters"]["measured_snr_db"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="statistic 'measured_snr_db'"):
        import_synthetic_wav(
            child,
            manifest_path,
            store=store,
            capture_dir=tmp_path / "parent-captures",
        )
