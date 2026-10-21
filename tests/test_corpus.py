from __future__ import annotations

import json

import pytest

from sigid.corpus import MODES, build_corpus
from sigid.dataset import CaptureStore
from sigid.synthetic_dataset import import_synthetic_wav


def test_corpus_builder_creates_six_methods_with_parent_grouped_awgn_and_resumes(tmp_path):
    output = tmp_path / "corpus"
    data = tmp_path / "data"
    first = build_corpus(output, data, examples_per_mode=1, snr_levels_db=[10], seed=22)
    assert first["record_count"] == len(MODES) * 2 + 1
    clean = [record for record in first["records"] if record["kind"] == "clean"]
    noisy = [record for record in first["records"] if record["kind"] == "awgn"]
    assert len(clean) == len(noisy) == len(MODES)
    assert {record["mode"] for record in clean} == set(MODES)
    assert all(record["lineage_group"] == record["capture_id"] for record in clean)
    assert {record["lineage_group"] for record in noisy} == {record["capture_id"] for record in clean}
    assert all(record["parent_capture_id"] == record["lineage_group"] for record in noisy)
    negatives = [record for record in first["records"] if record["kind"] == "noise"]
    assert len(negatives) == 1
    assert negatives[0]["mode"] == "noise"
    assert negatives[0]["parent_capture_id"] is None

    store = CaptureStore(data / "dataset.sqlite3")
    captures_before = {record["id"] for record in store.list_captures()}
    resumed = build_corpus(output, data, examples_per_mode=1, snr_levels_db=[10], seed=22)
    captures_after = {record["id"] for record in store.list_captures()}
    assert resumed["record_count"] == len(MODES) * 2 + 1
    assert captures_before == captures_after
    assert json.loads((output / "corpus.json").read_text())["records"] == resumed["records"]


def test_corpus_configuration_and_resource_limits_are_explicit(tmp_path):
    output = tmp_path / "corpus"
    data = tmp_path / "data"
    build_corpus(output, data, examples_per_mode=1, snr_levels_db=[10], seed=22)
    with pytest.raises(ValueError, match="different configuration"):
        build_corpus(output, data, examples_per_mode=2, snr_levels_db=[10], seed=22)
    with pytest.raises(ValueError, match="between 1 and"):
        build_corpus(tmp_path / "bad", data, examples_per_mode=0)
    with pytest.raises(ValueError, match="distinct SNR"):
        build_corpus(tmp_path / "bad-snr", data, snr_levels_db=[10, 10])


def test_noise_negative_has_no_taxonomy_label_and_waveform_is_verified(tmp_path):
    output = tmp_path / "corpus"
    data = tmp_path / "data"
    corpus = build_corpus(output, data, examples_per_mode=1, snr_levels_db=[], seed=44)
    record = next(record for record in corpus["records"] if record["kind"] == "noise")
    source = output / "noise" / "0000.wav"
    manifest = source.with_suffix(source.suffix + ".json")
    imported = next(item for item in CaptureStore(data / "dataset.sqlite3").regions(record["capture_id"]))
    assert imported["family"] is None
    assert imported["method"] is None
    assert imported["protocol"] is None
    assert imported["review_state"] == "accepted"

    corrupted = tmp_path / "corrupted-noise.wav"
    payload = bytearray(source.read_bytes())
    payload[-1] ^= 1
    corrupted.write_bytes(payload)
    with pytest.raises(ValueError, match="do not match the synthetic generator"):
        import_synthetic_wav(
            corrupted,
            manifest,
            store=CaptureStore(data / "dataset.sqlite3"),
            capture_dir=data / "captures",
        )
