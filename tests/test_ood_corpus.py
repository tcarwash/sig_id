from __future__ import annotations

import json
import hashlib

import pytest

from sigid.ood_corpus import build_ood_corpus, load_ood_corpus
from sigid.synthetic_dataset import _validate_manifest


def test_ood_corpus_is_deterministic_resumable_and_varies_signal_parameters(tmp_path):
    corpus_dir = tmp_path / "unknown-family"
    first = build_ood_corpus(corpus_dir, examples=6, seed=812)
    second = build_ood_corpus(corpus_dir, examples=6, seed=812)
    assert first == second
    assert first["record_count"] == 6
    assert all(record["independent_source"] and record["parent_index"] is None for record in first["records"])
    assert len({record["waveform_sha256"] for record in first["records"]}) == 6
    assert len({record["symbol_rate"] for record in first["records"]}) > 1
    assert len({record["carrier_hz"] for record in first["records"]}) > 1
    assert {record["modulation_method"] for record in first["records"]} == {
        "on-off keying (OOK)", "4-ary amplitude shift keying", "binary pulse-position modulation"
    }

    for record in first["records"]:
        wav_path = corpus_dir / record["waveform"]
        manifest_path = corpus_dir / record["manifest"]
        manifest = json.loads(manifest_path.read_text())
        _validate_manifest(manifest, wav_path)
        assert manifest["labels"]["modulation_family"] == record["modulation_family"]
        assert manifest["labels"]["protocol"] is None
    verified = load_ood_corpus(corpus_dir / "ood-corpus.json")
    validation = {record["index"] for record in verified if record["split"] == "validation"}
    test = {record["index"] for record in verified if record["split"] == "test"}
    assert validation.isdisjoint(test)
    assert validation | test == set(range(6))


def test_legacy_ask_benchmark_recipe_remains_reproducible(tmp_path):
    corpus_dir = tmp_path / "legacy-ask"
    result = build_ood_corpus(corpus_dir, examples=30, seed=1717, generator_version="0.2.0")
    assert result["config"]["generator_version"] == "0.2.0"
    assert result["config"]["modulation_family"] == "ASK"
    assert "modulation_family" not in result["records"][0]
    verified = load_ood_corpus(corpus_dir / "ood-corpus.json")
    assert {row["modulation_family"] for row in verified} == {"ASK"}
    assert verified[0]["waveform_sha256"] == "812bb839650cfc9afa9c3f7b0611591f7b4871a18d46bb82164abf5951ecb13e"
    assert hashlib.sha256((corpus_dir / "ood-corpus.json").read_bytes()).hexdigest() == (
        "494dc4cc207c981c9c93f3948ff2ba7d210161250df509e0efc9d1ea06f0851e"
    )


def test_ood_corpus_rejects_recipe_changes_and_invalid_counts(tmp_path):
    corpus_dir = tmp_path / "ood"
    build_ood_corpus(corpus_dir, examples=3, seed=11)
    with pytest.raises(ValueError, match="different configuration"):
        build_ood_corpus(corpus_dir, examples=4, seed=11)
    with pytest.raises(ValueError, match="examples"):
        build_ood_corpus(tmp_path / "bad", examples=0)


def test_ood_corpus_loader_rejects_changed_waveform_hash(tmp_path):
    corpus_dir = tmp_path / "ood"
    corpus = build_ood_corpus(corpus_dir, examples=5, seed=11)
    waveform = corpus_dir / corpus["records"][0]["waveform"]
    data = bytearray(waveform.read_bytes())
    data[-2] ^= 1
    waveform.write_bytes(data)
    with pytest.raises(ValueError, match="hashes"):
        load_ood_corpus(corpus_dir / "ood-corpus.json")
