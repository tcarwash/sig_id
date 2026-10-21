from __future__ import annotations

import json

import pytest

from sigid.ood_corpus import build_ood_corpus
from sigid.synthetic_dataset import _validate_manifest


def test_ood_corpus_is_deterministic_resumable_and_varies_signal_parameters(tmp_path):
    corpus_dir = tmp_path / "unknown-family"
    first = build_ood_corpus(corpus_dir, examples=5, seed=812)
    second = build_ood_corpus(corpus_dir, examples=5, seed=812)
    assert first == second
    assert first["record_count"] == 5
    assert all(record["independent_source"] and record["parent_index"] is None for record in first["records"])
    assert len({record["waveform_sha256"] for record in first["records"]}) == 5
    assert len({record["symbol_rate"] for record in first["records"]}) > 1
    assert len({record["carrier_hz"] for record in first["records"]}) > 1

    for record in first["records"]:
        wav_path = corpus_dir / record["waveform"]
        manifest_path = corpus_dir / record["manifest"]
        manifest = json.loads(manifest_path.read_text())
        _validate_manifest(manifest, wav_path)
        assert manifest["labels"]["modulation_family"] == "ASK"
        assert manifest["labels"]["protocol"] is None


def test_ood_corpus_rejects_recipe_changes_and_invalid_counts(tmp_path):
    corpus_dir = tmp_path / "ood"
    build_ood_corpus(corpus_dir, examples=3, seed=11)
    with pytest.raises(ValueError, match="different configuration"):
        build_ood_corpus(corpus_dir, examples=4, seed=11)
    with pytest.raises(ValueError, match="examples"):
        build_ood_corpus(tmp_path / "bad", examples=0)
