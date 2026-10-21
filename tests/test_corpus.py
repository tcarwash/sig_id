from __future__ import annotations

import json

import pytest

from sigid.corpus import MODES, build_corpus
from sigid.dataset import CaptureStore


def test_corpus_builder_creates_six_methods_with_parent_grouped_awgn_and_resumes(tmp_path):
    output = tmp_path / "corpus"
    data = tmp_path / "data"
    first = build_corpus(output, data, examples_per_mode=1, snr_levels_db=[10], seed=22)
    assert first["record_count"] == len(MODES) * 2
    clean = [record for record in first["records"] if record["kind"] == "clean"]
    noisy = [record for record in first["records"] if record["kind"] == "awgn"]
    assert len(clean) == len(noisy) == len(MODES)
    assert {record["mode"] for record in clean} == set(MODES)
    assert all(record["lineage_group"] == record["capture_id"] for record in clean)
    assert {record["lineage_group"] for record in noisy} == {record["capture_id"] for record in clean}
    assert all(record["parent_capture_id"] == record["lineage_group"] for record in noisy)

    store = CaptureStore(data / "dataset.sqlite3")
    captures_before = {record["id"] for record in store.list_captures()}
    resumed = build_corpus(output, data, examples_per_mode=1, snr_levels_db=[10], seed=22)
    captures_after = {record["id"] for record in store.list_captures()}
    assert resumed["record_count"] == len(MODES) * 2
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
