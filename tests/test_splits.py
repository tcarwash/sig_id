from __future__ import annotations

import json

import pytest

from sigid.corpus import build_corpus
from sigid.dataset import CaptureStore
from sigid.splits import create_split_manifest


def test_split_manifest_keeps_clean_sources_and_awgn_siblings_together(tmp_path):
    output = tmp_path / "corpus"
    data = tmp_path / "data"
    build_corpus(output, data, examples_per_mode=3, snr_levels_db=[10], seed=202)
    store = CaptureStore(data / "dataset.sqlite3")
    split = create_split_manifest(output / "corpus.json", store, seed=7)
    split_again = create_split_manifest(output / "corpus.json", store, seed=7)
    assert split == split_again
    assert split["counts"]["train"]["source_groups"] > 0
    assert split["counts"]["validation"]["source_groups"] > 0
    assert split["counts"]["test"]["source_groups"] > 0
    assert all(
        split["counts"][name]["by_mode"][mode] == 1
        for name in ("train", "validation", "test")
        for mode in ("2fsk", "4fsk", "8fsk", "16fsk", "bpsk", "qpsk")
    )

    assignments = {record["lineage_group"]: record["split"] for record in split["records"]}
    assert len(assignments) == 18
    assert all(assignments[record["lineage_group"]] == record["split"] for record in split["records"])
    for record in split["records"]:
        if record["kind"] == "awgn":
            root = next(item for item in split["records"] if item["capture_id"] == record["lineage_group"])
            assert root["split"] == record["split"]


def test_split_manifest_rejects_too_few_groups_and_tampered_lineage(tmp_path):
    output = tmp_path / "small"
    data = tmp_path / "data"
    build_corpus(output, data, examples_per_mode=1, snr_levels_db=[10], seed=3)
    store = CaptureStore(data / "dataset.sqlite3")
    with pytest.raises(ValueError, match="at least three independent"):
        create_split_manifest(output / "corpus.json", store)

    corpus_path = output / "corpus.json"
    corpus = json.loads(corpus_path.read_text())
    child = next(record for record in corpus["records"] if record["kind"] == "awgn")
    child["lineage_group"] = "invented-parent"
    corpus_path.write_text(json.dumps(corpus))
    with pytest.raises(ValueError, match="not a direct child"):
        create_split_manifest(corpus_path, store)
