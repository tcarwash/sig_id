from __future__ import annotations

import pytest

from sigid.corpus import build_corpus
from sigid.seed_evaluation import evaluate_split_seeds


def test_multi_seed_runner_trains_and_reports_a_lineage_safe_run(tmp_path):
    corpus_dir = tmp_path / "corpus"
    data_dir = tmp_path / "data"
    build_corpus(corpus_dir, data_dir, examples_per_mode=3, snr_levels_db=[], seed=712)
    result = evaluate_split_seeds(
        corpus_dir / "corpus.json",
        data_dir,
        tmp_path / "runs",
        seeds=[21],
        epochs=40,
    )
    assert result["split_seeds"] == [21]
    assert result["runs"][0]["seed"] == 21
    assert result["runs"][0]["split_sha256"]
    assert result["runs"][0]["model_sha256"]
    assert result["summary"]["coverage"]["count"] == 1
    assert (tmp_path / "runs" / "split-21.json").is_file()
    assert (tmp_path / "runs" / "model-21.npz").is_file()


@pytest.mark.parametrize("seeds", [[], [5, 5], [-1], [2**32]])
def test_multi_seed_runner_requires_unique_unsigned_seeds(tmp_path, seeds):
    with pytest.raises(ValueError, match="seed"):
        evaluate_split_seeds("missing", tmp_path, tmp_path / "out", seeds=seeds)
