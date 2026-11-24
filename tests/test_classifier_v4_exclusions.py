"""Safety contracts for classifier source SHA exclusions in v4 training."""
from __future__ import annotations

import json

import pytest

from sigid import hierarchical_training
from sigid.dataset import CaptureStore


def test_real_reviewed_source_exclusion_is_applied_before_region_scanning(tmp_path, monkeypatch):
    digest = "a" * 64

    class FakeStore:
        def __init__(self, path):
            self.path = path

        def accepted_mode_definitions(self):
            return []

        def list_captures(self):
            return [{
                "source_kind": "live",
                "source_metadata": {"auto_capture_session_id": "heldout-session"},
                "sha256": digest,
            }]

        def regions(self, capture_id):
            raise AssertionError("excluded heldout source must not be scanned")

    monkeypatch.setattr(hierarchical_training, "CaptureStore", FakeStore)
    rows, summary = hierarchical_training._reviewed_real_rows(
        [tmp_path / "recorded"],
        known_hashes=set(),
        excluded_capture_hashes={digest},
    )
    assert rows == []
    assert summary["skipped"]["excluded"] == 1
    assert summary["capture_count"] == 0


@pytest.mark.parametrize("manifest", [
    {"schema_version": 999, "excluded_capture_sha256": ["a" * 64]},
    {"schema_version": 1, "excluded_capture_sha256": [], "frozen_holdout_capture_sha256": []},
])
def test_classifier_train_rejects_invalid_holdout_manifest(tmp_path, monkeypatch, manifest):
    corpus_dir = tmp_path / "corpus"
    store = CaptureStore(corpus_dir / "dataset.sqlite3")
    split = tmp_path / "split.json"
    split.write_text("{}")
    excludes = tmp_path / "exclusions.json"
    excludes.write_text(json.dumps(manifest))
    monkeypatch.setattr(hierarchical_training, "_validated_rows", lambda *args, **kwargs: [])
    with pytest.raises(ValueError, match="frozen|invalid"):
        hierarchical_training.train_hierarchical(
            split,
            corpus_dir,
            tmp_path / "out" / "classifier.json",
            exclude_reviewed_sha_file=excludes,
            epochs=1,
        )
