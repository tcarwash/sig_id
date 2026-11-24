"""A pilot or undersized real holdout cannot be promoted by CV comparisons."""
import json
from pathlib import Path

from tools.compare_cv_refreshed import release_preconditions


def _make_dataset(tmp_path: Path, groups: int, fraction: float):
    ds = tmp_path / "dataset"
    ds.mkdir()
    windows = [
        {
            "split": "test", "source_kind": "live", "box_count": 1,
            "lineage_group": f"lineage-{i:04d}",
        }
        for i in range(groups)
    ]
    (ds / "windows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in windows) + "\n"
    )
    model = tmp_path / "model.pt"
    model.write_bytes(b"pilot")
    model.with_suffix(".pt.json").write_text(json.dumps({
        "train_fraction": fraction, "pilot_partial_training": fraction < 1,
    }))
    return ds, model


def test_release_requires_five_independent_real_lineages(tmp_path):
    ds, model = _make_dataset(tmp_path, groups=3, fraction=1)
    gates = release_preconditions(ds, model)
    assert gates["candidate_trained_on_full_training_split"]
    assert not gates["at_least_five_independent_heldout_live_lineages"]


def test_release_rejects_partial_training_even_with_five_lineages(tmp_path):
    ds, model = _make_dataset(tmp_path, groups=6, fraction=0.25)
    gates = release_preconditions(ds, model)
    assert gates["at_least_five_independent_heldout_live_lineages"]
    assert not gates["candidate_trained_on_full_training_split"]


def test_release_preconditions_allow_full_training_and_five_lineages(tmp_path):
    ds, model = _make_dataset(tmp_path, groups=5, fraction=1)
    assert all(release_preconditions(ds, model).values())
