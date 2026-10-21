from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from sigid.classifier import FEATURE_VERSION, load_model
from sigid.corpus import MODES
from sigid.ood_evaluation import evaluate_unmodeled_wavs
from sigid.synthetic import generate_2fsk_wav, generate_ook_wav


def _model(path, *, accept_known: bool):
    classes = [*MODES, "noise"]
    weights = np.zeros((260, len(classes)), dtype=np.float32)
    bias = np.full(len(classes), -10.0, dtype=np.float32)
    bias[classes.index("2fsk" if accept_known else "noise")] = 10.0
    with path.open("wb") as output:
        np.savez_compressed(
            output,
            mean=np.zeros(260, dtype=np.float32),
            scale=np.ones(260, dtype=np.float32),
            weights=weights,
            bias=bias,
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps({
        "schema_version": 1,
        "feature_version": FEATURE_VERSION,
        "feature_count": 260,
        "classes": classes,
        "abstention_threshold": 0.5,
        "family_abstention_threshold": 0.5,
        "model_sha256": digest,
    }))
    return path


def test_ood_evaluator_reports_method_and_family_false_accepts_with_provenance(tmp_path):
    model = _model(tmp_path / "model.npz", accept_known=True)
    first = tmp_path / "ook-one.wav"
    second = tmp_path / "ook-two.wav"
    generate_ook_wav(first, "010101" * 8)
    generate_ook_wav(second, "111000" * 8)

    result = evaluate_unmodeled_wavs(model, [first, second])

    assert result["examples"] == 2
    assert result["method_false_accepts"] == 2
    assert result["method_false_accept_rate"] == 1.0
    assert result["family_false_accepts"] == 2
    assert result["family_false_accept_rate"] == 1.0
    assert result["model_sha256"] == load_model(model)[0]["model_sha256"]
    assert [item["true_family"] for item in result["results"]] == ["ASK", "ASK"]
    assert all(len(item["waveform_sha256"]) == 64 for item in result["results"])


def test_ood_evaluator_reports_abstention_and_rejects_known_taxonomy_inputs(tmp_path):
    model = _model(tmp_path / "model.npz", accept_known=False)
    unknown = tmp_path / "ook.wav"
    generate_ook_wav(unknown, "001101" * 8)
    result = evaluate_unmodeled_wavs(model, [unknown])
    assert result["method_false_accept_rate"] == 0.0
    assert result["family_false_accept_rate"] == 0.0
    assert result["results"][0]["method_abstained"] is True

    known = tmp_path / "fsk.wav"
    generate_2fsk_wav(known, "001101" * 8)
    with pytest.raises(ValueError, match="already represented"):
        evaluate_unmodeled_wavs(model, [known])


def test_ood_evaluator_requires_at_least_one_verified_waveform(tmp_path):
    model = _model(tmp_path / "model.npz", accept_known=False)
    with pytest.raises(ValueError, match="at least one"):
        evaluate_unmodeled_wavs(model, [])
