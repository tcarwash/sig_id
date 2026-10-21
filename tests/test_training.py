from __future__ import annotations

import json

import numpy as np
import pytest

from sigid.classifier import extract_features, load_model, propose
from sigid.corpus import MODES, build_corpus
from sigid.dataset import CaptureStore
from sigid.splits import create_split_manifest
from sigid.training import _calibrate_threshold, train_and_evaluate
from sigid.wav import WavSource


def test_feature_extractor_has_fixed_shape_and_gain_invariance():
    sample_rate = 12000
    samples = np.sin(2 * np.pi * 1500 * np.arange(6000) / sample_rate)
    baseline = extract_features(samples, sample_rate)
    scaled = extract_features(samples * 0.25, sample_rate)
    assert baseline.shape == (260,)
    assert np.all(np.isfinite(baseline))
    assert scaled == pytest.approx(baseline, abs=1e-6)


def test_calibration_enforces_zero_noise_false_positive_target():
    probabilities = np.asarray([[0.9, 0.1]] * 6 + [[0.6, 0.4]])
    labels = np.asarray([0] * 6 + [-1])
    result = _calibrate_threshold(probabilities, labels)
    assert result["status"] == "calibrated"
    assert result["validation_accepted"] == 6
    assert result["validation_precision"] == 1.0
    assert result["validation_noise_false_positive_rate"] == 0.0
    assert result["threshold"] > 0.6


def test_training_uses_split_groups_calibrates_abstention_and_exports_loadable_model(tmp_path):
    output = tmp_path / "corpus"
    data = tmp_path / "data"
    corpus = build_corpus(output, data, examples_per_mode=3, snr_levels_db=[10], seed=311)
    store = CaptureStore(data / "dataset.sqlite3")
    split = create_split_manifest(output / "corpus.json", store, seed=29)
    split_path = output / "splits.json"
    split_path.write_text(json.dumps(split))

    model_path = tmp_path / "model" / "method-baseline.npz"
    trained = train_and_evaluate(split_path, data, model_path, epochs=300)
    metadata, parameters = load_model(model_path)
    evaluation = trained["evaluation"]
    assert metadata["classes"] == list(MODES)
    assert metadata["protocol_prediction_enabled"] is False
    assert metadata["unknown_rejection_validated"] is False
    assert metadata["noise_rejection_scope"] == "synthetic_noise_only"
    assert metadata["noise_rejection_validated"] is False
    assert evaluation["evaluation_scope"].startswith("synthetic generated modulation sources plus noise-only negatives")
    assert metadata["split_manifest_sha256"] == trained["evaluation"]["split_manifest_sha256"]
    assert set(evaluation["train_group_ids"]).isdisjoint(evaluation["validation_group_ids"])
    assert set(evaluation["train_group_ids"]).isdisjoint(evaluation["test_group_ids"])
    assert set(evaluation["validation_group_ids"]).isdisjoint(evaluation["test_group_ids"])
    assert evaluation["test"]["examples"] == 13
    assert evaluation["test"]["unknown_examples"] == 1
    assert evaluation["test"]["unknown_accepted"] == 0
    assert evaluation["test"]["unknown_family_false_positive_rate"] == 0.0
    assert evaluation["calibration"]["validation_noise_false_positive_rate"] == 0.0
    assert evaluation["calibration"]["status"] == "reject_all_no_threshold_meets_validation_constraints"
    assert evaluation["test"]["coverage"] == 0.0
    assert 0 <= evaluation["test"]["coverage"] <= 1
    assert evaluation["calibration"]["status"] in {
        "calibrated",
        "reject_all_no_threshold_meets_validation_constraints",
    }

    record = next(record for record in corpus["records"] if record["mode"] == "qpsk" and record["kind"] == "clean")
    capture = store.capture(record["capture_id"])
    samples = np.asarray([sample for block in WavSource(capture["audio_path"]).blocks() for sample in block.samples])
    result = propose(
        extract_features(samples, capture["sample_rate"]),
        parameters,
        metadata["classes"],
        threshold=metadata["abstention_threshold"],
    )
    assert result["protocol"] is None
    assert result["abstained"] or result["family"] in {"FSK", "MFSK", "PSK"}
    assert result["method"] is None or result["method"] in metadata["method_names"].values()

    with model_path.open("ab") as artifact:
        artifact.write(b"tamper")
    with pytest.raises(ValueError, match="metadata hash"):
        load_model(model_path)
