from __future__ import annotations

import json

import numpy as np
import pytest

from sigid.classifier import extract_features, load_model, propose
from sigid.corpus import MODES, build_corpus
from sigid.dataset import CaptureStore
from sigid.scenes import create_mixed_scene, import_mixed_scene
from sigid.splits import create_split_manifest
from sigid.training import _calibrate_threshold, train_and_evaluate
from sigid.ood_corpus import build_ood_corpus
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


def test_calibration_rejects_unknown_modulation_without_fitting_it_as_a_class():
    probabilities = np.asarray([[0.9, 0.1]] * 6 + [[0.8, 0.2]])
    labels = np.asarray([0] * 6 + [-1])
    result = _calibrate_threshold(probabilities, labels, noise_class_index=1)
    assert result["status"] == "calibrated"
    assert result["validation_accepted"] == 6
    assert result["validation_unknown_count"] == 1
    assert result["validation_unknown_false_positive_rate"] == 0.0
    assert result["threshold"] > 0.8


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
    assert metadata["classes"] == [*MODES, "noise"]
    assert metadata["noise_class"] == "noise"
    assert metadata["noise_class_trained"] is True
    assert metadata["protocol_prediction_enabled"] is False
    assert metadata["unknown_rejection_validated"] is False
    assert metadata["noise_class_validated"] is True
    assert metadata["noise_rejection_scope"] == "synthetic_noise_only"
    assert metadata["noise_rejection_validated"] is False
    assert evaluation["evaluation_scope"].startswith("synthetic generated modulation sources and lineage-safe mixed-scene constituent crops")
    assert metadata["split_manifest_sha256"] == trained["evaluation"]["split_manifest_sha256"]
    assert set(evaluation["train_group_ids"]).isdisjoint(evaluation["validation_group_ids"])
    assert set(evaluation["train_group_ids"]).isdisjoint(evaluation["test_group_ids"])
    assert set(evaluation["validation_group_ids"]).isdisjoint(evaluation["test_group_ids"])
    assert evaluation["test"]["examples"] == 13
    assert evaluation["test"]["noise_examples"] == 1
    assert evaluation["test"]["noise_accepted"] == 0
    assert evaluation["test"]["noise_class_correct"] == 1
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

    noise_record = next(record for record in corpus["records"] if record["mode"] == "noise")
    noise_capture = store.capture(noise_record["capture_id"])
    noise_samples = np.asarray([sample for block in WavSource(noise_capture["audio_path"]).blocks() for sample in block.samples])
    noise_result = propose(
        extract_features(noise_samples, noise_capture["sample_rate"]),
        parameters,
        metadata["classes"],
        threshold=metadata["abstention_threshold"],
    )
    assert noise_result["noise_detected"] is True
    assert noise_result["method"] is None

    with model_path.open("ab") as artifact:
        artifact.write(b"tamper")
    with pytest.raises(ValueError, match="metadata hash"):
        load_model(model_path)


def test_training_reports_only_held_out_mixed_scene_regions(tmp_path):
    output = tmp_path / "corpus"
    data = tmp_path / "data"
    build_corpus(output, data, examples_per_mode=4, snr_levels_db=[], seed=912)
    store = CaptureStore(data / "dataset.sqlite3")
    corpus = json.loads((output / "corpus.json").read_text())
    fsk = next(record for record in corpus["records"] if record["mode"] == "2fsk" and record["kind"] == "clean")
    bpsk = next(record for record in corpus["records"] if record["mode"] == "bpsk" and record["kind"] == "clean")
    scene_path = tmp_path / "scenes" / "test-scene.wav"
    create_mixed_scene(
        scene_path,
        [
            {"capture_id": fsk["capture_id"]},
            {"capture_id": bpsk["capture_id"], "scene_start_sample": 1000, "gain": 0.5},
        ],
        store=store,
    )
    scene = import_mixed_scene(
        scene_path,
        scene_path.with_suffix(".wav.json"),
        store=store,
        capture_dir=data / "captures",
    )

    split = create_split_manifest(output / "corpus.json", store, seed=19)
    dependency = next(item for item in split["mixed_scene_dependencies"] if item["capture_id"] == scene["capture"]["id"])
    assert dependency["split"] == "test"

    train_sources = []
    for mode in ("2fsk", "bpsk"):
        train_sources.append(next(
            record for record in split["records"]
            if record["mode"] == mode and record["kind"] == "clean" and record["split"] == "train"
        ))
    train_scene_path = tmp_path / "scenes" / "train-scene.wav"
    train_scene_manifest = create_mixed_scene(
        train_scene_path,
        [{"capture_id": record["capture_id"]} for record in train_sources],
        store=store,
    )
    train_scene = import_mixed_scene(
        train_scene_path,
        train_scene_path.with_suffix(".wav.json"),
        store=store,
        capture_dir=data / "captures",
    )
    split["mixed_scene_dependencies"].append({
        "capture_id": train_scene["capture"]["id"],
        "sha256": train_scene["capture"]["sha256"],
        "component_capture_ids": [record["capture_id"] for record in train_sources],
        "lineage_groups": sorted({record["lineage_group"] for record in train_sources}),
        "split": "train",
    })
    split_path = output / "scene-splits.json"
    split_path.write_text(json.dumps(split))

    result = train_and_evaluate(split_path, data, tmp_path / "model" / "method.npz", epochs=300)
    scene_test = result["evaluation"]["mixed_scene_test"]
    assert scene_test["scenes"] == 1
    assert scene_test["examples"] == 2
    assert scene_test["metrics"]["examples"] == 2
    assert set(scene_test["source_lineage_groups"]) == set(dependency["lineage_groups"])
    assert scene_test["scene_capture_ids"] == [scene["capture"]["id"]]
    assert set(scene_test["source_lineage_groups"]).isdisjoint(result["evaluation"]["train_group_ids"])
    assert set(scene_test["source_lineage_groups"]).isdisjoint(result["evaluation"]["validation_group_ids"])
    scene_train = result["evaluation"]["mixed_scene_training"]
    assert scene_train["scenes"] == 1
    assert scene_train["examples"] == 2
    assert set(scene_train["source_lineage_groups"]).issubset(result["evaluation"]["train_group_ids"])


def test_training_calibrates_unknown_rejection_without_fitting_ood_sources(tmp_path):
    output = tmp_path / "known-corpus"
    data = tmp_path / "data"
    build_corpus(output, data, examples_per_mode=10, snr_levels_db=[], seed=812)
    store = CaptureStore(data / "dataset.sqlite3")
    split = create_split_manifest(output / "corpus.json", store, seed=38)
    split_path = output / "splits.json"
    split_path.write_text(json.dumps(split))
    ood = build_ood_corpus(tmp_path / "held-out-ook", examples=10, seed=901)
    ood_index = tmp_path / "held-out-ook" / "ood-corpus.json"

    trained = train_and_evaluate(
        split_path,
        data,
        tmp_path / "model" / "method.npz",
        epochs=300,
        ood_corpus_index=ood_index,
    )

    calibration = trained["evaluation"]["calibration"]
    ood_test = trained["evaluation"]["ood_test"]
    assert calibration["validation_unknown_count"] == 6
    assert calibration["validation_unknown_false_positive_rate"] == 0.0
    assert len(trained["evaluation"]["ood_validation"]["source_indices"]) == 6
    assert len(trained["evaluation"]["ood_validation"]["waveform_sha256"]) == 6
    assert len(ood_test["source_indices"]) == 4
    assert len(ood_test["waveform_sha256"]) == 4
    assert set(trained["evaluation"]["ood_validation"]["source_indices"]).isdisjoint(ood_test["source_indices"])
    assert trained["evaluation"]["ood_validation"]["scope"].startswith("threshold calibration only")
    assert ood_test["scope"].startswith("independent synthetic ASK/OOK")
    assert 0 <= ood_test["metrics"]["method_false_accept_rate"] <= 1
    assert 0 <= ood_test["metrics"]["family_false_accept_rate"] <= 1
    assert set(ood_test["metrics"]["by_true_method"]) == {
        "on-off keying (OOK)", "4-ary amplitude shift keying"
    }
    assert all(item["examples"] == 2 for item in ood_test["metrics"]["by_true_method"].values())
    assert trained["model"]["unknown_rejection_scope"] == "synthetic ASK held-out"
    assert trained["model"]["ood_sources_fit_on"] == 0
    assert trained["model"]["ood_corpus_index_sha256"] == trained["evaluation"]["ood_corpus_index_sha256"]
    assert trained["model"]["unknown_rejection_validated"] is (
        calibration["status"] == "calibrated"
        and ood_test["metrics"]["method_false_accept_rate"] == 0.0
        and ood_test["metrics"]["family_false_accept_rate"] == 0.0
    )
    assert ood["record_count"] == 10
