"""Training, calibration and held-out evaluation for the baseline method model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .classifier import FEATURE_VERSION, NOISE_CLASS, extract_features, fit_softmax, predict_probabilities
from .corpus import MODES
from .dataset import CaptureStore
from .ood_corpus import load_ood_corpus
from .splits import FAMILY_BY_MODE, METHOD_BY_MODE
from .wav import WavSource


TARGET_VALIDATION_PRECISION = 0.80
MIN_CALIBRATION_EXAMPLES = 6
MAX_VALIDATION_NOISE_FALSE_POSITIVE_RATE = 0.0
MAX_VALIDATION_UNKNOWN_FALSE_POSITIVE_RATE = 0.0


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
        temporary = Path(file.name)
    temporary.replace(path)


def _validated_rows(split_path: Path, store: CaptureStore) -> list[dict[str, Any]]:
    split_raw = split_path.read_bytes()
    split_manifest = json.loads(split_raw)
    if split_manifest.get("schema_version") != 1 or split_manifest.get("generator") != "sig-id-split-corpus":
        raise ValueError("unsupported split manifest")
    records = split_manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("split manifest must contain records")
    allowed_splits = {"train", "validation", "test"}
    group_split: dict[str, str] = {}
    group_mode: dict[str, str] = {}
    capture_ids: set[str] = set()
    rows = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("split records must be objects")
        capture_id, group, mode, split = (
            record.get("capture_id"), record.get("lineage_group"), record.get("mode"), record.get("split")
        )
        if not all(isinstance(value, str) for value in (capture_id, group, mode)) or mode not in (*MODES, "noise") or split not in allowed_splits:
            raise ValueError("split record has invalid capture, lineage, mode or split fields")
        if capture_id in capture_ids:
            raise ValueError(f"capture {capture_id} appears more than once in split manifest")
        capture_ids.add(capture_id)
        if group in group_split and group_split[group] != split:
            raise ValueError(f"lineage leakage: source group {group} crosses split boundaries")
        if group in group_mode and group_mode[group] != mode:
            raise ValueError(f"lineage group {group} has conflicting method labels")
        group_split[group] = split
        group_mode[group] = mode
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise ValueError(f"split capture {capture_id} is missing from the dataset") from exc
        if capture["source_kind"] != "synthetic" or capture["sha256"] != record.get("sha256"):
            raise ValueError(f"split capture {capture_id} does not match its stored hash")
        if capture["parent_capture_id"] != record.get("parent_capture_id"):
            raise ValueError(f"split capture {capture_id} does not match its declared parent")
        regions = store.regions(capture_id)
        if len(regions) != 1:
            raise ValueError(f"split capture {capture_id} must have one full-capture method label")
        label = regions[0]
        if mode == "noise":
            valid_label = (
                label["sample_start"] == 0
                and label["sample_count"] == capture["sample_count"]
                and label["source"] == "synthetic"
                and label["review_state"] == "accepted"
                and label["family"] is None
                and label["method"] is None
                and label["protocol"] is None
                and label["variant"] is None
                and record.get("kind") == "noise"
            )
        else:
            valid_label = (
                label["sample_start"] == 0
                and label["sample_count"] == capture["sample_count"]
                and label["source"] == "synthetic"
                and label["review_state"] == "accepted"
                and label["family"] == FAMILY_BY_MODE[mode]
                and label["method"] == METHOD_BY_MODE[mode]
                and label["protocol"] is None
                and label["variant"] is None
            )
        if not valid_label:
            raise ValueError(f"split capture {capture_id} label does not match the claimed modulation method")
        rows.append({**record, "audio_path": capture["audio_path"], "sample_rate": capture["sample_rate"]})

    for split in allowed_splits:
        present = {row["mode"] for row in rows if row["split"] == split}
        if not set(MODES).issubset(present):
            raise ValueError(f"split {split} must contain all modulation methods")
        if split in {"validation", "test"} and "noise" not in present:
            raise ValueError(f"split {split} must contain noise-only negative examples")
    for group, mode in group_mode.items():
        if not any(row["lineage_group"] == group and row["capture_id"] == group for row in rows):
            raise ValueError(f"lineage group {group} has no clean root record")
    return rows


def _features_for(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_rows = []
    labels = []
    for row in rows:
        values = np.asarray([sample for block in WavSource(row["audio_path"]).blocks() for sample in block.samples])
        feature_rows.append(extract_features(values, int(row["sample_rate"])))
        labels.append(NOISE_CLASS if row["mode"] == "noise" else row["mode"])
    return np.asarray(feature_rows, dtype=np.float32), np.asarray(labels, dtype=object), [row["lineage_group"] for row in rows]


def _ood_features_for(rows: list[dict[str, Any]]) -> np.ndarray:
    features = []
    for row in rows:
        values = np.asarray(
            [sample for block in WavSource(row["waveform_path"]).blocks() for sample in block.samples],
            dtype=np.float64,
        )
        features.append(extract_features(values, int(row["sample_rate"])))
    return np.asarray(features, dtype=np.float32)


def _evaluate_unmodeled(
    probabilities: np.ndarray,
    classes: list[str],
    threshold: float,
    family_threshold: float,
    true_methods: list[str] | None = None,
) -> dict[str, Any]:
    if probabilities.ndim != 2 or not probabilities.shape[0]:
        raise ValueError("OOD evaluation requires at least one probability row")
    def summarize(matrix: np.ndarray) -> dict[str, Any]:
        noise_index = classes.index(NOISE_CLASS)
        top = np.argmax(matrix, axis=1)
        confidence = matrix[np.arange(len(top)), top]
        accepted = (confidence >= threshold) & (top != noise_index)
        family_names = ("FSK", "MFSK", "PSK")
        family_probabilities = np.asarray([
            matrix[:, [i for i, name in enumerate(classes) if name in FAMILY_BY_MODE and FAMILY_BY_MODE[name] == family]].sum(axis=1)
            for family in family_names
        ]).T
        family_confidence = np.max(family_probabilities, axis=1)
        family_accepted = (family_confidence >= family_threshold) & (top != noise_index)
        return {
            "examples": len(top),
            "method_false_accepts": int(accepted.sum()),
            "method_false_accept_rate": float(np.mean(accepted)),
            "family_false_accepts": int(family_accepted.sum()),
            "family_false_accept_rate": float(np.mean(family_accepted)),
            "top_candidate_counts": {
                name: int(np.sum(top == index)) for index, name in enumerate(classes)
            },
            "method_confidence_mean": float(np.mean(confidence)),
            "family_confidence_mean": float(np.mean(family_confidence)),
        }

    result = summarize(probabilities)
    if true_methods is not None:
        if len(true_methods) != len(probabilities):
            raise ValueError("OOD truth methods must align with probability rows")
        result["by_true_method"] = {
            method: summarize(probabilities[np.asarray([value == method for value in true_methods], dtype=bool)])
            for method in sorted(set(true_methods))
        }
    return result


def _include_scene_features(
    features: np.ndarray, labels: np.ndarray, scene_data: tuple[np.ndarray, np.ndarray, dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    scene_features, scene_labels, _ = scene_data
    if not scene_features.size:
        return features, labels
    return np.concatenate((features, scene_features), axis=0), np.concatenate((labels, scene_labels), axis=0)


def _mixed_scene_features(
    split_manifest: dict[str, Any], store: CaptureStore, split_name: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract labeled scene crops only when every source lineage shares this split."""
    dependencies = split_manifest.get("mixed_scene_dependencies", [])
    records = split_manifest.get("records", [])
    source_groups = {record["capture_id"]: record["lineage_group"] for record in records}
    group_splits = {record["lineage_group"]: record["split"] for record in records}
    mode_by_method = {method: mode for mode, method in METHOD_BY_MODE.items()}
    features: list[np.ndarray] = []
    labels: list[str] = []
    scene_ids: set[str] = set()
    scene_groups: set[str] = set()
    skipped_components = 0

    for dependency in dependencies:
        if dependency.get("split") != split_name:
            continue
        scene_id = dependency.get("capture_id")
        scene = store.capture(scene_id)
        if scene["source_kind"] != "synthetic" or scene["sha256"] != dependency.get("sha256"):
            raise ValueError(f"mixed-scene {split_name} capture {scene_id} does not match its split manifest")
        metadata = scene["source_metadata"]
        component_ids = dependency.get("component_capture_ids")
        if (
            metadata.get("generator") != "sig-id-mixed-scene"
            or not isinstance(component_ids, list)
            or metadata.get("component_capture_ids") != component_ids
            or any(component_id not in source_groups for component_id in component_ids)
        ):
            raise ValueError(f"mixed-scene {split_name} capture {scene_id} has invalid source dependencies")
        scene_lineages = {source_groups[component_id] for component_id in component_ids}
        if (
            set(dependency.get("lineage_groups", [])) != scene_lineages
            or any(group_splits.get(group) != split_name for group in scene_lineages)
        ):
            raise ValueError(f"mixed-scene {split_name} capture {scene_id} shares a source lineage outside {split_name}")

        regions = store.regions(scene_id)
        if len(regions) != len(component_ids):
            raise ValueError(f"mixed-scene {split_name} capture {scene_id} has inconsistent constituent regions")
        waveform = Path(scene["audio_path"])
        digest = hashlib.sha256(waveform.read_bytes()).hexdigest()
        if digest != scene["sha256"]:
            raise ValueError(f"mixed-scene {split_name} waveform {scene_id} changed after import")
        audio = np.asarray(
            [sample for block in WavSource(waveform).blocks() for sample in block.samples],
            dtype=np.float64,
        )
        for region in regions:
            evidence = region["evidence"]
            component_id = evidence.get("component_capture_id")
            if component_id not in component_ids:
                raise ValueError(f"mixed-scene {split_name} region {region['id']} has no source capture evidence")
            source_capture = store.capture(component_id)
            if source_capture["sha256"] != evidence.get("component_sha256"):
                raise ValueError(f"mixed-scene {split_name} source {component_id} changed after scene import")
            source_regions = store.regions(component_id)
            source_region = next(
                (item for item in source_regions if item["id"] == evidence.get("source_region_id")),
                None,
            )
            if source_region is None or any(
                region[field] != source_region[field]
                for field in ("family", "method", "protocol", "variant")
            ):
                raise ValueError(f"mixed-scene {split_name} region {region['id']} label disagrees with its source")
            start = int(region["sample_start"])
            end = start + int(region["sample_count"])
            if start < 0 or end > len(audio):
                raise ValueError(f"mixed-scene {split_name} region {region['id']} exceeds its waveform")
            mode = mode_by_method.get(region["method"])
            if region["review_state"] != "accepted" or region["source"] != "synthetic":
                raise ValueError(f"mixed-scene {split_name} region {region['id']} is not accepted synthetic truth")
            if mode is None:
                skipped_components += 1
                continue
            segment = audio[start:end]
            if segment.size < 1024:
                skipped_components += 1
                continue
            features.append(extract_features(segment, int(scene["sample_rate"])))
            labels.append(mode)
            scene_ids.add(scene_id)
        scene_groups.update(scene_lineages)

    feature_array = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels, dtype=object)
    details = {
        "scenes": len(scene_ids),
        "examples": len(labels),
        "skipped_components": skipped_components,
        "scene_capture_ids": sorted(scene_ids),
        "source_lineage_groups": sorted(scene_groups),
        "split": split_name,
    }
    return feature_array, label_array, details


def _calibrate_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    noise_class_index: int | None = None,
) -> dict[str, Any]:
    top = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(len(top)), top]
    correct = top == labels
    if noise_class_index is None:
        known = labels >= 0
        negative_mask = labels < 0
        noise_mask = negative_mask
        unknown_mask = negative_mask
        positive_predictions = np.ones(len(top), dtype=bool)
    else:
        known = (labels != noise_class_index) & (labels >= 0)
        noise_mask = labels == noise_class_index
        unknown_mask = labels < 0
        negative_mask = noise_mask | unknown_mask
        positive_predictions = top != noise_class_index
    candidates = sorted(set(float(value) for value in confidence))
    valid = []
    for threshold in candidates:
        selected_known = (confidence >= threshold) & known & positive_predictions
        selected_negative = (confidence >= threshold) & negative_mask & positive_predictions
        count = int(selected_known.sum())
        precision = float(np.mean(correct[selected_known])) if count else 0.0
        negative_fpr = float(np.mean(selected_negative[negative_mask])) if np.any(negative_mask) else None
        noise_fpr = float(np.mean(selected_negative[noise_mask])) if np.any(noise_mask) else None
        unknown_fpr = float(np.mean(selected_negative[unknown_mask])) if np.any(unknown_mask) else None
        if (
            count >= MIN_CALIBRATION_EXAMPLES
            and precision >= TARGET_VALIDATION_PRECISION
            and negative_fpr is not None
            and (noise_fpr is None or noise_fpr <= MAX_VALIDATION_NOISE_FALSE_POSITIVE_RATE)
            and (unknown_fpr is None or unknown_fpr <= MAX_VALIDATION_UNKNOWN_FALSE_POSITIVE_RATE)
        ):
            valid.append((threshold, count, precision, negative_fpr, noise_fpr, unknown_fpr))
    if valid:
        threshold, count, precision, negative_fpr, noise_fpr, unknown_fpr = min(valid, key=lambda value: value[0])
        status = "calibrated"
    else:
        threshold, count, precision = 1.000001, 0, 0.0
        negative_fpr = 0.0
        noise_fpr = 0.0 if np.any(noise_mask) else None
        unknown_fpr = 0.0 if np.any(unknown_mask) else None
        status = "reject_all_no_threshold_meets_validation_constraints"
    return {
        "status": status,
        "threshold": float(threshold),
        "target_precision": TARGET_VALIDATION_PRECISION,
        "validation_accepted": count,
        "validation_precision": precision,
        "validation_count": int(len(labels)),
        "validation_noise_count": int(noise_mask.sum()),
        "validation_noise_false_positive_rate": noise_fpr,
        "maximum_validation_noise_false_positive_rate": MAX_VALIDATION_NOISE_FALSE_POSITIVE_RATE,
        "validation_unknown_count": int(unknown_mask.sum()),
        "validation_unknown_false_positive_rate": unknown_fpr,
        "maximum_validation_unknown_false_positive_rate": MAX_VALIDATION_UNKNOWN_FALSE_POSITIVE_RATE,
        "validation_negative_false_positive_rate": negative_fpr,
    }


def _evaluate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    threshold: float,
    family_threshold: float,
) -> dict[str, Any]:
    noise_index = classes.index(NOISE_CLASS)
    top = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(len(top)), top]
    accepted = (confidence >= threshold) & (top != noise_index)
    correct = top == labels
    confusion = {truth: {prediction: 0 for prediction in [*classes, "unknown"]} for truth in classes}
    raw_confusion = {truth: {prediction: 0 for prediction in classes} for truth in classes}
    family_names = ("FSK", "MFSK", "PSK")
    family_probabilities = np.asarray([
        probabilities[:, [i for i, name in enumerate(classes) if name in FAMILY_BY_MODE and FAMILY_BY_MODE[name] == family]].sum(axis=1)
        for family in family_names
    ]).T
    family_top = np.argmax(family_probabilities, axis=1)
    family_labels = np.asarray([
        family_names.index(FAMILY_BY_MODE[classes[int(label)]])
        if int(label) >= 0 and classes[int(label)] in FAMILY_BY_MODE else -1 for label in labels
    ])
    family_confidence = family_probabilities[np.arange(len(family_top)), family_top]
    family_correct = family_top == family_labels
    family_accepted = family_confidence >= family_threshold
    family_confusion = {truth: {prediction: 0 for prediction in [*family_names, "unknown"]} for truth in [*family_names, "unknown"]}
    for truth, guess, is_accepted in zip(labels, top, accepted, strict=True):
        truth_name = classes[int(truth)]
        confusion[truth_name][classes[int(guess)] if is_accepted else "unknown"] += 1
        raw_confusion[truth_name][classes[int(guess)]] += 1
    for truth, guess, is_accepted in zip(family_labels, family_top, family_accepted, strict=True):
        truth_name = family_names[int(truth)] if int(truth) >= 0 else "unknown"
        family_confusion[truth_name][family_names[int(guess)] if is_accepted else "unknown"] += 1
    count = int(accepted.sum())
    return {
        "examples": int(len(labels)),
        "raw_accuracy": float(np.mean(correct)) if len(labels) else None,
        "coverage": count / len(labels) if len(labels) else 0.0,
        "accepted_accuracy": float(np.mean(correct[accepted])) if count else None,
        "accepted_examples": count,
        "rejected_examples": int(len(labels) - count),
        "confusion": confusion,
        "raw_confusion": raw_confusion,
        "family_raw_accuracy": float(np.mean(family_correct)) if len(labels) else None,
        "family_coverage": float(np.mean(family_accepted)) if len(labels) else 0.0,
        "family_accepted_accuracy": float(np.mean(family_correct[family_accepted])) if family_accepted.any() else None,
        "family_confusion": family_confusion,
        "noise_examples": int(np.sum(labels == noise_index)),
        "noise_class_correct": int(np.sum((labels == noise_index) & (top == noise_index))),
        "noise_accepted": int(np.sum((labels == noise_index) & accepted)),
        "noise_false_positive_rate": float(np.mean(accepted[labels == noise_index])) if np.any(labels == noise_index) else None,
        "unknown_family_accepted": int(np.sum((family_labels < 0) & family_accepted)),
        "unknown_family_false_positive_rate": (
            float(np.mean(family_accepted[family_labels < 0])) if np.any(family_labels < 0) else None
        ),
    }


def train_and_evaluate(
    split_manifest_path: str | Path,
    data_dir: str | Path,
    output_model_path: str | Path,
    *,
    epochs: int = 1200,
    ood_corpus_index: str | Path | None = None,
) -> dict[str, Any]:
    split_path = Path(split_manifest_path).expanduser().resolve(strict=True)
    data_dir = Path(data_dir).expanduser().resolve()
    output_model_path = Path(output_model_path).expanduser().resolve()
    if output_model_path.suffix != ".npz":
        raise ValueError("output model path must end in .npz")
    if epochs < 1 or epochs > 20_000:
        raise ValueError("epochs must be between 1 and 20000")
    split_digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    ood_index_digest = (
        hashlib.sha256(Path(ood_corpus_index).expanduser().resolve(strict=True).read_bytes()).hexdigest()
        if ood_corpus_index is not None else None
    )
    store = CaptureStore(data_dir / "dataset.sqlite3")
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    rows = _validated_rows(split_path, store)
    ood_rows = load_ood_corpus(ood_corpus_index) if ood_corpus_index is not None else []
    ood_validation_rows = [row for row in ood_rows if row["split"] == "validation"]
    ood_test_rows = [row for row in ood_rows if row["split"] == "test"]
    if ood_corpus_index is not None and (
        len(ood_validation_rows) < MIN_CALIBRATION_EXAMPLES or not ood_test_rows
    ):
        raise ValueError(
            f"OOD corpus needs at least {MIN_CALIBRATION_EXAMPLES} validation sources and one test source"
        )
    known_hashes = {row["sha256"] for row in rows}
    ood_hashes = {row["waveform_sha256"] for row in ood_rows}
    if known_hashes & ood_hashes:
        raise ValueError("OOD waveform hashes overlap the known-mode train/validation/test corpus")
    scene_features = {
        split: _mixed_scene_features(split_manifest, store, split)
        for split in ("train", "validation", "test")
    }
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]
    x_train, y_train_names, train_row_groups = _features_for(train_rows)
    x_validation, y_validation_names, validation_row_groups = _features_for(validation_rows)
    x_test, y_test_names, test_row_groups = _features_for(test_rows)
    x_ood_validation = _ood_features_for(ood_validation_rows) if ood_validation_rows else np.empty((0, 260), dtype=np.float32)
    x_ood_test = _ood_features_for(ood_test_rows) if ood_test_rows else np.empty((0, 260), dtype=np.float32)
    x_train, y_train_names = _include_scene_features(x_train, y_train_names, scene_features["train"])
    x_validation, y_validation_names = _include_scene_features(x_validation, y_validation_names, scene_features["validation"])
    x_test, y_test_names = _include_scene_features(x_test, y_test_names, scene_features["test"])
    train_groups = set(train_row_groups) | set(scene_features["train"][2]["source_lineage_groups"])
    validation_groups = set(validation_row_groups) | set(scene_features["validation"][2]["source_lineage_groups"])
    test_groups = set(test_row_groups) | set(scene_features["test"][2]["source_lineage_groups"])
    if set(train_groups) & set(validation_groups) or set(train_groups) & set(test_groups) or set(validation_groups) & set(test_groups):
        raise ValueError("lineage leakage detected before training")
    classes = [*MODES, NOISE_CLASS]
    parameters = fit_softmax(x_train, y_train_names, classes, epochs=epochs)
    class_indices = {name: index for index, name in enumerate(classes)}
    validation_indices = np.asarray([class_indices.get(name, -1) for name in y_validation_names], dtype=np.int64)
    test_indices = np.asarray([class_indices.get(name, -1) for name in y_test_names], dtype=np.int64)
    validation_probabilities = predict_probabilities(x_validation, parameters)
    calibration_probabilities = validation_probabilities
    calibration_labels = validation_indices
    ood_validation_probabilities = np.empty((0, len(classes)), dtype=np.float64)
    if len(x_ood_validation):
        ood_validation_probabilities = predict_probabilities(x_ood_validation, parameters)
        calibration_probabilities = np.concatenate((validation_probabilities, ood_validation_probabilities), axis=0)
        calibration_labels = np.concatenate((validation_indices, np.full(len(x_ood_validation), -1, dtype=np.int64)))
    calibration = _calibrate_threshold(
        calibration_probabilities, calibration_labels, noise_class_index=class_indices[NOISE_CLASS]
    )
    threshold = float(calibration["threshold"])
    family_names = ("FSK", "MFSK", "PSK")
    family_probabilities = np.asarray([
        validation_probabilities[:, [i for i, name in enumerate(classes) if name in FAMILY_BY_MODE and FAMILY_BY_MODE[name] == family]].sum(axis=1)
        for family in family_names
    ]).T
    validation_family_indices = np.asarray(
        [family_names.index(FAMILY_BY_MODE[name]) if name in FAMILY_BY_MODE else -1 for name in y_validation_names], dtype=np.int64
    )
    calibration_family_probabilities = family_probabilities
    calibration_family_indices = validation_family_indices
    if len(ood_validation_probabilities):
        ood_validation_family_probabilities = np.asarray([
            ood_validation_probabilities[:, [i for i, name in enumerate(classes) if name in FAMILY_BY_MODE and FAMILY_BY_MODE[name] == family]].sum(axis=1)
            for family in family_names
        ]).T
        calibration_family_probabilities = np.concatenate((family_probabilities, ood_validation_family_probabilities), axis=0)
        calibration_family_indices = np.concatenate((validation_family_indices, np.full(len(x_ood_validation), -1, dtype=np.int64)))
    family_calibration = _calibrate_threshold(calibration_family_probabilities, calibration_family_indices)
    family_threshold = float(family_calibration["threshold"])
    test_probabilities = predict_probabilities(x_test, parameters)
    ood_test_metrics = None
    if len(x_ood_test):
        ood_test_probabilities = predict_probabilities(x_ood_test, parameters)
        ood_test_metrics = _evaluate_unmodeled(
            ood_test_probabilities,
            classes,
            threshold,
            family_threshold,
            [row["modulation_method"] for row in ood_test_rows],
        )
    mixed_scene_evaluation = None
    mixed_scene_x, mixed_scene_names, mixed_scene_details = scene_features["test"]
    if mixed_scene_names.size:
        mixed_scene_indices = np.asarray([class_indices.get(name, -1) for name in mixed_scene_names], dtype=np.int64)
        mixed_scene_probabilities = predict_probabilities(mixed_scene_x, parameters)
        mixed_scene_evaluation = _evaluate(
            mixed_scene_probabilities, mixed_scene_indices, classes, threshold, family_threshold
        )
    evaluation = {
        "schema_version": 1,
        "model_type": "multinomial logistic regression",
        "feature_version": FEATURE_VERSION,
        "split_manifest_sha256": split_digest,
        "calibration": calibration,
        "family_calibration": family_calibration,
        "validation": _evaluate(validation_probabilities, validation_indices, classes, threshold, family_threshold),
        "test": _evaluate(test_probabilities, test_indices, classes, threshold, family_threshold),
        "ood_validation": {
            "examples": len(ood_validation_rows),
            "source_indices": [row["index"] for row in ood_validation_rows],
            "waveform_sha256": [row["waveform_sha256"] for row in ood_validation_rows],
            "scope": "threshold calibration only; excluded from classifier fitting",
        },
        "ood_test": {
            "metrics": ood_test_metrics,
            "source_indices": [row["index"] for row in ood_test_rows],
            "waveform_sha256": [row["waveform_sha256"] for row in ood_test_rows],
            "scope": "independent synthetic ASK/OOK lineages excluded from model fitting and threshold calibration",
        },
        "ood_corpus_index_sha256": ood_index_digest,
        "mixed_scene_test": {
            **mixed_scene_details,
            "metrics": mixed_scene_evaluation,
            "scope": "held-out component regions scored in their overlapping scene context",
        },
        "mixed_scene_training": scene_features["train"][2],
        "mixed_scene_validation": scene_features["validation"][2],
        "evaluation_scope": "synthetic generated modulation sources and lineage-safe mixed-scene constituent crops plus noise-only negatives; held-out ASK rejection across OOK and 4-ASK is evaluated only when a separate OOD corpus is supplied; independent radio captures remain unvalidated",
        "split_group_counts": {
            split: len({row["lineage_group"] for row in rows if row["split"] == split})
            for split in ("train", "validation", "test")
        },
        "test_group_ids": sorted(test_groups),
        "validation_group_ids": sorted(validation_groups),
        "train_group_ids": sorted(train_groups),
    }
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=output_model_path.parent, delete=False) as file:
        np.savez_compressed(file, **parameters)
        temporary_model_path = Path(file.name)
    temporary_model_path.replace(output_model_path)
    model_digest = hashlib.sha256(output_model_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "model_type": "multinomial logistic regression",
        "feature_version": FEATURE_VERSION,
        "feature_count": int(x_train.shape[1]),
        "sample_rate_hz": int(train_rows[0]["sample_rate"]),
        "classes": classes,
        "family_by_method": FAMILY_BY_MODE,
        "noise_class": NOISE_CLASS,
        "noise_class_trained": any(row["mode"] == "noise" for row in train_rows),
        "method_names": METHOD_BY_MODE,
        "protocol_prediction_enabled": False,
        "unknown_rejection_validated": bool(
            calibration["status"] == "calibrated"
            and ood_test_metrics is not None
            and ood_test_metrics["method_false_accept_rate"] == 0.0
            and ood_test_metrics["family_false_accept_rate"] == 0.0
        ),
        "unknown_rejection_scope": "synthetic ASK held-out" if ood_test_metrics is not None else "unvalidated",
        "ood_corpus_index_sha256": ood_index_digest,
        "ood_sources_fit_on": 0,
        "noise_class_validated": bool(
            evaluation["test"]["noise_examples"]
            and evaluation["test"]["noise_class_correct"]
            and evaluation["test"]["noise_false_positive_rate"] == 0.0
        ),
        "noise_rejection_scope": "synthetic_noise_only",
        "noise_rejection_validated": bool(
            calibration["status"] == "calibrated"
            and any(row["mode"] == "noise" for row in validation_rows)
            and any(row["mode"] == "noise" for row in test_rows)
            and evaluation["test"]["noise_false_positive_rate"] == 0.0
            and evaluation["test"]["unknown_family_false_positive_rate"] == 0.0
        ),
        "abstention_threshold": threshold,
        "family_abstention_threshold": family_threshold,
        "threshold_calibration": calibration,
        "training_epochs": epochs,
        "split_manifest_sha256": split_digest,
        "model_sha256": model_digest,
        "model_file": output_model_path.name,
    }
    _atomic_json(output_model_path.with_suffix(".json"), metadata)
    evaluation_path = output_model_path.with_name("evaluation.json")
    evaluation["model_sha256"] = model_digest
    _atomic_json(evaluation_path, evaluation)
    return {"model": metadata, "evaluation": evaluation, "evaluation_path": str(evaluation_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the synthetic modulation-method baseline")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("SIG_ID_DATA_DIR", "~/.local/share/sig-id")).expanduser())
    parser.add_argument("--output-model", type=Path, required=True, help="output .npz model path")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--ood-corpus", type=Path, help="separate synthetic OOK corpus index for threshold calibration and held-out evaluation")
    args = parser.parse_args()
    try:
        result = train_and_evaluate(
            args.split,
            args.data_dir,
            args.output_model,
            epochs=args.epochs,
            ood_corpus_index=args.ood_corpus,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"model": str(args.output_model), "evaluation": result["evaluation_path"], "test": result["evaluation"]["test"]}, indent=2))


if __name__ == "__main__":
    main()
