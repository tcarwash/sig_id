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

from .classifier import FEATURE_VERSION, extract_features, fit_softmax, predict_probabilities
from .corpus import MODES
from .dataset import CaptureStore
from .splits import FAMILY_BY_MODE, METHOD_BY_MODE
from .wav import WavSource


TARGET_VALIDATION_PRECISION = 0.80
MIN_CALIBRATION_EXAMPLES = 6
MAX_VALIDATION_NOISE_FALSE_POSITIVE_RATE = 0.0


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
        labels.append(row["mode"])
    return np.asarray(feature_rows, dtype=np.float32), np.asarray(labels, dtype=object), [row["lineage_group"] for row in rows]


def _calibrate_threshold(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    top = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(len(top)), top]
    correct = top == labels
    known = labels >= 0
    negatives = labels < 0
    candidates = sorted(set(float(value) for value in confidence))
    valid = []
    for threshold in candidates:
        selected_known = (confidence >= threshold) & known
        selected_negative = (confidence >= threshold) & negatives
        count = int(selected_known.sum())
        precision = float(np.mean(correct[selected_known])) if count else 0.0
        noise_fpr = float(np.mean(selected_negative[negatives])) if np.any(negatives) else None
        if (
            count >= MIN_CALIBRATION_EXAMPLES
            and precision >= TARGET_VALIDATION_PRECISION
            and noise_fpr is not None
            and noise_fpr <= MAX_VALIDATION_NOISE_FALSE_POSITIVE_RATE
        ):
            valid.append((threshold, count, precision, noise_fpr))
    if valid:
        threshold, count, precision, noise_fpr = min(valid, key=lambda value: value[0])
        status = "calibrated"
    else:
        threshold, count, precision = 1.000001, 0, 0.0
        noise_fpr = 0.0
        status = "reject_all_no_threshold_meets_validation_constraints"
    return {
        "status": status,
        "threshold": float(threshold),
        "target_precision": TARGET_VALIDATION_PRECISION,
        "validation_accepted": count,
        "validation_precision": precision,
        "validation_count": int(len(labels)),
        "validation_noise_count": int(negatives.sum()),
        "validation_noise_false_positive_rate": noise_fpr,
        "maximum_validation_noise_false_positive_rate": MAX_VALIDATION_NOISE_FALSE_POSITIVE_RATE,
    }


def _evaluate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    threshold: float,
    family_threshold: float,
) -> dict[str, Any]:
    top = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(len(top)), top]
    accepted = confidence >= threshold
    correct = top == labels
    confusion = {truth: {prediction: 0 for prediction in [*classes, "unknown"]} for truth in [*classes, "unknown"]}
    raw_confusion = {truth: {prediction: 0 for prediction in classes} for truth in [*classes, "unknown"]}
    family_names = ("FSK", "MFSK", "PSK")
    family_probabilities = np.asarray([
        probabilities[:, [i for i, name in enumerate(classes) if FAMILY_BY_MODE[name] == family]].sum(axis=1)
        for family in family_names
    ]).T
    family_top = np.argmax(family_probabilities, axis=1)
    family_labels = np.asarray([
        family_names.index(FAMILY_BY_MODE[classes[int(label)]]) if int(label) >= 0 else -1 for label in labels
    ])
    family_confidence = family_probabilities[np.arange(len(family_top)), family_top]
    family_correct = family_top == family_labels
    family_accepted = family_confidence >= family_threshold
    family_confusion = {truth: {prediction: 0 for prediction in [*family_names, "unknown"]} for truth in [*family_names, "unknown"]}
    for truth, guess, is_accepted in zip(labels, top, accepted, strict=True):
        truth_name = classes[int(truth)] if int(truth) >= 0 else "unknown"
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
        "unknown_examples": int(np.sum(labels < 0)),
        "unknown_accepted": int(np.sum((labels < 0) & accepted)),
        "unknown_false_positive_rate": float(np.mean(accepted[labels < 0])) if np.any(labels < 0) else None,
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
) -> dict[str, Any]:
    split_path = Path(split_manifest_path).expanduser().resolve(strict=True)
    data_dir = Path(data_dir).expanduser().resolve()
    output_model_path = Path(output_model_path).expanduser().resolve()
    if output_model_path.suffix != ".npz":
        raise ValueError("output model path must end in .npz")
    if epochs < 1 or epochs > 20_000:
        raise ValueError("epochs must be between 1 and 20000")
    split_digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    rows = _validated_rows(split_path, CaptureStore(data_dir / "dataset.sqlite3"))
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]
    known_train_rows = [row for row in train_rows if row["mode"] in MODES]
    x_train, y_train_names, _ = _features_for(known_train_rows)
    x_validation, y_validation_names, validation_groups = _features_for(validation_rows)
    x_test, y_test_names, test_groups = _features_for(test_rows)
    train_groups = [row["lineage_group"] for row in train_rows]
    if set(train_groups) & set(validation_groups) or set(train_groups) & set(test_groups) or set(validation_groups) & set(test_groups):
        raise ValueError("lineage leakage detected before training")
    classes = list(MODES)
    parameters = fit_softmax(x_train, y_train_names, classes, epochs=epochs)
    class_indices = {name: index for index, name in enumerate(classes)}
    validation_indices = np.asarray([class_indices.get(name, -1) for name in y_validation_names], dtype=np.int64)
    test_indices = np.asarray([class_indices.get(name, -1) for name in y_test_names], dtype=np.int64)
    validation_probabilities = predict_probabilities(x_validation, parameters)
    calibration = _calibrate_threshold(validation_probabilities, validation_indices)
    threshold = float(calibration["threshold"])
    family_names = ("FSK", "MFSK", "PSK")
    family_probabilities = np.asarray([
        validation_probabilities[:, [i for i, name in enumerate(classes) if FAMILY_BY_MODE[name] == family]].sum(axis=1)
        for family in family_names
    ]).T
    validation_family_indices = np.asarray(
        [family_names.index(FAMILY_BY_MODE[name]) if name in FAMILY_BY_MODE else -1 for name in y_validation_names], dtype=np.int64
    )
    family_calibration = _calibrate_threshold(family_probabilities, validation_family_indices)
    family_threshold = float(family_calibration["threshold"])
    test_probabilities = predict_probabilities(x_test, parameters)
    evaluation = {
        "schema_version": 1,
        "model_type": "multinomial logistic regression",
        "feature_version": FEATURE_VERSION,
        "split_manifest_sha256": split_digest,
        "calibration": calibration,
        "family_calibration": family_calibration,
        "validation": _evaluate(validation_probabilities, validation_indices, classes, threshold, family_threshold),
        "test": _evaluate(test_probabilities, test_indices, classes, threshold, family_threshold),
        "evaluation_scope": "synthetic generated modulation sources plus noise-only negatives; unseen modulated families and independent radio captures are not validated",
        "split_group_counts": {
            split: len({row["lineage_group"] for row in rows if row["split"] == split})
            for split in ("train", "validation", "test")
        },
        "test_group_ids": sorted(set(test_groups)),
        "validation_group_ids": sorted(set(validation_groups)),
        "train_group_ids": sorted(set(train_groups)),
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
        "method_names": METHOD_BY_MODE,
        "protocol_prediction_enabled": False,
        "unknown_rejection_validated": False,
        "noise_rejection_scope": "synthetic_noise_only",
        "noise_rejection_validated": bool(
            calibration["status"] == "calibrated"
            and any(row["mode"] == "noise" for row in validation_rows)
            and any(row["mode"] == "noise" for row in test_rows)
            and evaluation["test"]["unknown_false_positive_rate"] == 0.0
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
    args = parser.parse_args()
    try:
        result = train_and_evaluate(
            args.split,
            args.data_dir,
            args.output_model,
            epochs=args.epochs,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"model": str(args.output_model), "evaluation": result["evaluation_path"], "test": result["evaluation"]["test"]}, indent=2))


if __name__ == "__main__":
    main()
