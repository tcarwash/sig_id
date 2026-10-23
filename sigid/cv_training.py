"""Train and package a class-agnostic YOLO signal detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .cv_detector import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_FREQUENCY_HZ,
    MODEL_IMAGE_SIZE,
    PREPROCESSING_VERSION,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_yolo_boxes(path: Path) -> np.ndarray:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5 or int(parts[0]) != 0:
            raise ValueError(f"invalid class-agnostic YOLO label in {path}")
        cx, cy, width, height = map(float, parts[1:])
        boxes.append([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])
    return np.asarray(boxes, dtype=np.float64).reshape((-1, 4))


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-12)


def _match_pairs(predicted: np.ndarray, expected: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    pairs = sorted(
        ((_iou(predicted[i], expected[j]), i, j) for i in range(len(predicted)) for j in range(len(expected))),
        reverse=True,
    )
    used_predictions: set[int] = set()
    used_expected: set[int] = set()
    matches = []
    for overlap, prediction, target in pairs:
        if overlap < threshold:
            break
        if prediction not in used_predictions and target not in used_expected:
            used_predictions.add(prediction)
            used_expected.add(target)
            matches.append((prediction, target))
    return matches


def _match_counts(predicted: np.ndarray, expected: np.ndarray, threshold: float) -> tuple[int, int, int]:
    matches = _match_pairs(predicted, expected, threshold)
    return len(matches), len(predicted) - len(matches), len(expected) - len(matches)


def _evaluate_test_set(
    model: Any, dataset: Path, *, confidence: float, device: str | None = None,
) -> dict[str, Any]:
    from PIL import Image

    totals = {"0.25": [0, 0, 0], "0.50": [0, 0, 0]}
    duration_seconds = 0.0
    frequency_center_errors: list[float] = []
    frequency_bound_errors: list[float] = []
    time_center_errors: list[float] = []
    time_bound_errors: list[float] = []
    condition_totals: dict[str, dict[str, list[int]]] = {
        dimension: {}
        for dimension in (
            "source_kind", "signal_overlap", "noise_condition",
            "snr_condition", "drift_condition", "boundary_truncation",
        )
    }
    condition_window_counts: dict[str, dict[str, int]] = {
        dimension: {} for dimension in condition_totals
    }
    signal_type_counts: dict[str, list[int]] = {}
    inference_latencies: list[float] = []
    try:
        import torch
    except ImportError:
        torch = None
    accelerator_available = bool(torch and torch.cuda.is_available() and device != "cpu")
    if accelerator_available:
        torch.cuda.reset_peak_memory_stats()
    image_paths = sorted((dataset / "images" / "test").glob("*.png"))
    if not image_paths:
        raise ValueError("test split has no reviewed windows")
    manifest = [json.loads(line) for line in (dataset / "windows.jsonl").read_text(encoding="utf-8").splitlines()]
    test_windows = {Path(item["image"]).stem: item for item in manifest if item["split"] == "test"}
    for image_path in image_paths:
        label_path = dataset / "labels" / "test" / f"{image_path.stem}.txt"
        expected = _read_yolo_boxes(label_path)
        with Image.open(image_path) as opened:
            image = np.asarray(opened.convert("RGB"))
        if accelerator_available:
            torch.cuda.synchronize()
        inference_start = time.perf_counter()
        predict_options = {
            "conf": confidence, "verbose": False, "imgsz": MODEL_IMAGE_SIZE, "rect": True,
        }
        if device:
            predict_options["device"] = device
        results = model.predict(image, **predict_options)
        if accelerator_available:
            torch.cuda.synchronize()
        inference_latencies.append(time.perf_counter() - inference_start)
        boxes = results[0].boxes
        predicted = (
            boxes.xyxyn.detach().cpu().numpy().astype(np.float64)
            if boxes is not None and len(boxes) else np.empty((0, 4), dtype=np.float64)
        )
        for threshold, key in ((0.25, "0.25"), (0.5, "0.50")):
            counts = _match_counts(predicted, expected, threshold)
            totals[key] = [left + right for left, right in zip(totals[key], counts, strict=True)]
        window = test_windows.get(image_path.stem)
        if window is None:
            raise ValueError(f"test image {image_path.name} is missing from windows.jsonl")
        matches = _match_pairs(predicted, expected, 0.50)
        condition_counts = (len(matches), len(predicted) - len(matches), len(expected) - len(matches))
        conditions = window.get("conditions", {})
        breakdown_keys = {
            "source_kind": [conditions.get("source_kind", "unknown")],
            "signal_overlap": [conditions.get("signal_overlap", "unknown")],
            "noise_condition": [conditions.get("noise_condition", "unknown")],
            "snr_condition": [
                f"snr_db_{float(conditions['snr_db']):g}"
                if isinstance(conditions.get("snr_db"), (int, float))
                and not isinstance(conditions.get("snr_db"), bool)
                and np.isfinite(conditions.get("snr_db")) else "not_available"
            ],
            "drift_condition": [conditions.get("drift_condition", "unknown")],
            "boundary_truncation": [conditions.get("boundary_truncation", "unknown")],
        }
        for dimension, values in breakdown_keys.items():
            for value in values:
                key = str(value)
                counts = condition_totals[dimension].setdefault(key, [0, 0, 0])
                condition_totals[dimension][key] = [
                    previous + current for previous, current in zip(counts, condition_counts, strict=True)
                ]
                condition_window_counts[dimension][key] = condition_window_counts[dimension].get(key, 0) + 1
        box_types = window.get("box_types", [])
        if len(box_types) != len(expected):
            box_types = ["unknown"] * len(expected)
        matched_targets = {target for _, target in matches}
        for target, signal_type in enumerate(box_types):
            counts = signal_type_counts.setdefault(str(signal_type), [0, 0])
            counts[0] += 1
            counts[1] += int(target in matched_targets)
        duration = window["sample_count"] / window["sample_rate"]
        for predicted_index, expected_index in matches:
            prediction = predicted[predicted_index]
            target = expected[expected_index]
            pred_center_x = (prediction[0] + prediction[2]) / 2
            target_center_x = (target[0] + target[2]) / 2
            pred_center_y = (prediction[1] + prediction[3]) / 2
            target_center_y = (target[1] + target[3]) / 2
            frequency_center_errors.append(abs(pred_center_x - target_center_x) * MAX_FREQUENCY_HZ)
            frequency_bound_errors.extend(abs(prediction[[0, 2]] - target[[0, 2]]) * MAX_FREQUENCY_HZ)
            time_center_errors.append(abs(pred_center_y - target_center_y) * duration)
            time_bound_errors.extend(abs(prediction[[1, 3]] - target[[1, 3]]) * duration)

    def mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    for item in manifest:
        if item["split"] == "test":
            duration_seconds += item["sample_count"] / item["sample_rate"]

    results = {}
    for threshold, counts in totals.items():
        true_positive, false_positive, false_negative = counts
        results[f"iou_{threshold}"] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": true_positive / max(1, true_positive + false_positive),
            "recall": true_positive / max(1, true_positive + false_negative),
        }
    condition_breakdowns = {}
    for dimension, groups in condition_totals.items():
        condition_breakdowns[dimension] = {}
        for condition, counts in sorted(groups.items()):
            true_positive, false_positive, false_negative = counts
            condition_breakdowns[dimension][condition] = {
                "reviewed_windows": condition_window_counts[dimension][condition],
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": true_positive / max(1, true_positive + false_positive),
                "recall": true_positive / max(1, true_positive + false_negative),
            }
    signal_type_recall = {
        signal_type: {
            "ground_truth_boxes": counts[0],
            "matched_boxes": counts[1],
            "missed_boxes": counts[0] - counts[1],
            "recall": counts[1] / max(1, counts[0]),
        }
        for signal_type, counts in sorted(signal_type_counts.items())
    }
    inference_seconds = sum(inference_latencies)
    runtime_measurements: dict[str, Any] = {
        "inference_windows": len(inference_latencies),
        "mean_latency_ms_per_window": float(np.mean(inference_latencies) * 1000),
        "p50_latency_ms_per_window": float(np.percentile(inference_latencies, 50) * 1000),
        "p95_latency_ms_per_window": float(np.percentile(inference_latencies, 95) * 1000),
        "throughput_windows_per_second": len(inference_latencies) / max(inference_seconds, 1e-12),
        "accelerator_memory_measured": accelerator_available,
        "peak_accelerator_allocated_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if accelerator_available else None
        ),
        "peak_accelerator_reserved_mb": (
            float(torch.cuda.max_memory_reserved() / (1024 * 1024)) if accelerator_available else None
        ),
    }
    try:
        import psutil
        runtime_measurements["process_rss_mb_after_evaluation"] = float(
            psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        )
    except ImportError:
        runtime_measurements["process_rss_mb_after_evaluation"] = None
    return {
        "confidence_threshold": confidence,
        "reviewed_test_windows": len(image_paths),
        "reviewed_test_audio_seconds": duration_seconds,
        "false_detections_per_audio_hour": totals["0.50"][1] / max(duration_seconds / 3600.0, 1e-12),
        "localization_at_iou_0_50": {
            "matched_boxes": len(frequency_center_errors),
            "frequency_center_error_hz_mae": mean(frequency_center_errors),
            "frequency_bounds_error_hz_mae": mean(frequency_bound_errors),
            "time_center_error_seconds_mae": mean(time_center_errors),
            "time_bounds_error_seconds_mae": mean(time_bound_errors),
            "note": "Errors are calculated only for greedy IoU >= 0.50 matches; null means no boxes matched.",
        },
        "metrics": results,
        "condition_breakdowns_iou_0_50": condition_breakdowns,
        "signal_type_recall_iou_0_50": signal_type_recall,
        "runtime_measurements": runtime_measurements,
        "scope": "one fixed confidence threshold on capture/lineage-held-out, completely reviewed windows",
    }


def train_cv_detector(
    dataset_dir: str | Path,
    *,
    output_model: str | Path,
    base_model: str = "yolo26n.pt",
    epochs: int = 100,
    batch: int = 8,
    seed: int = 1,
    device: str | None = None,
    workers: int = 4,
    confidence: float = 0.25,
) -> dict[str, Any]:
    dataset = Path(dataset_dir).expanduser().resolve()
    destination = Path(output_model).expanduser().resolve()
    report_path = dataset / "export-report.json"
    data_yaml = dataset / "data.yaml"
    if not report_path.is_file() or not data_yaml.is_file():
        raise FileNotFoundError("dataset must be produced by sig-id-export-cv-dataset")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") not in {1, 2}:
        raise ValueError("unsupported CV dataset report schema")
    if report.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("dataset preprocessing version does not match this trainer")
    if report.get("split_images", {}).get("train", 0) < 1 or report.get("split_images", {}).get("validation", 0) < 1:
        raise ValueError("training and validation splits must both contain reviewed images")
    for split in ("train", "validation", "test"):
        if report.get("positive_images_by_split", {}).get(split, 0) < 1:
            raise ValueError(f"{split} split must contain at least one reviewed signal window")
        if report.get("negative_images_by_split", {}).get(split, 0) < 1:
            raise ValueError(f"{split} split must contain at least one completely reviewed no-signal window")
    if not any((dataset / "labels" / "train").glob("*.txt")):
        raise ValueError("training split has no labels")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("CV training requires the optional 'cv' install extra") from exc
    if epochs <= 0 or batch <= 0 or workers < 0 or not 0.0 < confidence < 1.0:
        raise ValueError("epochs and batch must be positive, workers cannot be negative")

    destination.parent.mkdir(parents=True, exist_ok=True)
    run_root = destination.parent / f".{destination.stem}-training"
    model = YOLO(base_model)
    train_options: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": max(IMAGE_HEIGHT, IMAGE_WIDTH),
        # The tall, narrow spectrogram should use its native rectangular batch
        # shape; square letterboxing wastes most of the input tensor.
        "rect": True,
        "batch": batch,
        "seed": seed,
        "deterministic": True,
        "workers": workers,
        "project": str(run_root),
        "name": "run",
        "exist_ok": True,
        "mosaic": 0.0,
        "mixup": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "degrees": 0.0,
        "scale": 0.0,
        "plots": True,
        "verbose": True,
    }
    if device:
        train_options["device"] = device
    model.train(**train_options)
    best = run_root / "run" / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"YOLO training completed without a best.pt artifact at {best}")

    evaluation_model = YOLO(str(best))
    test_metrics = _evaluate_test_set(evaluation_model, dataset, confidence=confidence, device=device)

    with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", dir=destination.parent, delete=False) as temp:
        temporary_model = Path(temp.name)
    shutil.copyfile(best, temporary_model)
    temporary_model.replace(destination)
    metadata = {
        "schema_version": 1,
        "task": "detect",
        "class_agnostic": True,
        "classes": ["signal"],
        "model_file": destination.name,
        "model_sha256": _sha256(destination),
        "base_model": base_model,
        "dataset_path": str(dataset),
        "dataset_report_sha256": _sha256(report_path),
        "dataset_schema_version": report["schema_version"],
        "data_yaml_sha256": _sha256(data_yaml),
        "windows_manifest_sha256": _sha256(dataset / "windows.jsonl"),
        "preprocessing_version": PREPROCESSING_VERSION,
        "image_shape": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
        "model_image_size": list(MODEL_IMAGE_SIZE),
        "rectangular_training": True,
        "epochs": epochs,
        "batch": batch,
        "seed": seed,
        "device": device or "auto",
        "training_finished_utc": datetime.now(timezone.utc).isoformat(),
        "training_run": str(run_root / "run"),
        "held_out_test": test_metrics,
        "model_size_bytes": destination.stat().st_size,
        "scope": "reviewed capture windows from the exported dataset; not a radio or decoder validation result",
    }
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"model": str(destination), "metadata": str(metadata_path), "model_sha256": metadata["model_sha256"],
            "training_run": metadata["training_run"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--base-model", default="yolo26n.pt", help="Ultralytics checkpoint or model YAML")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None, help="Ultralytics device selector, for example cpu or 0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.25, help="fixed confidence for held-out test metrics")
    args = parser.parse_args()
    print(json.dumps(train_cv_detector(args.dataset, output_model=args.output_model,
                                       base_model=args.base_model, epochs=args.epochs,
                                       batch=args.batch, seed=args.seed,
                                       device=args.device, workers=args.workers,
                                       confidence=args.confidence), indent=2))


if __name__ == "__main__":
    main()
