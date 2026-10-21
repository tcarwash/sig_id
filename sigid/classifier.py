"""Portable NumPy baseline for modulation-method proposals with abstention."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .splits import FAMILY_BY_MODE, METHOD_BY_MODE
from .corpus import MODES


FEATURE_VERSION = "passband-log-spectrum-v1"


def extract_features(samples: Any, sample_rate: int) -> np.ndarray:
    """Create fixed-size, gain-normalized spectral and temporal occupancy features."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size < 1024 or not np.all(np.isfinite(values)):
        raise ValueError("classifier input must be finite mono audio with at least 1024 samples")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate < 8000:
        raise ValueError("classifier sample rate must be an integer of at least 8000 Hz")
    frame_size = 1024
    hop_size = 512
    frames = 1 + (values.size - frame_size) // hop_size
    if frames > 512:
        frame_indices = np.linspace(0, frames - 1, 512).astype(np.int64)
    else:
        frame_indices = np.arange(frames)
    window = np.hanning(frame_size)
    frequencies = np.fft.rfftfreq(frame_size, 1 / sample_rate)
    target_frequencies = np.linspace(200.0, 3800.0, 128)
    rows = []
    for frame_index in frame_indices:
        start = int(frame_index) * hop_size
        spectrum = np.abs(np.fft.rfft(values[start : start + frame_size] * window))
        power = spectrum * spectrum
        total = float(power.sum())
        if total <= np.finfo(np.float64).tiny:
            normalized = np.zeros_like(power)
        else:
            normalized = power / total
        rows.append(np.interp(target_frequencies, frequencies, normalized, left=0.0, right=0.0))
    matrix = np.asarray(rows, dtype=np.float64)
    mean = np.mean(matrix, axis=0)
    variation = np.std(matrix, axis=0)
    mean_log = np.log10(mean + 1e-10)
    variation_log = np.log10(variation + 1e-10)
    centroid = float(np.sum(target_frequencies * mean) / max(float(mean.sum()), 1e-12))
    spread = float(np.sqrt(np.sum(((target_frequencies - centroid) ** 2) * mean) / max(float(mean.sum()), 1e-12)))
    normalized_mean = mean / max(float(mean.sum()), 1e-12)
    entropy = float(-np.sum(normalized_mean * np.log(normalized_mean + 1e-12)))
    temporal_activity = float(np.mean(np.sum(matrix > np.max(matrix, axis=1, keepdims=True) * 0.08, axis=1)))
    summary = np.asarray([
        centroid / 3800.0,
        spread / 3600.0,
        entropy / np.log(128),
        temporal_activity / 128,
    ])
    return np.concatenate((mean_log, variation_log, summary)).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.maximum(exponent.sum(axis=1, keepdims=True), np.finfo(np.float64).tiny)


def fit_softmax(
    features: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    *,
    epochs: int = 1200,
    learning_rate: float = 0.08,
    l2: float = 0.01,
) -> dict[str, np.ndarray]:
    """Fit a small multinomial logistic regression using training rows only."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features.ndim != 2 or labels.shape != (features.shape[0],) or not features.shape[0]:
        raise ValueError("features and labels must be non-empty aligned matrices")
    if set(labels.tolist()) - set(classes):
        raise ValueError("training labels contain a class outside the fixed taxonomy")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x = (features - mean) / scale
    class_index = {name: index for index, name in enumerate(classes)}
    y = np.zeros((len(labels), len(classes)), dtype=np.float64)
    y[np.arange(len(labels)), [class_index[name] for name in labels]] = 1.0
    weights = np.zeros((x.shape[1], len(classes)), dtype=np.float64)
    bias = np.zeros(len(classes), dtype=np.float64)
    for _ in range(epochs):
        probability = _softmax(x @ weights + bias)
        error = (probability - y) / len(x)
        weights -= learning_rate * (x.T @ error + l2 * weights)
        bias -= learning_rate * error.sum(axis=0)
    return {
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "weights": weights.astype(np.float32),
        "bias": bias.astype(np.float32),
    }


def predict_probabilities(features: np.ndarray, parameters: dict[str, np.ndarray]) -> np.ndarray:
    matrix = np.atleast_2d(np.asarray(features, dtype=np.float64))
    if matrix.shape[1] != len(parameters["mean"]):
        raise ValueError("feature dimension does not match the trained model")
    normalized = (matrix - parameters["mean"]) / parameters["scale"]
    return _softmax(normalized @ parameters["weights"] + parameters["bias"])


def propose(
    features: np.ndarray,
    parameters: dict[str, np.ndarray],
    classes: list[str],
    *,
    threshold: float,
    family_threshold: float | None = None,
) -> dict[str, Any]:
    probabilities = predict_probabilities(features, parameters)[0]
    top = int(np.argmax(probabilities))
    confidence = float(probabilities[top])
    accepted = confidence >= threshold
    family_names = ("FSK", "MFSK", "PSK")
    family_probabilities = np.asarray([
        sum(probabilities[index] for index, method in enumerate(classes) if FAMILY_BY_MODE[method] == family)
        for family in family_names
    ])
    family_index = int(np.argmax(family_probabilities))
    family_confidence = float(family_probabilities[family_index])
    chosen_family_threshold = threshold if family_threshold is None else family_threshold
    family_accepted = family_confidence >= chosen_family_threshold
    method_family = FAMILY_BY_MODE[classes[top]]
    if accepted:
        family_result = method_family
        family_result_confidence = float(family_probabilities[family_names.index(method_family)])
        family_result_abstained = False
    else:
        family_result = family_names[family_index] if family_accepted else None
        family_result_confidence = family_confidence
        family_result_abstained = not family_accepted
    return {
        "method": METHOD_BY_MODE[classes[top]] if accepted else None,
        "method_id": classes[top] if accepted else None,
        "family": family_result,
        "family_confidence": family_result_confidence,
        "family_threshold": chosen_family_threshold,
        "family_abstained": family_result_abstained,
        "protocol": None,
        "variant": None,
        "top_candidate": classes[top],
        "confidence": confidence,
        "threshold": threshold,
        "abstained": not accepted,
        "evidence": "synthetic-trained modulation-method model; proposal only",
    }


def load_model(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported model schema version")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("model_sha256"):
        raise ValueError("model weight file does not match its metadata hash")
    with np.load(path, allow_pickle=False) as archive:
        parameters = {name: archive[name].copy() for name in ("mean", "scale", "weights", "bias")}
    if metadata.get("feature_version") != FEATURE_VERSION:
        raise ValueError("unsupported model feature version")
    if metadata.get("classes") != list(MODES):
        raise ValueError("model taxonomy does not match the supported modulation methods")
    threshold = metadata.get("abstention_threshold")
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1.000001:
        raise ValueError("model abstention threshold is invalid")
    family_threshold = metadata.get("family_abstention_threshold")
    if not isinstance(family_threshold, (int, float)) or not np.isfinite(family_threshold) or not 0 <= family_threshold <= 1.000001:
        raise ValueError("model family abstention threshold is invalid")
    expected = metadata.get("feature_count")
    if any(parameters[name].dtype.kind not in "fiu" for name in parameters):
        raise ValueError("model arrays must be numeric")
    if parameters["mean"].shape != (expected,) or parameters["scale"].shape != (expected,):
        raise ValueError("model feature arrays do not match metadata")
    if parameters["weights"].shape != (expected, len(metadata["classes"])) or parameters["bias"].shape != (len(metadata["classes"]),):
        raise ValueError("model classifier arrays do not match taxonomy")
    return metadata, parameters
