"""Portable NumPy baseline for modulation-method proposals with abstention."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .splits import FAMILY_BY_MODE, METHOD_BY_MODE, families_for_modes, register_mode_definitions
from .corpus import MODES


LEGACY_FEATURE_VERSION = "passband-log-spectrum-v1"
FEATURE_VERSION = "passband-log-spectrum-envelope-v2"
ROI_FEATURE_VERSION = "passband-log-spectrum-envelope-roi-v3"
FIXED_ROI_FEATURE_VERSION = "passband-log-spectrum-envelope-roi-fixed-window-v4"
LEGACY_FIXED_2S_ROI_FEATURE_VERSION = "passband-log-spectrum-envelope-roi-fixed-2s-v4"
DEFAULT_FEATURE_VERSION = FIXED_ROI_FEATURE_VERSION
CLASSIFIER_WINDOW_SECONDS = 2.0
ROI_FEATURE_VERSIONS = {
    ROI_FEATURE_VERSION, FIXED_ROI_FEATURE_VERSION,
    LEGACY_FIXED_2S_ROI_FEATURE_VERSION,
}
FIXED_WINDOW_FEATURE_VERSIONS = {
    FIXED_ROI_FEATURE_VERSION, LEGACY_FIXED_2S_ROI_FEATURE_VERSION,
}
NOVELTY_FEATURE_VERSIONS = {FEATURE_VERSION, *ROI_FEATURE_VERSIONS}
FEATURE_COUNT_BY_VERSION = {
    LEGACY_FEATURE_VERSION: 260,
    FEATURE_VERSION: 266,
    ROI_FEATURE_VERSION: 266,
    FIXED_ROI_FEATURE_VERSION: 266,
    LEGACY_FIXED_2S_ROI_FEATURE_VERSION: 266,
}
NOISE_CLASS = "noise"


def fixed_duration_crop(samples: Any, sample_rate: int, *, duration_seconds: float = CLASSIFIER_WINDOW_SECONDS) -> np.ndarray:
    """Center-crop or center-pad audio to a stable duration for classifier features."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("fixed-duration classifier crop requires finite mono audio")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate < 1:
        raise ValueError("fixed-duration classifier crop needs a positive integer sample rate")
    if not np.isfinite(duration_seconds) or not 0.25 <= duration_seconds <= 10.0:
        raise ValueError("classifier crop duration must be between 0.25 and 10 seconds")
    target_count = max(1, round(sample_rate * duration_seconds))
    if values.size >= target_count:
        start = (values.size - target_count) // 2
        return values[start : start + target_count]
    output = np.zeros(target_count, dtype=np.float64)
    start = (target_count - values.size) // 2
    output[start : start + values.size] = values
    return output


def extract_features(
    samples: Any, sample_rate: int, *, feature_version: str = FEATURE_VERSION,
    frequency_range_hz: tuple[float, float] | None = None,
    classifier_window_seconds: float = CLASSIFIER_WINDOW_SECONDS,
) -> np.ndarray:
    """Create fixed-size, gain-normalized spectral and optional envelope features."""
    if feature_version not in FEATURE_COUNT_BY_VERSION:
        raise ValueError(f"unsupported model feature version: {feature_version}")
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size < 1024 or not np.all(np.isfinite(values)):
        raise ValueError("classifier input must be finite mono audio with at least 1024 samples")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate < 8000:
        raise ValueError("classifier sample rate must be an integer of at least 8000 Hz")
    if feature_version in FIXED_WINDOW_FEATURE_VERSIONS:
        values = fixed_duration_crop(values, sample_rate, duration_seconds=classifier_window_seconds)
    if feature_version in ROI_FEATURE_VERSIONS:
        nyquist = sample_rate / 2.0
        low, high = frequency_range_hz or (200.0, min(3800.0, nyquist - 20.0))
        if not np.isfinite(low) or not np.isfinite(high) or not 0 <= low < high <= min(nyquist, 3800.0):
            raise ValueError("ROI classifier frequency bounds must be finite, ordered and inside 0–3800 Hz")
        margin = max(100.0, (high - low) * 0.25)
        passband_low = max(20.0, low - margin)
        passband_high = min(nyquist - 20.0, high + margin)
        if passband_low >= passband_high:
            raise ValueError("ROI classifier passband is too narrow for this sample rate")
        from scipy.signal import butter, sosfiltfilt

        sections = butter(4, (passband_low, passband_high), btype="bandpass", fs=sample_rate, output="sos")
        try:
            values = sosfiltfilt(sections, values)
        except ValueError as exc:
            raise ValueError(f"ROI classifier filter could not process this audio range: {exc}") from exc
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
    result = np.concatenate((mean_log, variation_log, summary))
    if feature_version in NOVELTY_FEATURE_VERSIONS:
        from scipy.signal import hilbert

        envelope = np.abs(hilbert(values))
        block_size = max(8, round(sample_rate / 1000.0))
        usable = (envelope.size // block_size) * block_size
        blocks = envelope[:usable].reshape(-1, block_size)
        block_rms = np.sqrt(np.mean(blocks * blocks, axis=1))
        normalized = block_rms / max(float(np.mean(block_rms)), 1e-12)
        centered = normalized - float(np.mean(normalized))
        denominator = float(np.dot(centered, centered))
        lag_one_correlation = (
            float(np.dot(centered[:-1], centered[1:]) / denominator)
            if normalized.size > 1 and denominator > 1e-12 else 0.0
        )
        envelope_features = np.asarray([
            np.quantile(normalized, 0.1),
            np.quantile(normalized, 0.5),
            np.quantile(normalized, 0.9),
            float(np.std(normalized)),
            float(np.mean(normalized > 0.25)),
            lag_one_correlation,
        ])
        result = np.concatenate((result, envelope_features))
    return result.astype(np.float32)


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
    initial_parameters: dict[str, np.ndarray] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Fit a small multinomial logistic regression using training rows only."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features.ndim != 2 or labels.shape != (features.shape[0],) or not features.shape[0]:
        raise ValueError("features and labels must be non-empty aligned matrices")
    if set(labels.tolist()) - set(classes):
        raise ValueError("training labels contain a class outside the fixed taxonomy")
    if initial_parameters is None:
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-6] = 1.0
    else:
        mean = np.asarray(initial_parameters["mean"], dtype=np.float64).copy()
        scale = np.asarray(initial_parameters["scale"], dtype=np.float64).copy()
        if mean.shape != (features.shape[1],) or scale.shape != mean.shape:
            raise ValueError("initial classifier normalization does not match feature dimensions")
    x = (features - mean) / scale
    class_index = {name: index for index, name in enumerate(classes)}
    y = np.zeros((len(labels), len(classes)), dtype=np.float64)
    y[np.arange(len(labels)), [class_index[name] for name in labels]] = 1.0
    weights = np.zeros((x.shape[1], len(classes)), dtype=np.float64)
    bias = np.zeros(len(classes), dtype=np.float64)
    if initial_parameters is not None:
        old_weights = np.asarray(initial_parameters["weights"], dtype=np.float64)
        old_bias = np.asarray(initial_parameters["bias"], dtype=np.float64)
        if old_weights.shape[0] != x.shape[1]:
            raise ValueError("initial classifier weights do not match feature dimensions")
        for old_index, name in enumerate(initial_parameters.get("classes", [])):
            if name in class_index and old_index < old_weights.shape[1] and old_index < old_bias.size:
                weights[:, class_index[name]] = old_weights[:, old_index]
                bias[class_index[name]] = old_bias[old_index]
    report_every = max(1, epochs // 100)
    for epoch in range(epochs):
        probability = _softmax(x @ weights + bias)
        error = (probability - y) / len(x)
        weights -= learning_rate * (x.T @ error + l2 * weights)
        bias -= learning_rate * error.sum(axis=0)
        if progress_callback and ((epoch + 1) % report_every == 0 or epoch + 1 == epochs):
            progress_callback(epoch + 1, epochs)
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


def support_distances(features: np.ndarray, parameters: dict[str, np.ndarray]) -> np.ndarray:
    """Return RMS distance to the nearest trained modulation prototype."""
    centers = parameters.get("support_centers")
    if centers is None:
        raise ValueError("model does not contain novelty support prototypes")
    matrix = np.atleast_2d(np.asarray(features, dtype=np.float64))
    if matrix.shape[1] != len(parameters["mean"]):
        raise ValueError("feature dimension does not match the trained model")
    normalized = (matrix - parameters["mean"]) / parameters["scale"]
    delta = normalized[:, None, :] - centers[None, :, :]
    return np.sqrt(np.min(np.mean(delta * delta, axis=2), axis=1))


def propose(
    features: np.ndarray,
    parameters: dict[str, np.ndarray],
    classes: list[str],
    *,
    threshold: float,
    family_threshold: float | None = None,
    novelty_threshold: float | None = None,
) -> dict[str, Any]:
    probabilities = predict_probabilities(features, parameters)[0]
    top = int(np.argmax(probabilities))
    confidence = float(probabilities[top])
    noise_detected = classes[top] == NOISE_CLASS
    novelty_distance = (
        float(support_distances(features, parameters)[0])
        if "support_centers" in parameters else None
    )
    within_support = novelty_threshold is None or (
        novelty_distance is not None and novelty_distance <= novelty_threshold
    )
    accepted = confidence >= threshold and not noise_detected and within_support
    family_names = families_for_modes(classes)
    family_probabilities = np.asarray([
        sum(probabilities[index] for index, method in enumerate(classes) if method in FAMILY_BY_MODE and FAMILY_BY_MODE[method] == family)
        for family in family_names
    ])
    family_index = int(np.argmax(family_probabilities))
    family_confidence = float(family_probabilities[family_index])
    chosen_family_threshold = threshold if family_threshold is None else family_threshold
    family_accepted = family_confidence >= chosen_family_threshold and within_support
    method_family = FAMILY_BY_MODE.get(classes[top])
    if accepted:
        family_result = method_family
        family_result_confidence = float(family_probabilities[family_names.index(method_family)])
        family_result_abstained = False
    elif noise_detected:
        family_result = None
        family_result_confidence = 0.0
        family_result_abstained = True
    else:
        family_result = family_names[family_index] if family_accepted else None
        family_result_confidence = family_confidence
        family_result_abstained = not family_accepted
    return {
        "method": METHOD_BY_MODE.get(classes[top]) if accepted else None,
        "method_id": classes[top] if accepted else None,
        "family": family_result,
        "family_confidence": family_result_confidence,
        "family_threshold": chosen_family_threshold,
        "family_abstained": family_result_abstained,
        "novelty_distance": novelty_distance,
        "novelty_threshold": novelty_threshold,
        "novelty_rejected": bool(novelty_threshold is not None and not within_support),
        "protocol": None,
        "variant": None,
        "top_candidate": classes[top],
        "noise_detected": noise_detected,
        "confidence": confidence,
        "threshold": threshold,
        "abstained": not accepted,
        "evidence": "synthetic-trained modulation-method model; proposal only",
    }


def load_model(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = Path(path)
    if path.suffix == ".json":
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("model_type") != "hierarchical_family_experts" or metadata.get("schema_version") != 1:
            raise ValueError("unsupported classifier bundle")
        register_mode_definitions([
            {"id": mode, "family": family, "method": metadata.get("method_names", {}).get(mode)}
            for mode, family in metadata.get("family_by_method", {}).items()
        ])
        root = path.parent
        def read_component(spec: dict[str, Any]) -> dict[str, np.ndarray]:
            weights_path = root / spec["file"]
            if hashlib.sha256(weights_path.read_bytes()).hexdigest() != spec.get("sha256"):
                raise ValueError("hierarchical classifier component hash mismatch")
            with np.load(weights_path, allow_pickle=False) as archive:
                return {name: archive[name].copy() for name in ("mean", "scale", "weights", "bias")}
        components = {"family": read_component(metadata["family_model"]), "experts": {}}
        for family, spec in metadata["experts"].items():
            components["experts"][family] = read_component(spec)
        feature_version = metadata.get("feature_version")
        if feature_version not in FEATURE_COUNT_BY_VERSION or metadata.get("feature_count") != FEATURE_COUNT_BY_VERSION[feature_version]:
            raise ValueError("hierarchical classifier feature metadata is invalid")
        component_classes = {"family": metadata.get("families", [])}
        component_classes.update(metadata.get("mode_classes_by_family", {}))
        for key, params in [("family", components["family"]), *[(family, values) for family, values in components["experts"].items()]]:
            classes = component_classes[key]
            if (not isinstance(classes, list) or not classes
                    or params["mean"].shape != (metadata["feature_count"],)
                    or params["scale"].shape != (metadata["feature_count"],)
                    or params["weights"].shape != (metadata["feature_count"], len(classes))
                    or params["bias"].shape != (len(classes),)
                    or any(array.dtype.kind not in "fiu" for array in params.values())):
                raise ValueError("hierarchical classifier arrays do not match their declared classes")
        return metadata, components
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported model schema version")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("model_sha256"):
        raise ValueError("model weight file does not match its metadata hash")
    feature_version = metadata.get("feature_version")
    if feature_version not in FEATURE_COUNT_BY_VERSION:
        raise ValueError("unsupported model feature version")
    if feature_version in FIXED_WINDOW_FEATURE_VERSIONS:
        duration = metadata.get("classifier_window_seconds")
        if not isinstance(duration, (int, float)) or not np.isfinite(duration) or not 0.25 <= duration <= 10.0:
            raise ValueError("fixed-window model metadata has an invalid classifier window duration")
    with np.load(path, allow_pickle=False) as archive:
        parameters = {name: archive[name].copy() for name in ("mean", "scale", "weights", "bias")}
        if feature_version in NOVELTY_FEATURE_VERSIONS:
            if "support_centers" not in archive.files:
                raise ValueError("model lacks modulation novelty support prototypes")
            parameters["support_centers"] = archive["support_centers"].copy()
    classes = metadata.get("classes")
    family_by_method = metadata.get("family_by_method", {})
    method_names = metadata.get("method_names", {})
    if isinstance(family_by_method, dict) and isinstance(method_names, dict):
        register_mode_definitions([
            {"id": mode, "family": family, "method": method_names.get(mode)}
            for mode, family in family_by_method.items()
        ])
    if (
        not isinstance(classes, list) or len(classes) < 1
        or any(not isinstance(item, str) for item in classes)
        or len(set(classes)) != len(classes)
        or any(item not in METHOD_BY_MODE and item != NOISE_CLASS for item in classes)
        or NOISE_CLASS in classes[:-1]
    ):
        raise ValueError("model taxonomy contains duplicate or unsupported classifier classes")
    threshold = metadata.get("abstention_threshold")
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1.000001:
        raise ValueError("model abstention threshold is invalid")
    family_threshold = metadata.get("family_abstention_threshold")
    if not isinstance(family_threshold, (int, float)) or not np.isfinite(family_threshold) or not 0 <= family_threshold <= 1.000001:
        raise ValueError("model family abstention threshold is invalid")
    expected = metadata.get("feature_count")
    if expected != FEATURE_COUNT_BY_VERSION[feature_version]:
        raise ValueError("model feature count does not match its feature version")
    if any(parameters[name].dtype.kind not in "fiu" for name in parameters):
        raise ValueError("model arrays must be numeric")
    if parameters["mean"].shape != (expected,) or parameters["scale"].shape != (expected,):
        raise ValueError("model feature arrays do not match metadata")
    if parameters["weights"].shape != (expected, len(classes)) or parameters["bias"].shape != (len(classes),):
        raise ValueError("model classifier arrays do not match taxonomy")
    if feature_version in NOVELTY_FEATURE_VERSIONS:
        novelty_threshold = metadata.get("novelty_distance_threshold")
        if not isinstance(novelty_threshold, (int, float)) or not np.isfinite(novelty_threshold):
            raise ValueError("model novelty threshold is invalid")
        known_classes = [item for item in classes if item != NOISE_CLASS]
        if parameters["support_centers"].shape != (len(known_classes), expected):
            raise ValueError("model novelty support prototypes do not match the known modulation taxonomy")
    return metadata, parameters


def propose_loaded(features: np.ndarray, metadata: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    """Score with either the legacy flat model or the family/expert bundle."""
    if metadata.get("model_type") != "hierarchical_family_experts":
        return propose(
            features, parameters, metadata["classes"],
            threshold=float(metadata["abstention_threshold"]),
            family_threshold=float(metadata["family_abstention_threshold"]),
            novelty_threshold=metadata.get("novelty_distance_threshold"),
        )
    family_names = metadata["families"]
    family_probs = predict_probabilities(features, parameters["family"])[0]
    family_idx = int(np.argmax(family_probs))
    family = family_names[family_idx]
    family_conf = float(family_probs[family_idx])
    family_threshold = float(metadata["family_thresholds"][family])
    expert_spec = metadata["mode_classes_by_family"].get(family, [])
    expert_probs = predict_probabilities(features, parameters["experts"][family])[0]
    top = int(np.argmax(expert_probs))
    mode_id = expert_spec[top]
    confidence = float(expert_probs[top])
    threshold = float(metadata["mode_thresholds"][family])
    family_accepted = family_conf >= family_threshold
    mode_accepted = confidence >= threshold
    accepted = family_accepted and mode_accepted
    mode_details = metadata.get("mode_details", {}).get(mode_id, {})
    if not mode_details:
        from .modes import MODE_REGISTRY
        mode_details = MODE_REGISTRY.get(mode_id, {})
    mode_name = mode_details.get("display_name", mode_id)
    protocol = mode_details.get("protocol")
    variant = mode_details.get("variant")
    candidate_path = f"{family} → {mode_name}"
    return {"method": METHOD_BY_MODE.get(mode_id) if accepted else None,
            "method_id": mode_id if accepted else None,
            "mode_name": mode_name if accepted else None,
            "candidate_mode_id": mode_id,
            "candidate_mode_name": mode_name,
            "candidate_protocol": protocol,
            "candidate_variant": variant,
            "family": family if family_accepted else None,
            "candidate_family": family,
            "family_confidence": family_conf, "family_threshold": family_threshold,
            "family_abstained": not family_accepted,
            "protocol": protocol if accepted else None,
            "variant": variant if accepted else None,
            "top_candidate": candidate_path, "noise_detected": False, "confidence": confidence,
            "threshold": threshold, "abstained": not accepted,
            "evidence": "hierarchical classifier proposal; low certainty"}
