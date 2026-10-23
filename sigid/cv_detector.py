"""Optional YOLO signal-region detection over fixed spectrogram windows.

The base workbench does not depend on Ultralytics. Importing the module is safe;
the optional dependency is loaded only when a configured detector is opened.
"""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from .spectrum import waterfall_samples
from .wav import WavSource


PREPROCESSING_VERSION = "sig-id-cv-gray-v1"
WINDOW_SECONDS = 10
WINDOW_STRIDE_SECONDS = WINDOW_SECONDS // 2
OVERLAP_DEDUP_IOU = 0.30
FFT_SIZE = 4096
HOP_SIZE = 512
MAX_FREQUENCY_HZ = 3000.0
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 1024
MODEL_IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
DB_FLOOR = -120.0
DB_CEILING = -30.0


def spectrogram_image(samples: Any, sample_rate: int) -> np.ndarray:
    """Return the versioned RGB image used for both training and inference.

    Horizontal position is frequency from 0 to 3 kHz; vertical position is
    increasing time. The returned uint8 image has shape (1024, 256, 3).
    """
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size < FFT_SIZE:
        raise ValueError(f"audio window must contain at least {FFT_SIZE} mono samples")
    if sample_rate <= 0 or sample_rate / 2 < MAX_FREQUENCY_HZ:
        raise ValueError("sample rate must provide a Nyquist frequency of at least 3000 Hz")
    spectrum = waterfall_samples(values, sample_rate, fft_size=FFT_SIZE, hop_size=HOP_SIZE,
                                 max_rows=IMAGE_HEIGHT)
    db = np.asarray(spectrum["db"], dtype=np.float32)
    frequencies = np.asarray(spectrum["frequency_hz"], dtype=np.float64)
    target_frequency = np.linspace(0.0, MAX_FREQUENCY_HZ, IMAGE_WIDTH)
    frequency_scaled = np.empty((db.shape[0], IMAGE_WIDTH), dtype=np.float32)
    for row_index, row in enumerate(db):
        frequency_scaled[row_index] = np.interp(target_frequency, frequencies, row)
    source_rows = np.arange(frequency_scaled.shape[0], dtype=np.float64)
    target_rows = np.linspace(0.0, max(0, len(source_rows) - 1), IMAGE_HEIGHT)
    time_scaled = np.empty((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    for column in range(IMAGE_WIDTH):
        time_scaled[:, column] = np.interp(target_rows, source_rows, frequency_scaled[:, column])
    pixels = np.clip((time_scaled - DB_FLOOR) * (255.0 / (DB_CEILING - DB_FLOOR)), 0, 255)
    gray = np.rint(pixels).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def audio_windows(path: str | Path, seconds: int = WINDOW_SECONDS, *,
                  stride_seconds: int | None = None) -> Iterator[tuple[int, np.ndarray, int]]:
    """Yield bounded mono windows, optionally overlapping them by using a shorter stride."""
    if seconds != WINDOW_SECONDS:
        raise ValueError(f"CV window size is fixed at {WINDOW_SECONDS} seconds")
    stride_seconds = seconds if stride_seconds is None else stride_seconds
    if not isinstance(stride_seconds, int) or isinstance(stride_seconds, bool) or not 0 < stride_seconds <= seconds:
        raise ValueError("window stride must be a positive whole number no greater than the window size")
    source = WavSource(path)
    metadata = source.metadata()
    sample_rate = int(metadata["sample_rate"])
    window_samples = sample_rate * seconds
    stride_samples = sample_rate * stride_seconds
    buffered = np.empty(0, dtype=np.float64)
    buffer_start = 0
    for block in source.blocks():
        block_values = np.asarray(block.samples, dtype=np.float64)
        if not buffered.size:
            buffer_start = block.first_sample
        buffered = np.concatenate((buffered, block_values))
        while buffered.size >= window_samples:
            yield buffer_start, buffered[:window_samples].copy(), sample_rate
            buffered = buffered[stride_samples:]
            buffer_start += stride_samples
    while buffered.size >= FFT_SIZE:
        yield buffer_start, buffered[:window_samples].copy(), sample_rate
        buffered = buffered[stride_samples:]
        buffer_start += stride_samples


class YoloSignalDetector:
    """Thin, thread-safe wrapper around an optional Ultralytics YOLO artifact."""

    def __init__(self, model_path: str | Path):
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"CV model file does not exist: {path}")
        metadata_path = path.with_suffix(path.suffix + ".json")
        if not metadata_path.is_file():
            raise ValueError(f"CV model metadata file is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if metadata.get("model_sha256") != model_digest:
            raise ValueError("CV model hash does not match its metadata")
        if metadata.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError("CV model uses an incompatible spectrogram preprocessing version")
        if (
            metadata.get("schema_version") != 1
            or metadata.get("task") != "detect"
            or not metadata.get("class_agnostic")
            or metadata.get("classes") != ["signal"]
        ):
            raise ValueError("CV model metadata must identify a class-agnostic detector")
        confidence_threshold = metadata.get("confidence_threshold", 0.25)
        if not isinstance(confidence_threshold, (int, float)) or isinstance(confidence_threshold, bool) or not 0.0 < confidence_threshold < 1.0:
            raise ValueError("CV model metadata confidence threshold must be between 0 and 1")
        try:
            from ultralytics import YOLO
            import torch
        except ImportError as exc:
            raise RuntimeError("CV inference requires the optional 'cv' install extra") from exc
        self.model_path = str(path.resolve())
        self.model_sha256 = model_digest
        self.model_version = f"sha256:{model_digest}"
        self.confidence_threshold = float(confidence_threshold)
        self.held_out_test = metadata.get("held_out_test")
        self.training_scope = metadata.get("scope")
        if torch.version.hip:
            self.compute_backend = "rocm" if torch.cuda.is_available() else "rocm-unavailable"
        elif torch.cuda.is_available():
            self.compute_backend = "cuda"
        else:
            self.compute_backend = "cpu"
        self._model = YOLO(self.model_path)
        self._lock = Lock()

    def detect(self, image: np.ndarray, *, sample_start: int, sample_count: int,
               sample_rate: int, confidence: float | None = None) -> list[dict[str, Any]]:
        confidence = self.confidence_threshold if confidence is None else confidence
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        with self._lock:
            results = self._model.predict(image, conf=confidence, verbose=False, imgsz=MODEL_IMAGE_SIZE,
                                          rect=True)
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        image_height, image_width = image.shape[:2]
        found = []
        for coordinates, score in zip(xyxy, scores, strict=True):
            x1, y1, x2, y2 = (float(v) for v in coordinates)
            x1, x2 = np.clip([x1, x2], 0, image_width)
            y1, y2 = np.clip([y1, y2], 0, image_height)
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            if right <= left or bottom <= top:
                continue
            low_hz = float(left / image_width * MAX_FREQUENCY_HZ)
            high_hz = float(right / image_width * MAX_FREQUENCY_HZ)
            region_start = sample_start + int(round(top / image_height * sample_count))
            region_end = sample_start + int(round(bottom / image_height * sample_count))
            region_start = min(max(region_start, sample_start), sample_start + sample_count - 1)
            region_end = min(max(region_end, region_start + 1), sample_start + sample_count)
            found.append({
                "sample_start": region_start,
                "sample_count": region_end - region_start,
                "frequency_low_hz": low_hz,
                "frequency_high_hz": high_hz,
                "box_xyxy_normalized": [
                    float(left / image_width), float(top / image_height),
                    float(right / image_width), float(bottom / image_height),
                ],
                "image_width": image_width,
                "image_height": image_height,
                "confidence": float(score),
                "identity": None,
                "model_sha256": self.model_sha256,
                "model_version": self.model_version,
                "preprocessing_version": PREPROCESSING_VERSION,
                "window_sample_start": sample_start,
                "window_sample_count": sample_count,
                "sample_rate": sample_rate,
            })
        return found

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """Extract a normalized second-to-last-layer image embedding."""
        with self._lock:
            extracted = self._model.embed(image, imgsz=MODEL_IMAGE_SIZE, verbose=False, rect=True)
        value = extracted[0] if isinstance(extracted, (list, tuple)) else extracted
        if hasattr(value, "detach"):
            vector = value.detach().cpu().numpy()
        else:
            vector = np.asarray(value)
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("detector returned an invalid image embedding")
        return vector / norm

    def detect_capture(self, path: str | Path, *, confidence: float | None = None) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for start, samples, sample_rate in audio_windows(path, stride_seconds=WINDOW_STRIDE_SECONDS):
            image = spectrogram_image(samples, sample_rate)
            detections.extend(self.detect(image, sample_start=start, sample_count=len(samples),
                                          sample_rate=sample_rate, confidence=confidence))
        return _deduplicate_window_detections(detections)


def _box_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    x0 = max(float(left["frequency_low_hz"]), float(right["frequency_low_hz"]))
    y0 = max(int(left["sample_start"]), int(right["sample_start"]))
    x1 = min(float(left["frequency_high_hz"]), float(right["frequency_high_hz"]))
    y1 = min(int(left["sample_start"]) + int(left["sample_count"]),
             int(right["sample_start"]) + int(right["sample_count"]))
    intersection = max(0.0, x1 - x0) * max(0, y1 - y0)
    left_area = (float(left["frequency_high_hz"]) - float(left["frequency_low_hz"])) * int(left["sample_count"])
    right_area = (float(right["frequency_high_hz"]) - float(right["frequency_low_hz"])) * int(right["sample_count"])
    return intersection / max(1e-12, left_area + right_area - intersection)


def _deduplicate_window_detections(
    detections: list[dict[str, Any]], *, iou_threshold: float = OVERLAP_DEDUP_IOU
) -> list[dict[str, Any]]:
    """Suppress high-overlap proposals repeated by adjacent overlapping windows."""
    ranked = sorted(detections, key=lambda item: float(item["confidence"]), reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in ranked:
        duplicate = next((item for item in kept
                          if item["window_sample_start"] != candidate["window_sample_start"]
                          and _box_iou(item, candidate) >= iou_threshold), None)
        if duplicate is None:
            kept.append(candidate)
        else:
            duplicate.setdefault("supporting_windows", [{
                "sample_start": duplicate["window_sample_start"],
                "sample_count": duplicate["window_sample_count"],
            }])
            duplicate["supporting_windows"].append({
                "sample_start": candidate["window_sample_start"],
                "sample_count": candidate["window_sample_count"],
            })
    for item in kept:
        if "supporting_windows" in item:
            unique = {
                (window["sample_start"], window["sample_count"]): window
                for window in item["supporting_windows"]
            }
            item["supporting_windows"] = list(unique.values())
    return sorted(kept, key=lambda item: (int(item["sample_start"]), float(item["frequency_low_hz"])))
