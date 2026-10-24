"""Build a compact local nearest-neighbor index from reviewed signal regions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cv_detector import FFT_SIZE, MAX_FREQUENCY_HZ, WINDOW_SECONDS, YoloSignalDetector, spectrogram_image
from .dataset import CaptureStore
from .wav import WavSource


EMBEDDING_INDEX_SCHEMA_VERSION = 2
EMBEDDING_PREPROCESSING_VERSION = "sig-id-cv-region-embedding-v2"


def read_audio_range(path: str | Path, start: int, count: int) -> tuple[int, np.ndarray]:
    if start < 0 or count <= 0:
        raise ValueError("audio range must have a non-negative start and positive count")
    values: list[float] = []
    sample_rate: int | None = None
    end = start + count
    for block in WavSource(path).blocks():
        sample_rate = block.sample_rate
        block_end = block.first_sample + block.sample_count
        if block_end <= start:
            continue
        if block.first_sample >= end:
            break
        left = max(0, start - block.first_sample)
        right = min(block.sample_count, end - block.first_sample)
        values.extend(block.samples[left:right])
    if sample_rate is None or len(values) != count:
        raise ValueError("requested audio range extends outside the capture")
    return sample_rate, np.asarray(values, dtype=np.float64)


def region_image(samples: Any, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    if not (0 <= low_hz < high_hz <= MAX_FREQUENCY_HZ):
        raise ValueError("embedding region frequency bounds must be inside 0–3000 Hz")
    whole = spectrogram_image(samples, sample_rate)
    left = max(0, min(whole.shape[1] - 1, int(np.floor(low_hz / MAX_FREQUENCY_HZ * whole.shape[1]))))
    right = max(left + 1, min(whole.shape[1], int(np.ceil(high_hz / MAX_FREQUENCY_HZ * whole.shape[1]))))
    return whole[:, left:right, :].copy()


def build_embedding_index(
    data_dir: str | Path,
    *,
    model_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    store = CaptureStore(Path(data_dir).expanduser() / "dataset.sqlite3")
    detector = YoloSignalDetector(model_path)
    embeddings: list[np.ndarray] = []
    entries: list[dict[str, Any]] = []
    for capture in store.list_captures():
        for region in store.regions(capture["id"]):
            if region["lineage_state"] != "active" or region["review_state"] != "accepted":
                continue
            evidence = region.get("evidence") or {}
            if evidence.get("generator") == "sig-id-noise" or evidence.get("noise_only") is True:
                continue
            low, high = region["frequency_low_hz"], region["frequency_high_hz"]
            if low is None or high is None:
                continue
            low, high = max(0.0, float(low)), min(MAX_FREQUENCY_HZ, float(high))
            if low >= high:
                continue
            sample_rate = int(capture["sample_rate"])
            segment_samples = sample_rate * WINDOW_SECONDS
            region_start = int(region["sample_start"])
            region_end = region_start + int(region["sample_count"])
            segment_index = 0
            for start in range(region_start, region_end, segment_samples):
                count = min(segment_samples, region_end - start)
                if count < FFT_SIZE:
                    continue
                rate, samples = read_audio_range(capture["audio_path"], start, count)
                image = region_image(samples, rate, low, high)
                embedding = detector.embed_image(image)
                embeddings.append(embedding)
                entries.append({
                    "capture_id": capture["id"],
                    "capture_sha256": capture["sha256"],
                    "region_id": region["id"],
                    "label_revision": int(region["revision"]),
                    "geometry_revision": int(region["geometry_revision"]),
                    "segment_index": segment_index,
                    "sample_start": start,
                    "sample_count": count,
                    "sample_rate": rate,
                    "frequency_low_hz": float(low),
                    "frequency_high_hz": float(high),
                    "family": region["family"],
                    "method": region["method"],
                    "protocol": region["protocol"],
                    "variant": region["variant"],
                    "review_state": region["review_state"],
                })
                segment_index += 1
    if not embeddings:
        raise ValueError("no accepted, bounded signal regions are available for embedding retrieval")
    vectors = np.asarray(embeddings, dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": EMBEDDING_INDEX_SCHEMA_VERSION,
        "model_sha256": detector.model_sha256,
        "preprocessing_version": EMBEDDING_PREPROCESSING_VERSION,
        "count": len(entries),
        "embedding_dimensions": int(vectors.shape[1]),
        "entries": entries,
    }
    np.savez_compressed(output_path, vectors=vectors, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    return {
        "index": str(output_path),
        "index_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "model_sha256": detector.model_sha256,
        "embedding_count": len(entries),
        "embedding_dimensions": int(vectors.shape[1]),
    }


def load_embedding_index(path: str | Path, *, model_sha256: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    with np.load(Path(path).expanduser(), allow_pickle=False) as stored:
        vectors = np.asarray(stored["vectors"], dtype=np.float32)
        metadata = json.loads(str(stored["metadata"].item()))
    if metadata.get("schema_version") != EMBEDDING_INDEX_SCHEMA_VERSION or metadata.get("preprocessing_version") != EMBEDDING_PREPROCESSING_VERSION:
        raise ValueError("unsupported CV embedding index format")
    if metadata.get("model_sha256") != model_sha256:
        raise ValueError("CV embedding index was built with a different detector model")
    entries = metadata.get("entries")
    if not isinstance(entries, list) or len(entries) != len(vectors) or vectors.ndim != 2:
        raise ValueError("CV embedding index entries and vectors are inconsistent")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("CV embedding index contains non-finite values")
    if any(
        not isinstance(entry, dict)
        or not isinstance(entry.get("label_revision"), int)
        or not isinstance(entry.get("geometry_revision"), int)
        for entry in entries
    ):
        raise ValueError("CV embedding index lacks region revision metadata; rebuild it from current reviewed regions")
    return vectors, entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("~/.local/share/sig-id").expanduser())
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_embedding_index(args.data_dir, model_path=args.model, output=args.output), indent=2))


if __name__ == "__main__":
    main()
