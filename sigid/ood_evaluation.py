"""Evaluate model abstention on verified synthetic modulation families it did not train on."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .classifier import CLASSIFIER_WINDOW_SECONDS, ROI_FEATURE_VERSIONS, extract_features, load_model, propose
from .splits import METHOD_BY_MODE
from .synthetic_dataset import _validate_manifest, synthetic_frequency_bounds
from .wav import WavSource


def evaluate_unmodeled_wavs(model_path: str | Path, waveforms: list[str | Path]) -> dict[str, Any]:
    """Score verified synthetic signals whose modulation method is outside model taxonomy.

    This is a measurement tool, not a claim of general unknown-mode detection. The
    supplied waveforms must be held out of model fitting and threshold calibration.
    """
    if not waveforms:
        raise ValueError("at least one held-out synthetic waveform is required")
    model_path = Path(model_path).expanduser().resolve(strict=True)
    metadata, parameters = load_model(model_path)
    classes = metadata["classes"]
    known_methods = set(METHOD_BY_MODE.values())
    examples = []
    for item in waveforms:
        wav_path = Path(item).expanduser().resolve(strict=True)
        sidecar = wav_path.with_suffix(wav_path.suffix + ".json")
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        _validate_manifest(manifest, wav_path)
        occupied_band = synthetic_frequency_bounds(manifest)
        if metadata["feature_version"] in ROI_FEATURE_VERSIONS and occupied_band is None:
            raise ValueError(f"ROI feature evaluation requires verified frequency bounds for {wav_path}")
        labels = manifest["labels"]
        method = labels.get("modulation_method")
        family = labels.get("modulation_family")
        if method in known_methods or family in {"FSK", "MFSK", "PSK"}:
            raise ValueError(f"{wav_path} is labeled as a method or family already represented by the model")
        audio = WavSource(wav_path)
        values = np.asarray(
            [sample for block in audio.blocks() for sample in block.samples], dtype=np.float64
        )
        feature = extract_features(
            values,
            int(manifest["sample_rate"]),
            feature_version=metadata["feature_version"],
            frequency_range_hz=occupied_band if metadata["feature_version"] in ROI_FEATURE_VERSIONS else None,
            classifier_window_seconds=float(metadata.get("classifier_window_seconds") or CLASSIFIER_WINDOW_SECONDS),
        )
        prediction = propose(
            feature,
            parameters,
            classes,
            threshold=float(metadata["abstention_threshold"]),
            family_threshold=float(metadata["family_abstention_threshold"]),
            novelty_threshold=metadata.get("novelty_distance_threshold"),
        )
        digest = hashlib.sha256(wav_path.read_bytes()).hexdigest()
        examples.append({
            "waveform": wav_path.name,
            "waveform_sha256": digest,
            "generator": manifest["generator"],
            "true_family": family,
            "true_method": method,
            "predicted_method": prediction["method_id"],
            "top_candidate": prediction["top_candidate"],
            "method_confidence": prediction["confidence"],
            "method_abstained": prediction["abstained"],
            "predicted_family": prediction["family"],
            "family_confidence": prediction["family_confidence"],
            "family_abstained": prediction["family_abstained"],
        })
    count = len(examples)
    method_accepted = sum(not example["method_abstained"] for example in examples)
    family_accepted = sum(not example["family_abstained"] for example in examples)
    return {
        "schema_version": 1,
        "evaluation_kind": "held-out synthetic unmodeled modulation",
        "scope": "only the supplied verified synthetic waveforms; no field or comprehensive unknown-family claim",
        "model_sha256": metadata["model_sha256"],
        "abstention_threshold": metadata["abstention_threshold"],
        "family_abstention_threshold": metadata["family_abstention_threshold"],
        "examples": count,
        "method_false_accepts": method_accepted,
        "method_false_accept_rate": method_accepted / count,
        "family_false_accepts": family_accepted,
        "family_false_accept_rate": family_accepted / count,
        "results": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure abstention on held-out synthetic unknown modulations")
    parser.add_argument("--model", type=Path, required=True, help="trained .npz model")
    parser.add_argument("waveforms", nargs="+", type=Path, help="WAVs with verified synthetic sidecar manifests")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    try:
        result = evaluate_unmodeled_wavs(args.model, args.waveforms)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
