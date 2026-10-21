"""Deterministic, lineage-preserving impairments for synthetic WAV sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import wave

import numpy as np

from .wav import WavSource


def _read_mono_pcm16(path: str | Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav:
        if wav.getcomptype() != "NONE" or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("augmentation currently requires uncompressed mono 16-bit PCM WAV")
        return wav.getframerate(), np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy()


def _write_mono_pcm16(path: str | Path, sample_rate: int, pcm: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(pcm, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(values.tobytes())


def _awgn_pcm(parent_pcm: np.ndarray, *, snr_db: float, gain: float, seed: int) -> tuple[np.ndarray, dict[str, float | int]]:
    if not np.isfinite(snr_db) or not -10 <= snr_db <= 80:
        raise ValueError("snr_db must be finite and between -10 and 80 dB")
    if not np.isfinite(gain) or not 0.01 <= gain <= 4.0:
        raise ValueError("gain must be finite and between 0.01 and 4")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    clean = parent_pcm.astype(np.float64) / 32768.0
    gained = clean * gain
    signal_power = float(np.mean(gained * gained))
    if signal_power <= 0:
        raise ValueError("parent waveform must have non-zero signal power")
    noise_std = math.sqrt(signal_power / (10.0 ** (snr_db / 10.0)))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, size=gained.size)
    unbounded = gained + noise
    clipped = np.clip(unbounded, -1.0, 32767 / 32768)
    pcm = np.rint(clipped * 32768).astype("<i2")
    measured_noise = pcm.astype(np.float64) / 32768.0 - gained
    measured_power = float(np.mean(measured_noise * measured_noise))
    measured_snr = 10.0 * math.log10(signal_power / max(measured_power, np.finfo(float).tiny))
    return pcm, {
        "noise_std": noise_std,
        "measured_snr_db": measured_snr,
        "clipped_sample_count": int(np.count_nonzero(unbounded != clipped)),
    }


def apply_awgn_augmentation(
    parent_wav: str | Path,
    output_wav: str | Path,
    *,
    snr_db: float,
    gain: float = 1.0,
    seed: int,
) -> dict[str, float | int]:
    sample_rate, parent_pcm = _read_mono_pcm16(parent_wav)
    pcm, stats = _awgn_pcm(parent_pcm, snr_db=snr_db, gain=gain, seed=seed)
    _write_mono_pcm16(output_wav, sample_rate, pcm)
    return stats


def generate_awgn_augmentation(
    parent_wav: str | Path,
    parent_manifest_path: str | Path,
    output_wav: str | Path,
    *,
    snr_db: float,
    gain: float = 1.0,
    seed: int,
) -> dict[str, object]:
    """Create an exact, replayable AWGN child of an imported clean source."""
    parent_wav = Path(parent_wav).resolve(strict=True)
    parent_manifest_path = Path(parent_manifest_path).resolve(strict=True)
    output_wav = Path(output_wav)
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(parent_manifest, dict) or parent_manifest.get("source_kind") != "synthetic":
        raise ValueError("parent must have a synthetic generator manifest")

    # Use the same strict waveform/manifest verifier as ordinary ground-truth import.
    from .synthetic_dataset import _validate_manifest

    _validate_manifest(parent_manifest, parent_wav)
    parent_rate, parent_pcm = _read_mono_pcm16(parent_wav)
    pcm, stats = _awgn_pcm(parent_pcm, snr_db=snr_db, gain=gain, seed=seed)
    _write_mono_pcm16(output_wav, parent_rate, pcm)
    audio = WavSource(output_wav).metadata()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source_kind": "synthetic",
        "generator": "sig-id-awgn-augmentation",
        "generator_version": "0.1.0",
        "deterministic": True,
        "waveform": output_wav.name,
        "sample_rate": int(audio["sample_rate"]),
        "sample_count": int(audio["sample_count"]),
        "sample_width_bytes": int(audio["sample_width_bytes"]),
        "channels": int(audio["channels"]),
        "labels": parent_manifest["labels"],
        "parameters": {
            "noise_model": "additive white Gaussian noise",
            "requested_snr_db": snr_db,
            "gain": gain,
            "seed": seed,
            "snr_convention": "whole-waveform mean-square signal power divided by generated noise power",
            "transform_order": ["linear gain", "AWGN", "PCM16 clipping and quantization"],
            **stats,
        },
        "parent": {
            "waveform_sha256": hashlib.sha256(parent_wav.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(parent_manifest_path.read_bytes()).hexdigest(),
            "generator": parent_manifest["generator"],
        },
        "reference_bits": parent_manifest.get("reference_bits"),
        "reference_symbols": parent_manifest.get("reference_symbols"),
        "impairments": [{"kind": "awgn", "snr_db": snr_db, "seed": seed}],
    }
    output_wav.with_suffix(output_wav.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic AWGN child of a clean synthetic WAV")
    parser.add_argument("parent_waveform", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent-manifest", type=Path)
    parser.add_argument("--snr-db", type=float, required=True)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    parent_manifest = args.parent_manifest or args.parent_waveform.with_suffix(args.parent_waveform.suffix + ".json")
    manifest = generate_awgn_augmentation(
        args.parent_waveform,
        parent_manifest,
        args.output,
        snr_db=args.snr_db,
        gain=args.gain,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
