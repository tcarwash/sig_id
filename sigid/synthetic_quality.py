"""Read-only quality measurements for generated RF passband WAVs.

Band edges are measured from the actual audio waveform, not an arbitrary
fallback width for a named modem. These are *spectral occupancy estimates*,
not proof that every intended data symbol or packet was transmitted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import wave

import numpy as np


def measure_occupied_band(
    wav_path: str | Path, *, low_hz: float = 100.0, high_hz: float = 3000.0,
    tail_fraction: float = 0.05, pad_hz: float = 12.0,
) -> dict[str, float]:
    """Estimate the 90%-power passband and a conservative padded box.

    The frequency extent comes from the *actual* generated waveform. A
    short-time Welch PSD averages over modulated symbols and avoids
    single-FFT cancellation and extreme transient leakage. A 5% power
    tail is intentionally retained on either side; this is not a guaranteed
    occupied-bandwidth regulation measurement.
    """
    from scipy.signal import welch

    with wave.open(str(wav_path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("occupied-band inspection requires mono uncompressed PCM16 WAV")
        sample_rate = int(wav.getframerate())
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    if samples.size < 1024 or not np.isfinite(samples).all():
        raise ValueError("occupied-band inspection requires at least 1024 finite samples")
    rms = float(np.sqrt(np.mean(samples * samples)))
    if rms < 1e-4:
        raise ValueError("source contains too little signal energy to label")

    nperseg = min(4096, samples.size)
    frequencies, density = welch(samples, fs=sample_rate, window="hann",
                                 nperseg=nperseg, noverlap=nperseg // 2,
                                 scaling="density")
    high_hz = min(high_hz, sample_rate / 2.0 - 20.0)
    selected = (frequencies >= low_hz) & (frequencies <= high_hz)
    valid_frequency = frequencies[selected]
    power = density[selected]
    if valid_frequency.size < 3 or float(power.sum()) <= 0:
        raise ValueError("source has no measurable passband energy")
    cumulative = np.cumsum(power, dtype=np.float64)
    cumulative /= cumulative[-1]
    low, high = np.interp([tail_fraction, 1.0 - tail_fraction], cumulative, valid_frequency)
    guard = max(pad_hz, 0.03 * (high - low))
    box_low = max(float(low_hz), float(low - guard))
    box_high = min(float(high_hz), float(high + guard))
    if box_high - box_low < 30.0:
        middle = (box_low + box_high) / 2.0
        box_low = max(float(low_hz), middle - 15.0)
        box_high = min(float(high_hz), middle + 15.0)

    block_samples = max(1, int(round(sample_rate * 0.04)))
    blocks = samples[:(samples.size // block_samples) * block_samples]
    rms_blocks = np.sqrt(np.mean(blocks.reshape(-1, block_samples) ** 2, axis=1))
    active_threshold = max(1e-4, float(np.quantile(rms_blocks, 0.95)) * 0.08)
    active = np.flatnonzero(rms_blocks > active_threshold)
    if not active.size:
        raise ValueError("source has no identifiable active audio frames")
    # These temporal extents are descriptive; generated source regions stay
    # full-waveform until a separately verified trimming/migration protocol.
    return {
        "frequency_low_hz": round(box_low, 3),
        "frequency_high_hz": round(box_high, 3),
        "measured_power_low_hz": round(float(low), 3),
        "measured_power_high_hz": round(float(high), 3),
        "measured_occupied_bandwidth_hz": round(float(high - low), 3),
        "waveform_rms": round(rms, 6),
        "active_fraction": round(float(np.mean(rms_blocks > active_threshold)), 5),
        "active_start_seconds": round(float(active[0] * block_samples / sample_rate), 4),
        "active_end_seconds": round(float(min(samples.size, (active[-1] + 1) * block_samples) / sample_rate), 4),
        "measurement": "Welch PSD 5-95% passband energy + guard, 40ms RMS activity",
    }


def audit_scene_corpus(data_dir: str | Path, index_path: str | Path) -> dict:
    """Evaluate an existing synthetic scene corpus without modifying it."""
    from .dataset import CaptureStore
    root = Path(data_dir)
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    store = CaptureStore(root / "dataset.sqlite3")
    records = index.get("records", [])
    signal_methods: Counter[str] = Counter()
    offsets: Counter[str] = Counter()
    clock_values: Counter[str] = Counter()
    snr_bins: Counter[str] = Counter()
    source_errors: list[dict] = []
    label_warnings: list[dict] = []
    noise_reuse: Counter[str] = Counter()
    cache: dict[str, dict] = {}
    pair_count = temporal_pairs = colocated_pairs = 0
    for record in records:
        scene = store.capture(record["scene_capture_id"])
        regions = [r for r in store.regions(scene["id"]) if r.get("method")]
        if len(regions) == 2:
            pair_count += 1
            left, right = regions
            if max(left["sample_start"], right["sample_start"]) < min(
                left["sample_start"] + left["sample_count"],
                right["sample_start"] + right["sample_count"],
            ):
                temporal_pairs += 1
            if max(left["frequency_low_hz"], right["frequency_low_hz"]) < min(
                left["frequency_high_hz"], right["frequency_high_hz"],
            ):
                colocated_pairs += 1
        for region in regions:
            signal_methods[str(region["method"])] += 1
            evidence = region.get("evidence") or {}
            offsets[str(evidence.get("frequency_offset_hz", 0))] += 1
            clock_values[str(evidence.get("clock_error_ppm", 0))] += 1
            source_id = evidence.get("component_capture_id")
            if not source_id:
                continue
            if source_id not in cache:
                source = store.capture(source_id)
                try:
                    cache[source_id] = measure_occupied_band(source["audio_path"])
                except (OSError, ValueError, wave.Error) as exc:
                    source_errors.append({"capture_id": source_id, "error": str(exc)})
                    cache[source_id] = {}
            measurement = cache[source_id]
            observed = measurement.get("frequency_high_hz", 0) - measurement.get("frequency_low_hz", 0)
            annotated = region["frequency_high_hz"] - region["frequency_low_hz"]
            if observed > 0 and annotated / observed > 4.0:
                label_warnings.append({
                    "capture_id": source_id, "method": region["method"],
                    "label_bandwidth_hz": round(float(annotated), 2),
                    "observed_source_bandwidth_hz": round(float(observed), 2),
                    "width_ratio": round(float(annotated / observed), 2),
                })
        for item in record.get("signal_snr_measurements", []):
            snr = item.get("target_snr_db")
            if isinstance(snr, (int, float)):
                label = (
                    "extreme_below_-20" if snr < -20 else
                    "weak_-20_to_-10" if snr < -10 else
                    "borderline_-10_to_0" if snr < 0 else
                    "moderate_0_to_10" if snr < 10 else "clear_10_plus"
                )
                snr_bins[label] += 1
        noise = store.capture(record["noise_capture_id"])
        for segment in noise["source_metadata"].get("sources", []):
            noise_reuse[str(segment["sha256"])] += 1
    return {
        "scene_count": len(records),
        "signal_components": sum(signal_methods.values()),
        "signal_methods": dict(signal_methods),
        "snr_bins": dict(snr_bins),
        "frequency_offsets": dict(offsets),
        "clock_errors_ppm": dict(clock_values),
        "two_signal_scenes": pair_count,
        "temporally_overlapping_pairs": temporal_pairs,
        "frequency_overlapping_pairs": colocated_pairs,
        "unique_recorded_noise_sources": len(noise_reuse),
        "max_uses_of_single_recorded_noise_source": max(noise_reuse.values(), default=0),
        "oversized_label_count": len(label_warnings),
        "oversized_label_examples": label_warnings[:20],
        "source_measurement_errors": source_errors[:20],
        "source_measurement_error_count": len(source_errors),
        "review_needed": bool(label_warnings or source_errors),
        "note": "Spectral occupancy is estimated; labels require review, and generated noise is not independent on-air evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only synthetic corpus quality report")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--scene-index", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_scene_corpus(args.data_dir, args.scene_index)
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="utf-8")
    print(data, end="")


if __name__ == "__main__":
    main()
