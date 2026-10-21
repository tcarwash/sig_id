"""Protocol-independent symbol and bit recovery experiments."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def demodulate_ook(
    samples: Any,
    sample_rate: int,
    *,
    symbol_rate: float,
    carrier_hz: float,
    timing_offset_samples: float = 0.0,
    auto_timing: bool = False,
    sample_offset: int = 0,
    integration_fraction: float = 0.8,
) -> dict[str, Any]:
    """Recover uncoded on/off-keyed bits using noncoherent carrier amplitude."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("samples must be a non-empty finite mono array")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if not math.isfinite(symbol_rate) or symbol_rate <= 0 or sample_rate / symbol_rate < 8:
        raise ValueError("symbol_rate must provide at least eight samples per symbol")
    if not math.isfinite(carrier_hz) or not 0 < carrier_hz < sample_rate / 2:
        raise ValueError("carrier_hz must be above 0 Hz and below Nyquist")
    if not isinstance(auto_timing, bool):
        raise ValueError("auto_timing must be a boolean")
    if sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")
    samples_per_symbol = sample_rate / symbol_rate
    if not math.isfinite(timing_offset_samples) or (
        not auto_timing and not 0 <= timing_offset_samples < samples_per_symbol
    ):
        raise ValueError("timing_offset_samples must be within one symbol")
    if not 0.2 <= integration_fraction <= 1.0:
        raise ValueError("integration_fraction must be between 0.2 and 1.0")

    if auto_timing:
        phase_count = min(32, max(8, int(math.ceil(samples_per_symbol))))
        offsets = np.arange(phase_count, dtype=np.float64) * samples_per_symbol / phase_count
        trials = [
            demodulate_ook(
                values,
                sample_rate,
                symbol_rate=symbol_rate,
                carrier_hz=carrier_hz,
                timing_offset_samples=float(offset),
                auto_timing=False,
                sample_offset=sample_offset,
                integration_fraction=integration_fraction,
            )
            for offset in offsets
        ]
        scores = [float(trial["level_separation_score"]) for trial in trials]
        max_symbols = max(int(trial["symbol_count"]) for trial in trials)
        eligible = [index for index, trial in enumerate(trials) if int(trial["symbol_count"]) == max_symbols]
        best_index = max(eligible, key=lambda index: scores[index])
        return {
            **trials[best_index],
            "timing_offset_estimated": True,
            "timing_search": {
                "method": "uniform-grid-two-cluster-amplitude-separation",
                "phase_count": phase_count,
                "candidate_offsets_samples": offsets.tolist(),
                "candidate_scores": scores,
                "candidate_symbol_counts": [int(trial["symbol_count"]) for trial in trials],
                "selected_score": scores[best_index],
            },
        }

    integration_length = max(8, int(math.floor(samples_per_symbol * integration_fraction)))
    window = np.hanning(integration_length)
    local_times = np.arange(integration_length, dtype=np.float64) / sample_rate
    kernel = window * np.exp(-2j * np.pi * carrier_hz * local_times)
    amplitudes: list[float] = []
    symbol_bounds: list[list[int]] = []
    index = 0
    while True:
        boundary_start = timing_offset_samples + index * samples_per_symbol
        boundary_end = timing_offset_samples + (index + 1) * samples_per_symbol
        if boundary_end > values.size:
            break
        center = (boundary_start + boundary_end) / 2
        first = max(0, int(math.ceil(center - integration_length / 2)))
        last = first + integration_length
        if last > values.size:
            break
        segment = values[first:last]
        amplitudes.append(float(2 * abs(np.sum(segment * kernel)) / max(float(window.sum()), np.finfo(float).tiny)))
        symbol_bounds.append([
            sample_offset + int(round(boundary_start)),
            sample_offset + int(round(boundary_end)),
        ])
        index += 1

    level_values = np.asarray(amplitudes, dtype=np.float64)
    if level_values.size:
        centers = np.quantile(level_values, [0.2, 0.8])
        for _ in range(32):
            assignments = np.abs(level_values[:, None] - centers[None, :]).argmin(axis=1)
            updated = np.asarray([
                float(np.mean(level_values[assignments == label])) if np.any(assignments == label) else centers[label]
                for label in range(2)
            ])
            if np.max(np.abs(updated - centers)) <= 1e-12:
                centers = updated
                break
            centers = updated
        centers.sort()
        decision_threshold = float(np.mean(centers))
        level_delta = float(centers[1] - centers[0])
        has_level_contrast = level_delta > np.finfo(float).eps * max(1.0, abs(float(centers[1])))
        bits = (level_values >= decision_threshold).astype(np.uint8) if has_level_contrast else np.asarray([], dtype=np.uint8)
        soft_metrics = ((level_values - decision_threshold) / max(level_delta / 2, np.finfo(float).eps)).tolist()
        separation_score = level_delta / max(float(np.std(level_values)), np.finfo(float).eps)
    else:
        centers = np.asarray([0.0, 0.0])
        decision_threshold = 0.0
        level_delta = 0.0
        bits = np.asarray([], dtype=np.uint8)
        soft_metrics = []
        separation_score = 0.0
        has_level_contrast = False

    if not has_level_contrast:
        symbol_bounds = []

    return {
        "schema_version": 1,
        "demodulator": "noncoherent-ook-amplitude-correlator",
        "status": "complete" if has_level_contrast else "ambiguous_levels" if amplitudes else "no_symbols",
        "sample_rate": sample_rate,
        "symbol_rate": symbol_rate,
        "samples_per_symbol": samples_per_symbol,
        "carrier_hz": carrier_hz,
        "timing_offset_samples": timing_offset_samples,
        "timing_offset_estimated": False,
        "integration_fraction": integration_fraction,
        "symbol_count": len(bits),
        "symbols": bits.astype(int).tolist(),
        "bits": "".join(str(int(bit)) for bit in bits),
        "amplitude_estimates": amplitudes,
        "soft_metrics": soft_metrics,
        "estimated_zero_amplitude": float(centers[0]),
        "estimated_one_amplitude": float(centers[1]),
        "decision_threshold": decision_threshold,
        "level_separation": level_delta,
        "level_separation_score": float(separation_score),
        "symbol_sample_bounds": symbol_bounds,
        "protocol": None,
    }


def demodulate_2fsk(
    samples: Any,
    sample_rate: int,
    *,
    symbol_rate: float,
    tone_0_hz: float,
    tone_1_hz: float,
    timing_offset_samples: float = 0.0,
    auto_timing: bool = False,
    sample_offset: int = 0,
    integration_fraction: float = 0.8,
) -> dict[str, Any]:
    """Recover uncoded 2-FSK tone decisions without protocol parsing.

    Frequencies are audio-passband frequencies in the supplied capture. Each
    symbol is measured with phase-insensitive complex correlations at the two
    requested tones. Positive soft metrics favor bit 1; negative favor bit 0.
    """
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("samples must be a non-empty finite mono array")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if not math.isfinite(symbol_rate) or symbol_rate <= 0 or sample_rate / symbol_rate < 8:
        raise ValueError("symbol_rate must provide at least eight samples per symbol")
    if not all(math.isfinite(frequency) for frequency in (tone_0_hz, tone_1_hz)):
        raise ValueError("tone frequencies must be finite")
    if not (0 < tone_0_hz < sample_rate / 2 and 0 < tone_1_hz < sample_rate / 2):
        raise ValueError("tone frequencies must be above 0 Hz and below Nyquist")
    if tone_0_hz == tone_1_hz:
        raise ValueError("2-FSK tone frequencies must differ")
    if not isinstance(auto_timing, bool):
        raise ValueError("auto_timing must be a boolean")
    if sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")
    samples_per_symbol = sample_rate / symbol_rate
    if not math.isfinite(timing_offset_samples) or (not auto_timing and not 0 <= timing_offset_samples < samples_per_symbol):
        raise ValueError("timing_offset_samples must be within one symbol")
    if not 0.2 <= integration_fraction <= 1.0:
        raise ValueError("integration_fraction must be between 0.2 and 1.0")

    if auto_timing:
        phase_count = min(32, max(8, int(math.ceil(samples_per_symbol))))
        offsets = np.arange(phase_count, dtype=np.float64) * samples_per_symbol / phase_count
        trials = [
            demodulate_2fsk(
                values,
                sample_rate,
                symbol_rate=symbol_rate,
                tone_0_hz=tone_0_hz,
                tone_1_hz=tone_1_hz,
                timing_offset_samples=float(offset),
                auto_timing=False,
                sample_offset=sample_offset,
                integration_fraction=integration_fraction,
            )
            for offset in offsets
        ]
        soft_scores = [
            float(np.mean(np.abs(trial["soft_metrics"]))) if trial["soft_metrics"] else 0.0
            for trial in trials
        ]
        energy_scores = [
            float(np.mean(np.log10(np.asarray(trial["tone_0_energy"]) + np.asarray(trial["tone_1_energy"]) + np.finfo(float).tiny)))
            if trial["tone_0_energy"] else 0.0
            for trial in trials
        ]
        energy_center = float(np.median(energy_scores))
        scores = [soft + 0.25 * (energy - energy_center) for soft, energy in zip(soft_scores, energy_scores, strict=True)]
        best_index = int(np.argmax(scores))
        return {
            **trials[best_index],
            "timing_offset_estimated": True,
            "timing_search": {
                "method": "uniform-grid-soft-metric-and-correlator-energy",
                "phase_count": phase_count,
                "candidate_offsets_samples": offsets.tolist(),
                "candidate_scores": scores,
                "candidate_soft_scores": soft_scores,
                "candidate_log_energy_scores": energy_scores,
                "selected_score": scores[best_index],
            },
        }

    symbols: list[int] = []
    soft_metrics: list[float] = []
    tone_0_energy: list[float] = []
    tone_1_energy: list[float] = []
    symbol_samples: list[list[int]] = []
    symbol_index = 0
    while True:
        boundary_start = timing_offset_samples + symbol_index * samples_per_symbol
        boundary_end = timing_offset_samples + (symbol_index + 1) * samples_per_symbol
        if boundary_end > values.size:
            break
        integration_length = max(8, int(math.floor(samples_per_symbol * integration_fraction)))
        center = (boundary_start + boundary_end) / 2
        first = max(0, int(math.ceil(center - integration_length / 2)))
        last = min(values.size, first + integration_length)
        segment = values[first:last]
        if segment.size < 8:
            break
        window = np.hanning(segment.size)
        times = np.arange(first, last, dtype=np.float64) / sample_rate
        energy_0 = float(abs(np.sum(segment * window * np.exp(-2j * np.pi * tone_0_hz * times))) ** 2)
        energy_1 = float(abs(np.sum(segment * window * np.exp(-2j * np.pi * tone_1_hz * times))) ** 2)
        denominator = energy_0 + energy_1 + np.finfo(np.float64).tiny
        metric = (energy_1 - energy_0) / denominator
        symbols.append(1 if metric >= 0 else 0)
        soft_metrics.append(float(metric))
        tone_0_energy.append(energy_0)
        tone_1_energy.append(energy_1)
        symbol_samples.append([
            sample_offset + int(round(boundary_start)),
            sample_offset + int(round(boundary_end)),
        ])
        symbol_index += 1

    return {
        "schema_version": 1,
        "demodulator": "noncoherent-2fsk-tone-correlator",
        "status": "complete" if symbols else "no_symbols",
        "sample_rate": sample_rate,
        "symbol_rate": symbol_rate,
        "samples_per_symbol": samples_per_symbol,
        "tone_0_hz": tone_0_hz,
        "tone_1_hz": tone_1_hz,
        "timing_offset_samples": timing_offset_samples,
        "timing_offset_estimated": False,
        "integration_fraction": integration_fraction,
        "symbol_count": len(symbols),
        "symbols": symbols,
        "bits": "".join(str(symbol) for symbol in symbols),
        "soft_metrics": soft_metrics,
        "tone_0_energy": tone_0_energy,
        "tone_1_energy": tone_1_energy,
        "symbol_sample_bounds": symbol_samples,
    }


def demodulate_mfsk(
    samples: Any,
    sample_rate: int,
    *,
    symbol_rate: float,
    tone_frequencies_hz: list[float] | tuple[float, ...],
    timing_offset_samples: float = 0.0,
    auto_timing: bool = False,
    auto_rate_search: bool = False,
    rate_search_min: float = 20.0,
    rate_search_max: float = 500.0,
    rate_search_steps: int = 9,
    sample_offset: int = 0,
    integration_fraction: float = 0.8,
) -> dict[str, Any]:
    """Recover M-FSK tone indices and natural-binary bits when representable."""
    values = np.asarray(samples, dtype=np.float64)
    frequencies = np.asarray(tone_frequencies_hz, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("samples must be a non-empty finite mono array")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if not math.isfinite(symbol_rate) or symbol_rate <= 0:
        raise ValueError("symbol_rate must be positive and finite")
    if not auto_rate_search and sample_rate / symbol_rate < 8:
        raise ValueError("symbol_rate must provide at least eight samples per symbol")
    if frequencies.ndim != 1 or not 2 <= frequencies.size <= 64 or not np.all(np.isfinite(frequencies)):
        raise ValueError("tone_frequencies_hz must contain 2 to 64 finite frequencies")
    if np.any(frequencies <= 0) or np.any(frequencies >= sample_rate / 2):
        raise ValueError("tone frequencies must be above 0 Hz and below Nyquist")
    if np.unique(frequencies).size != frequencies.size:
        raise ValueError("tone frequencies must be distinct")
    if not isinstance(auto_timing, bool):
        raise ValueError("auto_timing must be a boolean")
    if not isinstance(auto_rate_search, bool):
        raise ValueError("auto_rate_search must be a boolean")
    if sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")
    samples_per_symbol = sample_rate / symbol_rate
    if not math.isfinite(timing_offset_samples) or (not (auto_timing or auto_rate_search) and not 0 <= timing_offset_samples < samples_per_symbol):
        raise ValueError("timing_offset_samples must be within one symbol")
    if not 0.2 <= integration_fraction <= 1.0:
        raise ValueError("integration_fraction must be between 0.2 and 1.0")

    if auto_rate_search:
        if not math.isfinite(rate_search_min) or not math.isfinite(rate_search_max):
            raise ValueError("rate search bounds must be finite")
        if rate_search_min <= 0 or rate_search_max <= rate_search_min:
            raise ValueError("rate_search_max must be greater than a positive rate_search_min")
        if sample_rate / rate_search_max < 8:
            raise ValueError("every searched symbol rate must provide at least eight samples per symbol")
        if not isinstance(rate_search_steps, int) or isinstance(rate_search_steps, bool) or not 3 <= rate_search_steps <= 24:
            raise ValueError("rate_search_steps must be an integer from 3 to 24")
        candidate_rates = np.geomspace(rate_search_min, rate_search_max, rate_search_steps)
        if rate_search_min <= symbol_rate <= rate_search_max:
            candidate_rates = np.unique(np.append(candidate_rates, symbol_rate))
        trials = [
            demodulate_mfsk(
                values,
                sample_rate,
                symbol_rate=float(candidate_rate),
                tone_frequencies_hz=frequencies.tolist(),
                auto_timing=True,
                sample_offset=sample_offset,
                integration_fraction=integration_fraction,
            )
            for candidate_rate in candidate_rates
        ]
        confidence_scores = [
            float(np.mean(trial["soft_confidence"])) if trial["soft_confidence"] else 0.0
            for trial in trials
        ]
        power_scores = [
            float(np.mean(np.log10(np.asarray(trial["correlator_power"]) + np.finfo(float).tiny)))
            if trial["correlator_power"] else -300.0
            for trial in trials
        ]
        power_center = float(np.median(power_scores))
        scores = [confidence + 0.1 * (power - power_center) for confidence, power in zip(confidence_scores, power_scores, strict=True)]
        best_index = int(np.argmax(scores))
        return {
            **trials[best_index],
            "symbol_rate_estimated": True,
            "rate_search": {
                "method": "log-grid-soft-confidence-and-normalized-correlator-power",
                "candidate_rates": candidate_rates.tolist(),
                "candidate_scores": scores,
                "candidate_confidence_scores": confidence_scores,
                "candidate_log_power_scores": power_scores,
                "selected_score": scores[best_index],
            },
        }

    if auto_timing:
        phase_count = min(32, max(8, int(math.ceil(samples_per_symbol))))
        offsets = np.arange(phase_count, dtype=np.float64) * samples_per_symbol / phase_count
        trials = [
            demodulate_mfsk(
                values,
                sample_rate,
                symbol_rate=symbol_rate,
                tone_frequencies_hz=frequencies.tolist(),
                timing_offset_samples=float(offset),
                auto_timing=False,
                sample_offset=sample_offset,
                integration_fraction=integration_fraction,
            )
            for offset in offsets
        ]
        confidence_scores = [
            float(np.mean(trial["soft_confidence"])) if trial["soft_confidence"] else 0.0
            for trial in trials
        ]
        energy_scores = [
            float(np.mean(np.log10(np.asarray(trial["correlator_energy"]) + np.finfo(float).tiny)))
            if trial["correlator_energy"] else 0.0
            for trial in trials
        ]
        symbol_counts = [int(trial["symbol_count"]) for trial in trials]
        energy_center = float(np.median(energy_scores))
        scores = [confidence + 0.25 * (energy - energy_center) for confidence, energy in zip(confidence_scores, energy_scores, strict=True)]
        max_symbols = max(symbol_counts)
        eligible = [index for index, count in enumerate(symbol_counts) if count == max_symbols]
        best_index = max(eligible, key=lambda index: scores[index])
        return {
            **trials[best_index],
            "timing_offset_estimated": True,
            "timing_search": {
                "method": "uniform-grid-soft-confidence-and-correlator-energy",
                "phase_count": phase_count,
                "candidate_offsets_samples": offsets.tolist(),
                "candidate_scores": scores,
                "candidate_confidence_scores": confidence_scores,
                "candidate_log_energy_scores": energy_scores,
                "candidate_symbol_counts": symbol_counts,
                "selected_score": scores[best_index],
            },
        }

    tone_indices: list[int] = []
    confidence: list[float] = []
    correlator_energy: list[float] = []
    correlator_power: list[float] = []
    sample_bounds: list[list[int]] = []
    integration_length = max(8, int(math.floor(samples_per_symbol * integration_fraction)))
    window = np.hanning(integration_length)
    local_times = np.arange(integration_length, dtype=np.float64) / sample_rate
    correlations = np.exp(-2j * np.pi * frequencies[:, None] * local_times[None, :])
    symbol_index = 0
    while True:
        boundary_start = timing_offset_samples + symbol_index * samples_per_symbol
        boundary_end = timing_offset_samples + (symbol_index + 1) * samples_per_symbol
        if boundary_start >= values.size:
            break
        center = (boundary_start + boundary_end) / 2
        first = max(0, int(math.ceil(center - integration_length / 2)))
        last = first + integration_length
        if last > values.size:
            break
        segment = values[first:last]
        if segment.size < 8:
            break
        # Starting each local kernel at t=0 changes only correlation phase, not power.
        powers = np.abs(np.sum(correlations * (segment * window)[None, :], axis=1)) ** 2
        ranking = np.argsort(powers)
        best = int(ranking[-1])
        second_power = float(powers[ranking[-2]])
        confidence.append(float((powers[best] - second_power) / (powers[best] + np.finfo(np.float64).tiny)))
        correlator_energy.append(float(powers[best]))
        correlator_power.append(float(powers[best] / max(float(window.sum()) ** 2, np.finfo(float).tiny)))
        tone_indices.append(best)
        sample_bounds.append([
            sample_offset + max(0, min(values.size, int(round(boundary_start)))),
            sample_offset + max(0, min(values.size, int(round(boundary_end)))),
        ])
        symbol_index += 1

    bits_per_symbol = int(math.log2(frequencies.size)) if frequencies.size & (frequencies.size - 1) == 0 else None
    bits = (
        "".join(f"{index:0{bits_per_symbol}b}" for index in tone_indices)
        if bits_per_symbol is not None
        else None
    )
    return {
        "schema_version": 1,
        "demodulator": "noncoherent-mfsk-tone-correlator",
        "status": "complete" if tone_indices else "no_symbols",
        "sample_rate": sample_rate,
        "symbol_rate": symbol_rate,
        "symbol_rate_estimated": False,
        "samples_per_symbol": samples_per_symbol,
        "tone_frequencies_hz": frequencies.tolist(),
        "timing_offset_samples": timing_offset_samples,
        "timing_offset_estimated": False,
        "integration_fraction": integration_fraction,
        "tone_count": int(frequencies.size),
        "symbol_count": len(tone_indices),
        "symbols": tone_indices,
        "bits": bits,
        "bits_per_symbol": bits_per_symbol,
        "soft_confidence": confidence,
        "correlator_energy": correlator_energy,
        "correlator_power": correlator_power,
        "symbol_sample_bounds": sample_bounds,
        "protocol": None,
    }


def demodulate_bpsk(
    samples: Any,
    sample_rate: int,
    *,
    symbol_rate: float,
    carrier_hz: float,
    timing_offset_samples: float = 0.0,
    sample_offset: int = 0,
    integration_fraction: float = 0.8,
) -> dict[str, Any]:
    """Recover BPSK decisions with a phase estimate and explicit polarity ambiguity."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("samples must be a non-empty finite mono array")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if not math.isfinite(symbol_rate) or symbol_rate <= 0 or sample_rate / symbol_rate < 8:
        raise ValueError("symbol_rate must provide at least eight samples per symbol")
    if not math.isfinite(carrier_hz) or not 0 < carrier_hz < sample_rate / 2:
        raise ValueError("carrier_hz must be above 0 Hz and below Nyquist")
    if sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")
    samples_per_symbol = sample_rate / symbol_rate
    if not math.isfinite(timing_offset_samples) or not 0 <= timing_offset_samples < samples_per_symbol:
        raise ValueError("timing_offset_samples must be within one symbol")
    if not 0.2 <= integration_fraction <= 1.0:
        raise ValueError("integration_fraction must be between 0.2 and 1.0")

    correlations: list[complex] = []
    sample_bounds: list[list[int]] = []
    symbol_index = 0
    while True:
        boundary_start = timing_offset_samples + symbol_index * samples_per_symbol
        boundary_end = timing_offset_samples + (symbol_index + 1) * samples_per_symbol
        if boundary_end > values.size:
            break
        integration_length = max(8, int(math.floor(samples_per_symbol * integration_fraction)))
        center = (boundary_start + boundary_end) / 2
        first = max(0, int(math.ceil(center - integration_length / 2)))
        last = min(values.size, first + integration_length)
        segment = values[first:last]
        if segment.size < 8:
            break
        window = np.hanning(segment.size)
        times = np.arange(first, last, dtype=np.float64) / sample_rate
        correlations.append(complex(np.sum(segment * window * np.exp(-2j * np.pi * carrier_hz * times))))
        sample_bounds.append([
            sample_offset + int(round(boundary_start)),
            sample_offset + int(round(boundary_end)),
        ])
        symbol_index += 1

    if correlations:
        values_complex = np.asarray(correlations, dtype=np.complex128)
        phase = float(0.5 * np.angle(np.sum(values_complex**2)))
        rotated = values_complex * np.exp(-1j * phase)
        denominator = np.abs(rotated) + np.finfo(np.float64).tiny
        soft = (rotated.real / denominator).astype(np.float64)
        bits = "".join("0" if metric >= 0 else "1" for metric in soft)
    else:
        phase = 0.0
        soft = np.asarray([], dtype=np.float64)
        bits = ""
    return {
        "schema_version": 1,
        "demodulator": "phase-estimated-bpsk-correlator",
        "status": "complete" if bits else "no_symbols",
        "sample_rate": sample_rate,
        "symbol_rate": symbol_rate,
        "samples_per_symbol": samples_per_symbol,
        "carrier_hz": carrier_hz,
        "estimated_phase_radians": phase,
        "timing_offset_samples": timing_offset_samples,
        "integration_fraction": integration_fraction,
        "symbol_count": len(bits),
        "bits": bits,
        "inverted_bits": "".join("1" if bit == "0" else "0" for bit in bits),
        "soft_metrics": soft.tolist(),
        "polarity_ambiguous": bool(bits),
        "symbol_sample_bounds": sample_bounds,
        "protocol": None,
    }


def demodulate_qpsk(
    samples: Any,
    sample_rate: int,
    *,
    symbol_rate: float,
    carrier_hz: float,
    timing_offset_samples: float = 0.0,
    sample_offset: int = 0,
    integration_fraction: float = 0.8,
) -> dict[str, Any]:
    """Recover Gray-mapped QPSK symbols, retaining all four rotation hypotheses."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("samples must be a non-empty finite mono array")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if not math.isfinite(symbol_rate) or symbol_rate <= 0 or sample_rate / symbol_rate < 8:
        raise ValueError("symbol_rate must provide at least eight samples per symbol")
    if not math.isfinite(carrier_hz) or not 0 < carrier_hz < sample_rate / 2:
        raise ValueError("carrier_hz must be above 0 Hz and below Nyquist")
    if sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")
    samples_per_symbol = sample_rate / symbol_rate
    if not math.isfinite(timing_offset_samples) or not 0 <= timing_offset_samples < samples_per_symbol:
        raise ValueError("timing_offset_samples must be within one symbol")
    if not 0.2 <= integration_fraction <= 1.0:
        raise ValueError("integration_fraction must be between 0.2 and 1.0")

    correlations: list[complex] = []
    sample_bounds: list[list[int]] = []
    symbol_index = 0
    while True:
        boundary_start = timing_offset_samples + symbol_index * samples_per_symbol
        boundary_end = timing_offset_samples + (symbol_index + 1) * samples_per_symbol
        if boundary_end > values.size:
            break
        integration_length = max(8, int(math.floor(samples_per_symbol * integration_fraction)))
        center = (boundary_start + boundary_end) / 2
        first = max(0, int(math.ceil(center - integration_length / 2)))
        last = min(values.size, first + integration_length)
        segment = values[first:last]
        if segment.size < 8:
            break
        window = np.hanning(segment.size)
        times = np.arange(first, last, dtype=np.float64) / sample_rate
        correlations.append(complex(np.sum(segment * window * np.exp(-2j * np.pi * carrier_hz * times))))
        sample_bounds.append([
            sample_offset + int(round(boundary_start)),
            sample_offset + int(round(boundary_end)),
        ])
        symbol_index += 1

    if correlations:
        correlation_values = np.asarray(correlations, dtype=np.complex128)
        phase = float(0.25 * np.angle(np.sum(correlation_values**4)))
        mapping = ["00", "01", "11", "10"]
        hypotheses = []
        for rotation_index in range(4):
            rotation = rotation_index * math.pi / 2
            corrected = correlation_values * np.exp(-1j * (phase + rotation))
            angles = np.mod(np.angle(corrected), 2 * math.pi)
            symbols = (np.rint(angles / (math.pi / 2)).astype(np.int64) % 4).tolist()
            bits = "".join(mapping[index] for index in symbols)
            hypotheses.append({"rotation_radians": rotation, "symbols": symbols, "bits": bits})
        base_angles = np.mod(np.angle(correlation_values * np.exp(-1j * phase)), 2 * math.pi)
        residual = np.abs((base_angles + math.pi / 4) % (math.pi / 2) - math.pi / 4)
        soft_confidence = np.clip(1.0 - residual / (math.pi / 4), 0.0, 1.0).tolist()
    else:
        phase = 0.0
        hypotheses = [{"rotation_radians": index * math.pi / 2, "symbols": [], "bits": ""} for index in range(4)]
        soft_confidence = []
    return {
        "schema_version": 1,
        "demodulator": "phase-estimated-qpsk-correlator",
        "status": "complete" if correlations else "no_symbols",
        "sample_rate": sample_rate,
        "symbol_rate": symbol_rate,
        "samples_per_symbol": samples_per_symbol,
        "carrier_hz": carrier_hz,
        "estimated_phase_radians": phase,
        "timing_offset_samples": timing_offset_samples,
        "integration_fraction": integration_fraction,
        "symbol_count": len(correlations),
        "bits_per_symbol": 2,
        "rotation_ambiguous": bool(correlations),
        "symbol_hypotheses": hypotheses,
        "soft_confidence": soft_confidence,
        "symbol_sample_bounds": sample_bounds,
        "protocol": None,
    }
