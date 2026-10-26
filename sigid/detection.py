"""Interpretable spectral activity detection and provisional signal candidates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def estimate_tone_grid(
    tone_centers_hz: list[float] | tuple[float, ...],
    *,
    frequency_resolution_hz: float,
) -> dict[str, Any]:
    """Estimate spacing for an observed, contiguous equally-spaced tone set."""
    centers = np.asarray(tone_centers_hz, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2 or not np.all(np.isfinite(centers)):
        raise ValueError("tone_centers_hz must contain at least two finite frequencies")
    if np.any(np.diff(centers) <= 0):
        raise ValueError("tone_centers_hz must be strictly increasing")
    if not np.isfinite(frequency_resolution_hz) or frequency_resolution_hz <= 0:
        raise ValueError("frequency_resolution_hz must be positive and finite")
    differences = np.diff(centers)
    spacing = float(np.median(differences))
    residual = float(np.max(np.abs(differences - spacing)))
    tolerance = max(2.0 * frequency_resolution_hz, 0.15 * spacing)
    return {
        "tone_spacing_hz_candidate": spacing,
        "tone_grid_residual_hz": residual,
        "tone_grid_tolerance_hz": tolerance,
        "tone_grid_regular": residual <= tolerance,
    }


def detect_tracks(
    spectrum: Mapping[str, Any],
    *,
    threshold_db: float = 18.0,
    minimum_peak_db: float = -25.0,
    minimum_rows: int = 4,
    minimum_occupancy: float = 0.08,
    maximum_tone_gap_hz: float = 800.0,
    minimum_frequency_hz: float = 100.0,
) -> list[dict[str, Any]]:
    """Return provisional candidates from repeated narrow spectral peaks.

    This interpretable first pass groups active tone centers across time. It can
    suggest FSK/MFSK structure, but does not infer a protocol or claim a decode.
    Candidate time bounds use the supplied waterfall's sample-indexed rows.
    """
    if threshold_db <= 0 or minimum_rows < 1:
        raise ValueError("threshold_db and minimum_rows must be positive")
    if not 0 < minimum_occupancy <= 1:
        raise ValueError("minimum_occupancy must be in (0, 1]")
    if maximum_tone_gap_hz <= 0 or minimum_frequency_hz < 0:
        raise ValueError("maximum_tone_gap_hz must be positive and minimum_frequency_hz non-negative")

    rows = np.asarray(spectrum.get("db", []), dtype=np.float64)
    frequencies = np.asarray(spectrum.get("frequency_hz", []), dtype=np.float64)
    sample_starts = np.asarray(spectrum.get("sample_starts", []), dtype=np.int64)
    sample_rate = int(spectrum.get("sample_rate", 0))
    fft_size = int(spectrum.get("fft_size", 0))
    sample_count = int(spectrum.get("sample_count", 0))
    if rows.ndim != 2 or frequencies.ndim != 1 or rows.shape[1] != frequencies.size or rows.shape[0] != sample_starts.size:
        raise ValueError("spectrum rows, frequency bins, and sample positions do not agree")
    if sample_rate <= 0 or fft_size <= 0 or sample_count < 0 or not rows.size:
        raise ValueError("spectrum is missing valid sample metadata or rows")

    bin_hz = sample_rate / fft_size
    hop_samples = int(np.median(np.diff(sample_starts))) if sample_starts.size > 1 else fft_size
    first_bin = max(1, int(np.searchsorted(frequencies, minimum_frequency_hz)))
    if first_bin >= frequencies.size - 2:
        return []

    peaks: list[dict[str, float | int]] = []
    for row_index, row in enumerate(rows):
        floor = float(np.median(row[first_bin:]))
        gate = max(minimum_peak_db, floor + threshold_db)
        local_maxima = (
            (row[1:-1] >= row[:-2])
            & (row[1:-1] > row[2:])
            & (row[1:-1] >= gate)
        )
        bins = np.flatnonzero(local_maxima) + 1
        bins = bins[bins >= first_bin]
        ranked_bins = sorted(bins.tolist(), key=lambda index: float(row[index]), reverse=True)
        selected_bins: list[int] = []
        for index in ranked_bins:
            if all(abs(index - selected) >= 4 for selected in selected_bins):
                selected_bins.append(index)
        bins = np.asarray(sorted(selected_bins), dtype=np.int64)
        for index in bins:
            peaks.append(
                {
                    "bin": int(index),
                    "frequency_hz": float(frequencies[index]),
                    "snr_db": float(row[index] - floor),
                    "row": row_index,
                    "sample": int(sample_starts[row_index]),
                }
            )

    if not peaks:
        return []
    peaks.sort(key=lambda peak: float(peak["frequency_hz"]))

    # Small frequency-bin motion and window leakage are combined into a tone
    # center. Occupancy later rejects one-off noise maxima.
    tolerance_hz = max(1.55 * bin_hz, 20.0)
    tone_clusters: list[list[dict[str, float | int]]] = []
    for peak in peaks:
        if not tone_clusters or float(peak["frequency_hz"]) - float(tone_clusters[-1][-1]["frequency_hz"]) > tolerance_hz:
            tone_clusters.append([peak])
        else:
            tone_clusters[-1].append(peak)

    tones = []
    for cluster in tone_clusters:
        active_rows = {int(peak["row"]) for peak in cluster}
        first_row, last_row = min(active_rows), max(active_rows)
        occupancy = len(active_rows) / max(1, last_row - first_row + 1)
        if len(active_rows) < minimum_rows or occupancy < minimum_occupancy:
            continue
        weights = np.power(10.0, np.asarray([peak["snr_db"] for peak in cluster]) / 10.0)
        center_hz = float(np.average([peak["frequency_hz"] for peak in cluster], weights=weights))
        row_centers = []
        for row_index in sorted(active_rows):
            row_peaks = [peak for peak in cluster if int(peak["row"]) == row_index]
            row_weights = np.power(10.0, np.asarray([peak["snr_db"] for peak in row_peaks]) / 10.0)
            row_centers.append((
                (int(sample_starts[row_index]) + fft_size / 2) / sample_rate,
                float(np.average([peak["frequency_hz"] for peak in row_peaks], weights=row_weights)),
            ))
        drift_hz_per_second = (
            float(np.polyfit([item[0] for item in row_centers], [item[1] for item in row_centers], 1)[0])
            if len(row_centers) >= 2 else 0.0
        )
        tones.append(
            {
                "center_hz": center_hz,
                "drift_hz_per_second": drift_hz_per_second,
                "first_row": first_row,
                "last_row": last_row,
                "active_rows": active_rows,
                "occupancy": occupancy,
                "peak_snr_db": float(np.mean([peak["snr_db"] for peak in cluster])),
                "peak_count": len(cluster),
            }
        )

    if not tones:
        return []
    tones.sort(key=lambda tone: float(tone["center_hz"]))
    groups: list[list[dict[str, Any]]] = []
    for tone in tones:
        if not groups or float(tone["center_hz"]) - float(groups[-1][-1]["center_hz"]) > maximum_tone_gap_hz:
            groups.append([tone])
        else:
            groups[-1].append(tone)

    candidates = []
    for group in groups:
        centers = [float(tone["center_hz"]) for tone in group]
        rows_in_group = set().union(*(tone["active_rows"] for tone in group))
        # Keep one candidate through ordinary symbol gaps and only split when
        # activity disappears for a clearly separate interval.
        active_rows = sorted(rows_in_group)
        gap_samples = max(int(1.5 * sample_rate), hop_samples * (minimum_rows + 1))
        row_runs: list[list[int]] = []
        for row_index in active_rows:
            if not row_runs or int(sample_starts[row_index]) - int(sample_starts[row_runs[-1][-1]]) > gap_samples:
                row_runs.append([row_index])
            else:
                row_runs[-1].append(row_index)
        tone_count = len(centers)
        tone_grid = estimate_tone_grid(centers, frequency_resolution_hz=bin_hz) if tone_count >= 2 else {
            "tone_spacing_hz_candidate": None,
            "tone_grid_residual_hz": None,
            "tone_grid_tolerance_hz": None,
            "tone_grid_regular": False,
        }
        hypothesis = "single-tone / unknown" if tone_count == 1 else "FSK-like" if tone_count == 2 else "MFSK-like"
        for run in row_runs:
            if len(run) < minimum_rows:
                continue
            first_row, last_row = run[0], run[-1]
            span_rows = max(1, last_row - first_row + 1)
            occupancy = len(run) / span_rows
            sample_start = int(sample_starts[first_row])
            sample_end = min(sample_count, int(sample_starts[last_row]) + fft_size)
            candidates.append(
                {
                    "id": f"track-{len(candidates) + 1:03d}",
                    "sample_start": sample_start,
                    "sample_count": max(1, sample_end - sample_start),
                    "time_start_seconds": sample_start / sample_rate,
                    "time_end_seconds": sample_end / sample_rate,
                    "frequency_low_hz": max(0.0, centers[0] - bin_hz / 2),
                    "frequency_high_hz": min(sample_rate / 2, centers[-1] + bin_hz / 2),
                    "center_hz": float(np.median(centers)),
                    "bandwidth_hz": max(bin_hz, centers[-1] - centers[0] + bin_hz),
                    "drift_hz_per_second": float(np.median([tone["drift_hz_per_second"] for tone in group])),
                    "peak_snr_db": float(np.mean([tone["peak_snr_db"] for tone in group])),
                    "occupancy": occupancy,
                    "active_rows": len(run),
                    "tone_centers_hz": centers,
                    "tone_count_candidate": tone_count,
                    **tone_grid,
                    "frequency_resolution_hz": bin_hz,
                    "modulation_hypothesis": hypothesis,
                    "identification_state": "provisional",
                    "protocol": None,
                }
            )
    return candidates
