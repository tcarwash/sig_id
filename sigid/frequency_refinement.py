"""Conservative, frequency-only refinement of broad CV audio regions.

This uses high-resolution audio STFT evidence to *suggest* independently
occupied sub-bands. It does not assert separate protocol instances, produce
new signal identities, or modify any decoder/capture state.
"""

from __future__ import annotations

from math import gcd
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import resample_poly, stft

REFINEMENT_VERSION = "audio-stft-subbands-v1"
MAX_REFINEMENT_SECONDS = 16.0
MAX_REFINEMENT_COMPONENTS = 8
TARGET_SAMPLE_RATE = 6000
FFT_LENGTH = 4096


def refine_frequency_region(
    samples: Any,
    sample_rate: int,
    frequency_low_hz: float,
    frequency_high_hz: float,
    *,
    min_component_width_hz: float = 15.0,
    min_gap_hz: float = 11.0,
) -> dict[str, Any]:
    """Suggest a split only with two resolvable, separated occupied bands.

    A narrow single carrier, sparse FSK tone pair, noise floor, or overlapping
    emissions must not become a confident two-station assertion. Time geometry
    remains owned by CV; the returned frequency-only proposals require review.
    """
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate < 4000:
        raise ValueError("sample_rate must be an integer of at least 4000 Hz")
    low = float(frequency_low_hz)
    high = float(frequency_high_hz)
    if not np.isfinite((low, high)).all() or not 0 <= low < high <= min(3000, sample_rate / 2):
        raise ValueError("frequency region must fit inside the observable audio passband")
    if min_component_width_hz < 6 or min_gap_hz < 4:
        raise ValueError("refinement widths must be at least 6 Hz and gaps at least 4 Hz")

    base: dict[str, Any] = {
        "version": REFINEMENT_VERSION,
        "source": "audio_stft_frequency_only",
        "status": "insufficient_evidence",
        "split_suggested": False,
        "parent_frequency_hz": [low, high],
        "subbands": [],
        "limitation": "Frequency-only proposals, not independent signals or verified modes.",
    }
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("refinement requires finite, mono audio")
    if high - low < max(70.0, 2 * min_component_width_hz + min_gap_hz):
        return {**base, "reason": "region too narrow for an independently evidenced split"}
    if len(values) < sample_rate * 2:
        return {**base, "reason": "need at least two seconds of audio"}
    # Analyze the middle rather than an arbitrary start of a burst; cap CPU,
    # memory and latency regardless of the source recording size.
    bounded_samples = min(len(values), int(round(MAX_REFINEMENT_SECONDS * sample_rate)))
    start = max(0, (len(values) - bounded_samples) // 2)
    values = values[start : start + bounded_samples]

    if sample_rate > TARGET_SAMPLE_RATE:
        divisor = gcd(TARGET_SAMPLE_RATE, sample_rate)
        values = resample_poly(values, TARGET_SAMPLE_RATE // divisor, sample_rate // divisor)
        analysis_rate = TARGET_SAMPLE_RATE
    else:
        analysis_rate = sample_rate
    if len(values) < FFT_LENGTH:
        return {**base, "reason": "audio excerpt shorter than the high-resolution FFT"}
    freq, _, spectrum = stft(
        values, fs=analysis_rate, window="hann", nperseg=FFT_LENGTH,
        noverlap=FFT_LENGTH - max(1, analysis_rate // 12),
        boundary=None, padded=False,
    )
    resolution = float(freq[1] - freq[0])
    band = (freq >= low) & (freq <= high)
    if np.count_nonzero(band) < 8:
        return {**base, "reason": "insufficient frequency bins", "resolution_hz": resolution}

    # Whiten each time frame against nearby audio frequencies, preventing
    # broadband noise or receiver gain changes from looking like an occupied
    # sub-band. 41 bins at 6 kHz ~= 60 Hz, wider than ordinary tone spacing.
    power_db = 10.0 * np.log10(np.maximum(np.abs(spectrum) ** 2, 1e-16))
    background_db = median_filter(power_db, size=(41, 1), mode="nearest")
    excess_db = np.clip(power_db - background_db - 6.0, 0.0, 18.0)
    profile = gaussian_filter1d(np.mean(excess_db[band], axis=1), 1.35)
    band_freq = freq[band]
    noise_score = float(np.median(profile))
    peak_score = float(profile.max())
    result = {
        **base,
        "resolution_hz": resolution,
        "analyzed_seconds": float(len(values) / analysis_rate),
        "peak_excess_score": peak_score,
        "median_excess_score": noise_score,
    }
    # This absolute floor and local contrast requirement prevent splitting
    # quiet/noise-dominated recordings into arbitrary spectral islands.
    if peak_score < max(0.65, noise_score + 0.45, noise_score * 3.5):
        return {**result, "reason": "no sufficiently prominent spectral components"}

    threshold = max(0.30, noise_score + max(0.20, (peak_score - noise_score) * 0.17))
    occupied = profile > threshold
    active_bins = np.flatnonzero(occupied)
    if not len(active_bins):
        return {**result, "reason": "no occupied bins above local contrast threshold"}

    # Merge narrow spectral gaps that could represent adjacent hopping tones.
    # A real separation must be wider than min_gap_hz after this step.
    groups: list[np.ndarray] = []
    current = [int(active_bins[0])]
    for index in active_bins[1:]:
        index = int(index)
        gap_hz = (index - current[-1] - 1) * resolution
        if gap_hz <= min_gap_hz:
            current.append(index)
        else:
            groups.append(np.asarray(current, dtype=int))
            current = [index]
    groups.append(np.asarray(current, dtype=int))

    candidates: list[dict[str, Any]] = []
    for group in groups:
        start_bin, end_bin = int(group[0]), int(group[-1])
        width = (end_bin - start_bin + 1) * resolution
        if width < min_component_width_hz:
            continue
        # Guard against weak side lobes: each candidate needs a meaningful
        # absolute peak above the whitened local floor.
        segment = profile[start_bin : end_bin + 1]
        if float(segment.max()) < 0.65:
            continue
        margin = 1.5 * resolution
        f0 = max(low, float(band_freq[start_bin] - margin))
        f1 = min(high, float(band_freq[end_bin] + margin))
        candidates.append({
            "frequency_low_hz": f0,
            "frequency_high_hz": f1,
            "peak_excess_score": float(segment.max()),
            "frequency_resolution_hz": resolution,
        })
    if not 2 <= len(candidates) <= MAX_REFINEMENT_COMPONENTS:
        return {**result, "reason": f"{len(candidates)} resolved components; split requires 2–8"}

    # The unclaimed gap is explicitly preserved. A splitter does not tile
    # the entire CV box or assign unobserved frequencies to a station.
    for left, right in zip(candidates, candidates[1:]):
        if right["frequency_low_hz"] - left["frequency_high_hz"] < min_gap_hz - 3 * resolution:
            return {**result, "reason": "components do not have a resolvable separating gap"}
    return {
        **result,
        "status": "split_suggested",
        "split_suggested": True,
        "reason": f"{len(candidates)} spectrally separated, independently occupied bands",
        "subbands": candidates,
    }
