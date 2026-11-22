"""Bounded, receive-only audio timing evidence for a selected narrowband region.

The estimator observes changes in instantaneous audio frequency. A coherent
transition rhythm can suggest a symbol clock for FSK-like signals, but a frame
sync, protocol, or station identity is never inferred from this measurement.
"""

from __future__ import annotations

from math import gcd
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import butter, find_peaks, hilbert, resample_poly, sosfiltfilt

TIMING_EVIDENCE_VERSION = "audio-frequency-transitions-v1"
MAX_TIMING_SECONDS = 16
TIMING_SAMPLE_RATE = 6000


def measure_timing_evidence(
    samples: Any,
    sample_rate: int,
    frequency_low_hz: float,
    frequency_high_hz: float,
) -> dict[str, Any]:
    """Estimate possible tone spacing and symbol timing; abstain when ambiguous.

    This is meaningful for narrowband frequency-stepped emissions. Other
    modulation families may not have measurable *frequency* transitions.
    No BPSK/PSK clock is claimed from the absence of this evidence.
    """
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate < 4000:
        raise ValueError("sample_rate must be an integer >= 4000")
    low, high = float(frequency_low_hz), float(frequency_high_hz)
    if not np.isfinite([low, high]).all() or not 0 <= low < high <= min(3000, sample_rate / 2):
        raise ValueError("selected audio frequency bounds must be ordered and in band")
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("samples must be finite mono audio")
    base: dict[str, Any] = {
        "version": TIMING_EVIDENCE_VERSION,
        "source": "audio_instantaneous_frequency_transitions",
        "available": False,
        "symbol_rate_baud_candidate": None,
        "symbol_period_seconds_candidate": None,
        "tone_spacing_hz_candidate": None,
        "tone_centers_hz_candidate": [],
        "transition_count": 0,
        "timing_coherence": None,
        "selected_band_hz": [low, high],
        "limitation": (
            "Frequency-transition evidence only. Other modulations need different "
            "estimators; neither a symbol rate nor a tone grid verifies a protocol."
        ),
    }
    if len(values) < 3 * sample_rate:
        return {**base, "reason": "need at least 3 seconds of selected audio"}
    if high - low > 190:
        return {**base, "reason": "ROI is too wide for a single-tone-trajectory estimate; narrow or split first"}
    if high - low < 12 or low < 25:
        return {**base, "reason": "ROI cannot resolve independent frequency changes"}
    count = min(len(values), MAX_TIMING_SECONDS * sample_rate)
    start = (len(values) - count) // 2
    values = values[start : start + count]
    if float(np.sqrt(np.mean(values**2))) < 1e-5:
        return {**base, "reason": "insufficient input audio energy"}

    rate = sample_rate
    if sample_rate > TIMING_SAMPLE_RATE:
        divisor = gcd(sample_rate, TIMING_SAMPLE_RATE)
        values = resample_poly(values, TIMING_SAMPLE_RATE // divisor, sample_rate // divisor)
        rate = TIMING_SAMPLE_RATE

    # The bandpass suppresses other signals outside the user/CV-defined box.
    # A crowded *inside* ROI still creates ambiguity and must be reviewed.
    band_low = max(20.0, low - 8.0)
    band_high = min(rate / 2 - 20.0, high + 8.0)
    sos = butter(4, [band_low, band_high], fs=rate, btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, values)
    analytic = hilbert(filtered)
    amplitude = np.abs(analytic)
    if float(np.percentile(amplitude, 90)) < 1e-4:
        return {**base, "reason": "no measurable energy inside the selected audio band"}

    # Circular phase difference avoids unwrapping across phase discontinuities.
    phase_step = np.angle(analytic[1:] * np.conj(analytic[:-1]))
    smoothed_phase = uniform_filter1d(phase_step, size=max(3, round(rate * 0.018)))
    hop = max(1, round(rate * 0.02))
    trajectory = smoothed_phase[hop // 2 :: hop] * rate / (2 * np.pi)
    levels = amplitude[1 + hop // 2 :: hop][: len(trajectory)]
    times = (np.arange(len(trajectory)) * hop + hop // 2) / rate
    trajectory = median_filter(trajectory, size=3)
    valid = (
        np.isfinite(trajectory)
        & (trajectory >= low - 7)
        & (trajectory <= high + 7)
        & (levels >= max(1e-5, float(np.percentile(levels, 20)) * 0.6))
    )
    if int(valid.sum()) < 80:
        return {**base, "reason": "too few stable in-band tone observations"}

    # Quantized instantaneous-frequency occupancy, not a raw averaged FFT:
    # hopping tones should form a small set of clusters over time.
    step_hz = 0.75
    edges = np.arange(max(0, low - 8), high + 9 + step_hz, step_hz)
    histogram, _ = np.histogram(trajectory[valid], bins=edges)
    smoothed_histogram = gaussian_filter1d(histogram.astype(float), 1.0)
    peaks, _ = find_peaks(
        smoothed_histogram,
        prominence=max(2.0, float(smoothed_histogram.max()) * 0.12),
        distance=max(1, round(3.8 / step_hz)),
    )
    peak_indices = sorted(peaks, key=lambda idx: smoothed_histogram[idx], reverse=True)[:16]
    centers = sorted(float((edges[idx] + edges[idx + 1]) / 2) for idx in peak_indices)
    tone_spacing = None
    if len(centers) >= 3:
        differences = np.diff(centers)
        nearest = differences[(differences >= 3.0) & (differences <= 30.0)]
        if len(nearest) >= 2:
            candidate_spacing = float(np.median(nearest))
            # Reject irregular spacings rather than assuming a false tone bank.
            quantization_error = [
                abs((center - centers[0]) / candidate_spacing
                    - round((center - centers[0]) / candidate_spacing))
                for center in centers
            ]
            if float(np.median(quantization_error)) <= 0.22:
                tone_spacing = candidate_spacing

    difference = np.abs(np.diff(trajectory))
    both_valid = valid[1:] & valid[:-1]
    difference = np.where(both_valid, difference, 0.0)
    local_threshold = max(2.4, float(np.percentile(difference[both_valid], 80)) if both_valid.any() else 2.4)
    events, _ = find_peaks(
        difference,
        height=local_threshold,
        prominence=1.3,
        distance=max(2, round(0.07 * rate / hop)),
    )
    # The first/last 0.3 s of filtered data can contain Hilbert/filter transients.
    event_times = times[events + 1]
    event_times = event_times[(event_times >= 0.3) & (event_times <= times[-1] - 0.3)]
    result = {
        **base,
        "available": True,
        "analyzed_seconds": float(len(values) / rate),
        "time_step_seconds": float(hop / rate),
        "tone_centers_hz_candidate": [round(value, 2) for value in centers],
        "tone_spacing_hz_candidate": round(tone_spacing, 3) if tone_spacing is not None else None,
        "transition_count": len(event_times),
        "trajectory_valid_fraction": float(np.mean(valid)),
    }
    if len(event_times) < 10 or len(centers) < 2:
        return {**result, "reason": "not enough coherent frequency transitions for a clock"}
    rates = np.arange(3.0, 40.001, 0.05)
    coherence = np.abs(np.mean(
        np.exp(2j * np.pi * rates[:, None] * event_times[None, :]), axis=1,
    ))
    coverage = len(event_times) / (rates * (times[-1] - times[0]))
    scores = coherence * np.clip(coverage / 0.5, 0, 1)
    index = int(np.argmax(scores))
    selected_rate = float(rates[index])
    selected_coherence = float(coherence[index])
    if selected_coherence < 0.64 or float(coverage[index]) < 0.20:
        return {**result, "reason": "transition timing is not sufficiently periodic",
                "timing_coherence": selected_coherence}
    return {
        **result,
        "symbol_rate_baud_candidate": round(selected_rate, 3),
        "symbol_period_seconds_candidate": round(1 / selected_rate, 4),
        "timing_coherence": selected_coherence,
        "reason": "periodic frequency transitions observed; candidate only, not protocol sync",
    }
