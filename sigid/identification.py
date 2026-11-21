"""Audio-only DSP evidence and conservative decoder shortlisting.

The measurements describe the selected audio ROI; they are not protocol
identification, and they cannot recover RF characteristics lost in demodulation.
Inspired by the staged-evidence approach of OnAir, without using its code.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .modes import MODE_REGISTRY
from .trials import MODULATION_TRIAL_ORDER, VALIDATED_TRIAL_MODES

EVIDENCE_VERSION = "audio-roi-spectral-v1"
MAX_ANALYSIS_SECONDS = 2
MAX_PEAKS = 8


def measure_audio_evidence(
    samples: Any,
    sample_rate: int,
    frequency_low_hz: float | None = None,
    frequency_high_hz: float | None = None,
) -> dict[str, Any]:
    """Measure spectral evidence on a bounded excerpt of a selected audio band.

    Peaks in an averaged spectrum are only *possible* tones, not proof of FSK.
    Periodic/symbol and phase features require separate estimators and are not
    inferred from an audio magnitude spectrum.
    """
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate < 1000:
        raise ValueError("audio evidence requires a positive sample rate >= 1000 Hz")
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("audio evidence requires finite mono samples")
    low = 0.0 if frequency_low_hz is None else float(frequency_low_hz)
    high = sample_rate / 2.0 if frequency_high_hz is None else float(frequency_high_hz)
    if not np.isfinite(low) or not np.isfinite(high) or not 0 <= low < high <= sample_rate / 2:
        raise ValueError("audio evidence band must be ordered and within Nyquist")
    if values.size < 1024:
        return {"available": False, "version": EVIDENCE_VERSION, "reason": "less than 1024 samples"}

    values = values[: min(values.size, MAX_ANALYSIS_SECONDS * sample_rate)]
    nfft = min(4096, values.size)
    hop = max(1, nfft // 2)
    offsets = np.arange(0, values.size - nfft + 1, hop, dtype=np.int64)
    if len(offsets) > 64:
        offsets = offsets[np.linspace(0, len(offsets) - 1, 64).round().astype(int)]
    window = np.hanning(nfft)
    frequencies = np.fft.rfftfreq(nfft, d=1 / sample_rate)
    mask = (frequencies >= low) & (frequencies <= high)
    if np.count_nonzero(mask) < 3:
        return {"available": False, "version": EVIDENCE_VERSION, "reason": "selected band has fewer than 3 FFT bins"}

    rows: list[np.ndarray] = []
    for offset in offsets:
        spectrum = np.fft.rfft(values[offset : offset + nfft] * window)
        rows.append(np.abs(spectrum[mask]) ** 2)
    power = np.mean(rows, axis=0)
    band_frequencies = frequencies[mask]
    if not np.isfinite(power).all() or float(power.max()) <= 1e-15:
        return {"available": False, "version": EVIDENCE_VERSION, "reason": "no measurable in-band energy"}

    floor = float(np.median(power))
    peak = float(power.max())
    epsilon = np.finfo(np.float64).tiny
    flatness = float(np.exp(np.mean(np.log(np.maximum(power, epsilon)))) / max(float(power.mean()), epsilon))
    peak_over_median_db = float(10 * np.log10(max(peak, epsilon) / max(floor, epsilon)))

    # Exclude the broad floor when estimating occupied *excess* energy.
    excess = np.maximum(power - floor, 0.0)
    total = float(np.sum(excess))
    occupied_bandwidth_hz = None
    if total > 0:
        cumulative = np.cumsum(excess) / total
        left = int(np.searchsorted(cumulative, 0.005))
        right = int(np.searchsorted(cumulative, 0.995))
        occupied_bandwidth_hz = float(band_frequencies[right] - band_frequencies[left])

    local_peaks = np.flatnonzero(
        (power[1:-1] > power[:-2])
        & (power[1:-1] >= power[2:])
        & (power[1:-1] >= max(floor * 4, peak * 0.08))
    ) + 1
    strongest = sorted(local_peaks.tolist(), key=lambda i: (-power[i], i))
    separated: list[int] = []
    for index in strongest:
        if all(abs(index - other) >= 2 for other in separated):
            separated.append(index)
        if len(separated) == MAX_PEAKS:
            break
    tones = sorted(float(band_frequencies[index]) for index in separated)
    spacing = None
    regular = None
    if len(tones) >= 2:
        differences = np.diff(tones)
        spacing = float(np.median(differences))
        tolerance = max(2 * sample_rate / nfft, 0.15 * spacing)
        regular = bool(np.max(np.abs(differences - spacing)) <= tolerance)

    # Only report a burst activity estimate if the recorded frames exhibit
    # significant dynamic range. A constant carrier cannot yield useful duty.
    row_energy = np.asarray([float(np.sum(row)) for row in rows])
    energy_db = 10 * np.log10(np.maximum(row_energy, epsilon))
    activity_fraction = None
    if len(energy_db) >= 4 and np.ptp(energy_db) >= 6:
        gate = float(np.percentile(energy_db, 20) + 3)
        activity_fraction = float(np.mean(energy_db > gate))

    return {
        "available": True,
        "version": EVIDENCE_VERSION,
        "source": "bounded_audio_roi",
        "sample_rate_hz": sample_rate,
        "analyzed_samples": int(values.size),
        "selected_band_hz": [low, high],
        "frequency_resolution_hz": float(sample_rate / nfft),
        "occupied_excess_bandwidth_hz": occupied_bandwidth_hz,
        "spectral_flatness": flatness,
        "peak_over_median_db": peak_over_median_db,
        "candidate_tone_peaks_hz": tones,
        "tone_spacing_hz_candidate": spacing,
        "tone_grid_regular": regular,
        "burst_activity_fraction": activity_fraction,
        "limitation": "Audio-band observations only; no modulation or protocol is verified.",
    }


def decoder_shortlist(proposal: dict[str, Any], *, max_modes: int = 3) -> list[str]:
    """Rank the existing loopback-tested Fldigi modes; preserve alternatives.

    The classifier influences *order*, never truth. Do not admit a new decoder
    based only on an ML score: the automatic path remains restricted to the
    project's independently loopback-tested shortlist.
    """
    if not 1 <= max_modes <= len(VALIDATED_TRIAL_MODES):
        raise ValueError("max_modes must fit the validated automatic trial catalog")
    allowed = set(VALIDATED_TRIAL_MODES)
    ranked: list[str] = []

    def add(mode: str) -> None:
        if mode in allowed and mode not in ranked:
            ranked.append(mode)

    candidates = proposal.get("ranked_candidates") or []
    for candidate in candidates[:8]:
        if not isinstance(candidate, dict):
            continue
        mode_id = candidate.get("mode_id")
        if not isinstance(mode_id, str):
            continue
        definition = MODE_REGISTRY.get(mode_id)
        if definition is not None:
            # Do not infer a native decoder from a generic waveform class.
            native = str(definition.get("backend_mode") or definition.get("display_name") or "")
            add(native)
        elif mode_id == "2fsk":
            add("RTTY")
        elif mode_id == "bpsk":
            add("BPSK31")
        family = candidate.get("family")
        if family == "MFSK":
            add("THOR16")
        elif family == "PSK":
            add("BPSK31")
        elif family == "FSK":
            add("RTTY")

    family = proposal.get("family") or proposal.get("candidate_family")
    hypothesis = {"FSK": "FSK-like", "MFSK": "MFSK-like", "PSK": "single-tone / unknown"}.get(family, family)
    for mode in MODULATION_TRIAL_ORDER.get(hypothesis, VALIDATED_TRIAL_MODES):
        add(mode)
    for mode in VALIDATED_TRIAL_MODES:
        add(mode)
    return ranked[:max_modes]
