"""Conservative energy-gap acquisition proposals for bounded live audio.

These are *window suggestions*, never decoded frames or mode identities.
A returned span must be surrounded by actual low-energy guard intervals;
noise and continuous carriers should not manufacture packet boundaries.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

SEGMENT_GAP_SECONDS = 0.13
FRAME_MIN_SECONDS = 0.45
HOP_SECONDS = 0.015
MAX_CANDIDATES = 2

# Keep this narrow enough to reject adjacent-band traffic but broad enough to
# capture FSK shifts, not just the center carrier.
HALF_WIDTH = {"rtty": 140.0, "bpsk31": 85.0}
MIN_CONTRAST = 5.0


def propose_burst_windows(samples: np.ndarray, sample_rate: int,
                          carrier_hz: float, mode: str) -> tuple[tuple[int, int], ...]:
    """Return at most two fully bounded on-air activity regions, relative samples.

    This function makes no claims of successful decoding or accurate signal
    start/end timing. It is deliberately conservative at low SNR.
    """
    if mode not in HALF_WIDTH or not 8000 <= sample_rate <= 192000:
        return ()
    width = HALF_WIDTH[mode]
    if not (width + 20 < carrier_hz < sample_rate / 2 - width - 20):
        return ()
    values = np.asarray(samples)
    if values.ndim != 1 or not .8 * sample_rate <= len(values) <= 4 * sample_rate:
        return ()
    # Both independent FSK tones must exist before any RTTY segmentation,
    # even for the old broad energy detector. A keyed single-frequency CW
    # carrier is not sufficient evidence of a Baudot transmission.
    if mode == "rtty" and not _rtty_two_tone_gate(values, sample_rate, carrier_hz):
        return ()
    try:
        sos = butter(3, [carrier_hz - width, carrier_hz + width],
                     btype="bandpass", fs=sample_rate, output="sos")
        filtered = sosfiltfilt(sos, values)
    except ValueError:
        return ()
    hop = max(1, round(HOP_SECONDS * sample_rate))
    count = len(filtered) // hop
    if count < 16:
        return ()
    power = np.mean(filtered[:count * hop].reshape(count, hop) ** 2, axis=1)
    floor, peak = np.percentile(power, (12, 90))
    if not (np.isfinite(floor) and np.isfinite(peak)
            and peak / max(floor, 1e-10) >= MIN_CONTRAST):
        # Narrow, independent two-tone evidence can make an RTTY signal
        # visible when broad-band energy contrast has collapsed. This second
        # acquisition gate does *not* run for PSK or unknown signals.
        return _weak_rtty_bursts(values, sample_rate, carrier_hz) if mode == "rtty" else ()
    threshold = max(np.sqrt(max(floor, 1e-12) * peak), floor * 1.7)
    active = power >= threshold
    # A short low tone or phase reversal inside a modem message isn't a gap.
    # Only silence lasting at least 130ms separates candidate transmissions.
    gap = max(1, round(SEGMENT_GAP_SECONDS / HOP_SECONDS))
    lows = np.flatnonzero(np.diff(np.r_[False, ~active, False].astype(np.int8)))
    for start, stop in zip(lows[::2], lows[1::2]):
        if stop - start < gap:
            active[start:stop] = True
    edges = np.flatnonzero(np.diff(np.r_[False, active, False].astype(np.int8)))
    proposed: list[tuple[int, int]] = []
    margin = round(.03 * sample_rate)
    min_frame = round(FRAME_MIN_SECONDS * sample_rate)
    for lo, hi in zip(edges[::2], edges[1::2]):
        # Only attempt complete interior bursts; cut-off leading/trailing
        # portions are expected to be unreliable in a fixed 4s live crop.
        if lo < gap or hi > count - gap:
            continue
        start = max(0, lo * hop - margin)
        end = min(len(values), hi * hop + margin)
        if end - start >= min_frame:
            proposed.append((start, end))
    # Longest complete bursts first, preserving original sample indices.
    proposed.sort(key=lambda span: (span[1]-span[0]), reverse=True)
    return tuple(proposed[:MAX_CANDIDATES])


def _weak_rtty_bursts(samples: np.ndarray, rate: int,
                      carrier_hz: float) -> tuple[tuple[int, int], ...]:
    """Conservative two-tone, narrowband RTTY acquisition for weak bursts.

    The caller has already verified that both fixed 170-Hz-shift tone
    neighborhoods rise above a *separate* nearby spectral noise floor. A noise-window with transient
    energy contrast alone must not open a decoder trial. The alignment search
    and text/identity review remain the responsibility of the native modem.
    """
    tones = (carrier_hz - 85.0, carrier_hz + 85.0)
    if tones[0] - 35 <= 0 or tones[1] + 35 >= rate / 2:
        return ()
    hop = max(1, round(0.030 * rate))
    power = np.zeros(len(samples) // hop, dtype=np.float64)
    for tone in tones:
        sos = butter(4, [tone - 35, tone + 35],
                     btype="bandpass", fs=rate, output="sos")
        filtered = sosfiltfilt(sos, samples)
        power += np.mean(filtered[:len(power) * hop].reshape(-1, hop)**2, axis=1)
    floor, peak = np.percentile(power, (12, 90))
    if peak / max(floor, 1e-12) < 3.3:
        return ()
    active = power >= max(np.sqrt(max(floor, 1e-12) * peak), floor * 1.55)
    gap = max(1, round(.12 * rate / hop))
    edges = np.flatnonzero(np.diff(np.r_[False, ~active, False].astype(np.int8)))
    for lo, hi in zip(edges[::2], edges[1::2]):
        if hi - lo < gap:
            active[lo:hi] = True
    edges = np.flatnonzero(np.diff(np.r_[False, active, False].astype(np.int8)))
    spans: list[tuple[int, int]] = []
    margin = round(.025 * rate)
    for lo, hi in zip(edges[::2], edges[1::2]):
        # Complete interior signal only. No guessed or synthetic boundaries.
        if lo < gap or hi > len(power) - gap:
            continue
        start = max(0, lo * hop - margin)
        end = min(len(samples), hi * hop + margin)
        if .7 * rate <= end - start <= 3.2 * rate:
            spans.append((start, end))
    spans.sort(key=lambda span: span[1]-span[0], reverse=True)
    return tuple(spans[:MAX_CANDIDATES])


def _rtty_two_tone_gate(samples: np.ndarray, rate: int, carrier_hz: float) -> bool:
    """Require distinct balanced mark and space peaks above an off-tone floor.

    No station identity, protocol label, or payload truth is inferred from
    these peaks. This only allows the energy-gap segmentation attempt.
    """
    tones = (carrier_hz - 85.0, carrier_hz + 85.0)
    if tones[0] - 35 <= 0 or tones[1] + 35 >= rate / 2:
        return False
    frequencies, psd = welch(samples, fs=rate,
                             nperseg=min(8192, max(1024, rate // 2)))
    guards = ((frequencies > carrier_hz - 360)
              & (frequencies < carrier_hz + 360)
              & ((frequencies < carrier_hz - 170)
                 | (frequencies > carrier_hz + 170)))
    if np.count_nonzero(guards) < 8:
        return False
    noise_density = float(np.median(psd[guards]))
    if not np.isfinite(noise_density) or noise_density <= 0:
        return False
    powers = []
    for tone in tones:
        bins = (frequencies > tone - 35) & (frequencies < tone + 35)
        if not np.any(bins):
            return False
        tone_density = float(np.mean(psd[bins]))
        if not np.isfinite(tone_density) or tone_density < 10**(.6) * noise_density:
            return False
        powers.append(tone_density)
    # An isolated keyed single tone can have extreme Welch leakage above a
    # near-zero noise floor at the other frequency. Guard against that,
    # without requiring the two valid tones to have identical symbol duty.
    return max(powers) <= 10**(1.2) * min(powers)
