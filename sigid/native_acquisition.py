"""Conservative energy-gap acquisition proposals for bounded live audio.

These are *window suggestions*, never decoded frames or mode identities.
A returned span must be surrounded by actual low-energy guard intervals;
noise and continuous carriers should not manufacture packet boundaries.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

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
        return ()
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
