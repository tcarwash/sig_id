"""Conservative, receive-only carrier refinement for BPSK31 live acquisition.

A BPSK carrier has a spectral line after squaring the analytic baseband,
even when the varicode phase reversals smear the original carrier peak.
This estimator only proposes a carrier; the existing BPSK31 demodulator
must still pass independent timing/coherence and codeword gates.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

MAX_SEARCH_HZ = 35.0
MAX_WINDOW_SECONDS = 4.0
MIN_PEAK_RATIO = 18.0
MIN_REFINEMENT_HZ = 3.5


def propose_bpsk31_carrier(samples: np.ndarray, sample_rate: int,
                           rough_carrier_hz: float) -> dict | None:
    values=np.asarray(samples,dtype=np.float64)
    if (values.ndim!=1 or not 8000<=sample_rate<=192000
            or not .8*sample_rate<=len(values)<=MAX_WINDOW_SECONDS*sample_rate
            or not math.isfinite(rough_carrier_hz)
            or not 100<rough_carrier_hz<min(2900,sample_rate/2-100)
            or not np.isfinite(values).all()):
        return None
    power=float(np.mean(values*values))
    if power<1e-8:
        return None
    t=np.arange(values.size,dtype=np.float64)/sample_rate
    analytic=hilbert(values)
    base=analytic*np.exp(-2j*np.pi*rough_carrier_hz*t)
    sos=butter(3,75.0,fs=sample_rate,output="sos")
    filtered=sosfiltfilt(sos,base)
    squared=filtered*filtered*np.hanning(len(filtered))
    spectrum=np.abs(np.fft.fft(squared))
    frequencies=np.fft.fftfreq(len(squared),d=1.0/sample_rate)
    eligible=(np.abs(frequencies)>=.5)&(np.abs(frequencies)<=2*MAX_SEARCH_HZ)
    if int(np.count_nonzero(eligible))<10:
        return None
    indexes=np.flatnonzero(eligible)
    peak_idx=int(indexes[np.argmax(spectrum[eligible])])
    background=float(np.median(spectrum[eligible]))
    ratio=float(spectrum[peak_idx]/background) if background>0 else 0.0
    correction=float(frequencies[peak_idx]/2)
    if (not math.isfinite(ratio) or ratio<MIN_PEAK_RATIO
            or not MIN_REFINEMENT_HZ<=abs(correction)<=MAX_SEARCH_HZ):
        return None
    proposed=rough_carrier_hz+correction
    if not 100<proposed<min(2900,sample_rate/2-100):
        return None
    return {
        "strategy":"bpsk31-squared-analytic-carrier-v1",
        "initial_carrier_hz":round(rough_carrier_hz,3),
        "proposed_carrier_hz":round(proposed,3),
        "correction_hz":round(correction,3),
        "peak_to_median":round(ratio,2),
        "max_search_hz":MAX_SEARCH_HZ,
        "confirmed_identity":False,
        "not_verified":True,
    }
