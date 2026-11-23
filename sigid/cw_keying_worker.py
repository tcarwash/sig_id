"""Worker-local CW stream registry and bounded trial product.

The single live decoder child process owns these states. No DSP enters the
audio callback or UI request thread, and states are keyed by session/track.
"""
from __future__ import annotations

from collections import OrderedDict
import time
from typing import Any

import numpy as np

from .cw_keying_stream import CWKeyingStream, MAX_CARRIER_RETUNE_STEP_HZ

MAX_TRACKS = 12
_streams: OrderedDict[tuple[str, str], CWKeyingStream] = OrderedDict()


def stream_cw_candidate(pcm: np.ndarray, sample_rate: int, sample_start: int,
                        carrier_hz: float, stream_id: tuple[str, str]) -> dict[str, Any]:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    session_id, track_id = stream_id
    key = (session_id, track_id)
    state = _streams.get(key)
    reset_cause = None
    if state is None:
        reset_cause = "new-stream"
    elif state.sample_rate != sample_rate:
        reset_cause = "sample-rate-changed"
    elif abs(state.carrier_hz-carrier_hz) > MAX_CARRIER_RETUNE_STEP_HZ:
        reset_cause = "carrier-jump"
    if reset_cause is not None:
        state = CWKeyingStream(session_id, track_id, sample_rate, carrier_hz)
        _streams[key] = state
    try:
        event = state.feed(pcm, sample_start=sample_start, carrier_hz=carrier_hz)
    except ValueError:
        # A gap, retune, backwards capture or changed session cannot splice
        # together Morse symbols from unrelated audio.
        state = CWKeyingStream(session_id, track_id, sample_rate, carrier_hz)
        _streams[key] = state
        event = state.feed(pcm, sample_start=sample_start, carrier_hz=carrier_hz)
        event["reason"] = "reset-after-source-discontinuity"
        reset_cause = "source-discontinuity"
    event["state_reset_reason"] = reset_cause
    _streams.move_to_end(key)
    while len(_streams) > MAX_TRACKS:
        _streams.popitem(last=False)

    delta = event["delta_text"]
    grade = "coherent-unverified" if delta else "none"
    result = {
        "decoder_id": "cw", "version": "cw-key-transition-v1",
        "source": "sigid-independent-implementation",
        "status": "decoded" if delta else "no_decode",
        "text": delta, "tokens": [], "frames": [], "soft_bits": [],
        "note": "Experimental incremental Morse timing; not independently verified.",
        "evidence": {
            "carrier_hz": state.carrier_hz,
            "initial_carrier_hz": event["initial_carrier_hz"],
            "carrier_retune_count": event["carrier_retune_count"],
            "mixer_phase_continuous": event["mixer_phase_continuous"],
            "confidence_kind": "unverified_timing_fit",
            "source_window_start_sample": sample_start,
            "source_window_sample_count": len(pcm),
            "stream_only_new_samples": event["processed_new_samples"],
        },
        "cw_stream": event,
        "live_acquisition": {
            "strategy": event["strategy"],
            "original_sample_start": sample_start,
            "original_sample_count": len(pcm),
            "selected_sample_start": sample_start,
            "selected_sample_count": len(pcm),
            "confirmed_identity": False,
        },
    }
    wall = round((time.perf_counter()-start_wall)*1000, 2)
    cpu = round((time.process_time()-start_cpu)*1000, 2)
    attempt = {
        "mode": "cw", "status": result["status"], "grade": grade,
        "text": delta, "result": result,
        "acquisition": result["live_acquisition"],
        "wall_ms": wall, "cpu_ms": cpu,
    }
    return {
        "attempted": [attempt], "selected_mode": "cw",
        "grade": grade, "text": delta, "decoder_result": result,
        "cpu_ms": cpu, "wall_ms": wall,
    }
