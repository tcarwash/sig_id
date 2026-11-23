"""Bounded, receive-only native decoder evidence for live signal candidates.

All modem DSP runs in one separate process. The audio callback and detection
endpoints never wait for a decoder. Evidence assists, but never overwrites,
classifier output or automatically asserts a verified protocol identity.
"""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import Executor, ProcessPoolExecutor, Future
import multiprocessing as mp
from threading import Lock
import math
import statistics
import time
from typing import Any

import numpy as np

from .native_acquisition import propose_burst_windows
from .cw_token_stream import CWTokenStream, MAX_CW_STREAM_TRACKS
from .modem_live_stream import ModemEvidenceStream, SUPPORTED_MODES

MAX_WINDOW_SECONDS = 4
MAX_CW_WINDOW_SECONDS = 8
MAX_MODES_PER_ATTEMPT = 2
MAX_ATTEMPTS_PER_MINUTE = 5
MAX_CW_STREAM_ATTEMPTS_PER_MINUTE = 16
MAX_MODEM_FOLLOWUP_ATTEMPTS_PER_MINUTE = 10
CW_STREAM_TRACK_COOLDOWN_SECONDS = 2.8
MODEM_FOLLOWUP_TRACK_COOLDOWN_SECONDS = 5.8
MAX_MODEM_STREAM_TRACKS = 20
MAX_CPU_MS_PER_MINUTE = 3000
TRACK_COOLDOWN_SECONDS = 24
MIN_ATTEMPT_GAP_SECONDS = 3
MAX_RECENT_RESULTS = 40
MAX_TEXT_LENGTH = 2048
# Speculative detector evidence never gets the unlimited CW stream lane:
# a small separate per-minute quota is layered atop the shared CPU budget.
MAX_PROVISIONAL_PROBES_PER_MINUTE = 3
PROVISIONAL_MIN_SECONDS = 1.5
PROVISIONAL_MIN_CV_CONFIDENCE = 0.85
PROVISIONAL_MIN_SNR_DB = 14.0


def _ranked_mode_hint(proposal: dict[str, Any]) -> str | None:
    """Strong *relative ranking* for trial order, never an accepted identity.

    Only use a mode when the model provides bounded numeric scores and
    enough separation from the runner-up. No probabilities are claimed.
    """
    if not proposal.get("available") or not proposal.get("abstained"):
        return None
    ranks = proposal.get("ranked_candidates")
    if not isinstance(ranks, list) or not ranks or not isinstance(ranks[0], dict):
        return None
    top = ranks[0]
    try:
        score = float(top["relative_model_score"])
        second = float(ranks[1]["relative_model_score"]) if len(ranks) > 1 else 0.0
    except (KeyError, ValueError, TypeError, IndexError):
        return None
    if (not math.isfinite(score) or not math.isfinite(second)
            or not 0 <= score <= 1 or not 0 <= second <= 1
            or score < .55 or score-second < .15):
        return None
    mode = str(top.get("mode_id") or "").lower()
    if mode in ("cw", "fldigi-cw", "morse"):
        return "cw"
    if mode in ("rtty", "fldigi-rtty"):
        return "rtty"
    if mode in ("bpsk31", "psk31", "fldigi-bpsk31"):
        return "bpsk31"
    if mode in ("ax25", "ax.25", "aprs"):
        return "ax25"
    return None


def explicit_modem_hypothesis(track: dict[str, Any]) -> str | None:
    """Model-based follow-up lane, never verified mode identity."""
    proposal = track.get("classifier_proposal") or track.get("model_proposal") or {}
    hint = _ranked_mode_hint(proposal)
    if hint is not None:
        return hint
    if not proposal.get("available") or proposal.get("abstained"):
        return None
    identity = " ".join(str(proposal.get(x) or "").lower()
                        for x in ("method", "method_id", "family"))
    for needle, mode in (("ax.25", "ax25"), ("ax25", "ax25"),
                         ("aprs", "ax25"), ("rtty", "rtty"),
                         ("baudot", "rtty"), ("bpsk31", "bpsk31"),
                         ("psk31", "bpsk31"), ("morse", "cw"),
                         ("fldigi-cw", "cw")):
        if needle in identity:
            return mode
    return None


def _explicit_cw_hypothesis(track: dict[str, Any]) -> bool:
    proposal = track.get("classifier_proposal") or track.get("model_proposal") or {}
    if _ranked_mode_hint(proposal) == "cw":
        return True
    if not proposal.get("available") or proposal.get("abstained"):
        return False
    identity = " ".join(str(proposal.get(x) or "").lower()
                        for x in ("method", "method_id", "family"))
    return "morse" in identity or " cw " in f" {identity} " or "fldigi-cw" in identity


def provisional_probe_kind(track: dict[str, Any], sample_rate: int) -> str | None:
    """Allow only well-supported early trials; never elevate track_state."""
    if track.get("track_state") != "provisional" or sample_rate <= 0:
        return None
    count = int(track.get("sample_count") or 0)
    if count < PROVISIONAL_MIN_SECONDS * sample_rate:
        return None
    low, high = float(track.get("frequency_low_hz") or 0), float(track.get("frequency_high_hz") or 0)
    if not (20 <= low < high <= min(3000., sample_rate/2)):
        return None
    snr = float(track.get("peak_snr_db") or 0)
    confidence = float(track.get("confidence") or 0)
    if not (math.isfinite(snr) and math.isfinite(confidence)):
        return None
    if snr < PROVISIONAL_MIN_SNR_DB and confidence < PROVISIONAL_MIN_CV_CONFIDENCE:
        return None
    if high-low <= 80 and _explicit_cw_hypothesis(track):
        return "provisional_cw"
    if int(track.get("track_observations") or 0) >= 2 and count >= 2*sample_rate:
        return "provisional_modem"
    return None


def candidate_modes(track: dict[str, Any]) -> tuple[str, ...]:
    """Two *unverified* modem hypotheses, with explicit model evidence first.

    Audio frequency is never itself a modulation ID. An early Signal Lab
    heuristic erroneously forced CW at low carriers and PSK31 at high ones,
    even after an accepted classifier named another mode. We preserve those
    useful fallback lanes without allowing them to suppress a classifier's
    explicit suggestion. A wrong classifier still gets a spectral fallback.
    """
    proposal = track.get("classifier_proposal") or track.get("model_proposal") or {}
    ranked_hint = _ranked_mode_hint(proposal)
    if proposal.get("abstained") or not proposal.get("available"):
        proposal = {}
    identity = " ".join(str(proposal.get(key) or "").lower()
                        for key in ("method", "method_id", "family", "top_candidate"))
    low = float(track.get("frequency_low_hz", 0.0))
    high = float(track.get("frequency_high_hz", 0.0))
    width = high - low
    center = (low + high) / 2

    if 0 < center < 420 and high < 850:
        fallback = ("cw", "rtty")
    elif center > 2300 and low > 2100:
        fallback = ("bpsk31", "cw")
    elif width > 600:
        fallback = ("ax25", "rtty")
    elif width > 170:
        fallback = ("rtty", "bpsk31")
    elif center >= 1600:
        fallback = ("bpsk31", "cw")
    else:
        fallback = ("cw", "rtty")

    explicit = None
    if "ax.25" in identity or "ax25" in identity or "aprs" in identity:
        explicit = "ax25"
    elif "rtty" in identity or "baudot" in identity:
        explicit = "rtty"
    elif "bpsk31" in identity or "psk31" in identity:
        explicit = "bpsk31"
    elif "morse" in identity or " cw " in f" {identity} ":
        explicit = "cw"
    elif "psk" in identity:
        explicit = "bpsk31"
    elif "fsk" in identity:
        explicit = "rtty"

    # The detector frequency and classifier label are separate, uncertain
    # sources. Never silently override the model with a Signal Lab frequency
    # lane, and never allow any candidate count to exceed the worker limit.
    candidates = (explicit, *fallback) if explicit else (
        (ranked_hint, *fallback) if ranked_hint else fallback
    )
    return tuple(dict.fromkeys(mode for mode in candidates if mode))[:MAX_MODES_PER_ATTEMPT]


def _evidence_grade(result: Any) -> tuple[str, str]:
    """Return a conservative review grade, never an automatic label."""
    if result.status != "decoded":
        return "none", ""
    if any(check.outcome == "passed" and check.kind in ("crc", "checksum")
           for frame in result.frames for check in frame.integrity):
        return "frame-integrity-passed", result.text[:MAX_TEXT_LENGTH]
    text = result.text.strip()
    if (len(text) < 5 or len(text) > MAX_TEXT_LENGTH
            or sum(c.isprintable() for c in text) / len(text) < 0.98
            or not any(c.isalpha() for c in text)
            or len(set(text.replace(" ", ""))) < 2):
        return "none", ""
    return "coherent-unverified", text[:MAX_TEXT_LENGTH]


# This module-level cache is local to the dedicated child process.
_worker_library = None


def decode_live_candidate(pcm: bytes, sample_rate: int, sample_start: int,
                          carrier_hz: float, modes: tuple[str, ...],
                          stream_id: tuple[str, str] | None = None) -> dict[str, Any]:
    """Isolated process worker: no DB, HTTP, device, CAT, audio output or PTT."""
    from .libsigdecode import DecodeWindow, create_library
    global _worker_library
    if _worker_library is None:
        _worker_library = create_library()
    if not 8000 <= sample_rate <= 192000:
        raise ValueError("unsupported live sample rate")
    if not 1 <= len(modes) <= MAX_MODES_PER_ATTEMPT:
        raise ValueError("native live decode requires one or two hypotheses")
    values = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    max_seconds = MAX_CW_WINDOW_SECONDS if "cw" in modes else MAX_WINDOW_SECONDS
    if not len(values) or len(values) > max_seconds * sample_rate:
        raise ValueError("live decoder window must be bounded for the requested modes")
    if not np.isfinite(carrier_hz) or not 20 <= carrier_hz < sample_rate / 2:
        raise ValueError("invalid audio carrier")
    if stream_id is not None and modes[0] == "cw":
        from .cw_keying_worker import stream_cw_candidate
        return stream_cw_candidate(values, sample_rate, sample_start, carrier_hz, stream_id)
    attempts = []
    best = None
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    for mode in modes:
        started = time.perf_counter()
        cpu_started = time.process_time()
        # A CW candidate can use an eight-second lookback to accumulate
        # complete characters across partial windows. Other modes retain
        # their original four-second DSP cap even when paired with CW.
        if mode != "cw" and len(values) > MAX_WINDOW_SECONDS * sample_rate:
            mode_values = values[-MAX_WINDOW_SECONDS * sample_rate:]
            mode_start = sample_start + len(values) - len(mode_values)
        else:
            mode_values = values
            mode_start = sample_start
        try:
            # The CV detector gives a *region* center, not the sub-3 Hz
            # carrier estimate required by our BPSK31 demodulator. A strong
            # squared-analytic spectral line can propose a bounded correction;
            # only the existing coherent varicode decoder may publish text.
            refined_carrier = carrier_hz
            carrier_refinement = None
            if mode == "bpsk31":
                from .psk31_carrier_acquisition import propose_bpsk31_carrier
                carrier_refinement = propose_bpsk31_carrier(
                    mode_values, sample_rate, carrier_hz)
                if carrier_refinement is not None:
                    refined_carrier = carrier_refinement["proposed_carrier_hz"]
            # A four-second crop can contain an entire 1.5-second RTTY or
            # PSK31 message *between* two truncated messages. Whole-crop
            # decoders can then fail acquisition even though the signal is
            # strong. Look for low-energy inter-transmission gaps first.
            # Only complete interior spans qualify; no signal generator truth
            # or test-specific message length is used at runtime.
            windows = propose_burst_windows(mode_values, sample_rate, refined_carrier, mode)
            acquisition_attempts = []
            result = None
            grade, text = "none", ""
            selected_trial = -1
            extended_cw = mode == "cw" and len(mode_values) > MAX_WINDOW_SECONDS * sample_rate
            # CW: compare an 8s history with the familiar recent 4s slice.
            # The larger window can contain an earlier partial symbol and
            # misleading noise; never force it over a cleaner short decode.
            # Neither slice is a verified Morse message or packet boundary.
            if extended_cw:
                trials = ((0, len(mode_values)),
                          (len(mode_values)-MAX_WINDOW_SECONDS*sample_rate, len(mode_values)))
            else:
                trials = (*windows, (0, len(mode_values)))
            best_quality = (-1, -1.0, -1.0)
            # Other modes retain up to two energy-gap candidates followed by
            # one full-window fallback. No extra workers or queued jobs.
            for begin, end in trials:
                proposed = _worker_library.decode(
                    mode, DecodeWindow(
                        mode_values[begin:end], sample_rate,
                        sample_start=mode_start + int(begin),
                        carrier_hz=(1200.0 if mode == "ax25" else refined_carrier),
                    ))
                candidate_grade, candidate_text = _evidence_grade(proposed)
                acquisition_attempts.append({
                    "sample_start": mode_start + int(begin),
                    "sample_count": int(end - begin),
                    "kind": ("cw-recent-four-second-check" if extended_cw and begin > 0
                             else "energy-gap-segment" if (begin, end) != (0, len(mode_values))
                             else "full-window"),
                    "status": proposed.status,
                    "grade": candidate_grade,
                })
                if extended_cw:
                    evidence = proposed.evidence
                    quality = (
                        1 if candidate_grade != "none" else 0,
                        float(evidence.get("mean_symbol_timing_fit", -1)),
                        1 - float(evidence.get("unknown_token_fraction", 1)),
                    )
                    if quality > best_quality:
                        result, grade, text = proposed, candidate_grade, candidate_text
                        selected_trial = len(acquisition_attempts)-1
                        best_quality = quality
                else:
                    result, grade, text = proposed, candidate_grade, candidate_text
                    selected_trial = len(acquisition_attempts)-1
                    if grade != "none":
                        break
            assert result is not None and selected_trial >= 0
            acquisition_attempts[selected_trial]["selected"] = True
            result_dict = result.to_dict()
            result_dict["live_acquisition"] = {
                "strategy": ("cw-long-or-recent-window-v2" if mode == "cw" and len(mode_values) > MAX_WINDOW_SECONDS * sample_rate
                             else "bounded-energy-gap-v1"),
                "original_sample_start": mode_start,
                "original_sample_count": len(mode_values),
                "selected_sample_start": acquisition_attempts[selected_trial]["sample_start"],
                "selected_sample_count": acquisition_attempts[selected_trial]["sample_count"],
                "selected_attempt_index": selected_trial,
                "attempts": acquisition_attempts,
                "candidate_spans": len(windows),
                "carrier_refinement": carrier_refinement,
                "confirmed_identity": False,
            }
            item = {
                "mode": mode, "status": result.status, "grade": grade,
                "text": text, "result": result_dict,
                "acquisition": result_dict["live_acquisition"],
                "wall_ms": round((time.perf_counter() - started) * 1000, 2),
                "cpu_ms": round((time.process_time() - cpu_started) * 1000, 2),
            }
        except Exception as exc:
            item = {
                "mode": mode, "status": "failed", "grade": "none", "text": "",
                "error": str(exc)[:200],
                "wall_ms": round((time.perf_counter() - started) * 1000, 2),
                "cpu_ms": round((time.process_time() - cpu_started) * 1000, 2),
            }
        attempts.append(item)
        rank = {"none": 0, "coherent-unverified": 1, "frame-integrity-passed": 2}
        if best is None or rank[item["grade"]] > rank[best["grade"]]:
            best = item
        if item["grade"] == "frame-integrity-passed":
            break
    assert best is not None
    return {
        "attempted": attempts, "selected_mode": best["mode"],
        "grade": best["grade"], "text": best["text"],
        "decoder_result": best.get("result"),
        "cpu_ms": round((time.process_time() - start_cpu) * 1000, 2),
        "wall_ms": round((time.perf_counter() - start_wall) * 1000, 2),
    }


class NativeLiveRunner:
    """At most one in-flight child process; skip overload, never queue a backlog."""

    def __init__(self, *, executor: Executor | None = None,
                 max_attempts_per_minute: int = MAX_ATTEMPTS_PER_MINUTE,
                 max_cpu_ms_per_minute: int = MAX_CPU_MS_PER_MINUTE) -> None:
        self._own_executor = executor is None
        self._executor = executor or ProcessPoolExecutor(
            max_workers=1, mp_context=mp.get_context("spawn"))
        self._lock = Lock()
        self._session: str | None = None
        self._active = False
        self._last_started = -1e9
        self._recent_tracks: dict[str, float] = {}
        self._cw_streams: dict[str, CWTokenStream] = {}
        self._modem_streams: dict[tuple[str, str], ModemEvidenceStream] = {}
        self._attempts: deque[tuple[float, float]] = deque(maxlen=64)
        self._provisional_probes: deque[float] = deque(maxlen=16)
        self._results: deque[dict[str, Any]] = deque(maxlen=MAX_RECENT_RESULTS)
        self._counts: Counter[str] = Counter()
        self._latencies: dict[str, deque[float]] = {}
        self._max_attempts = max_attempts_per_minute
        # Preserve intentionally lower constructor-specified test/operator
        # limits. Default capacity may be larger only for narrow CW-first
        # streams, and the shared process-CPU cap never changes.
        self._max_cw_attempts = (MAX_CW_STREAM_ATTEMPTS_PER_MINUTE
                                 if max_attempts_per_minute == MAX_ATTEMPTS_PER_MINUTE
                                 else max_attempts_per_minute)
        self._max_modem_attempts = (MAX_MODEM_FOLLOWUP_ATTEMPTS_PER_MINUTE
                                    if max_attempts_per_minute == MAX_ATTEMPTS_PER_MINUTE
                                    else max_attempts_per_minute)
        self._max_cpu_ms = max_cpu_ms_per_minute

    def close(self) -> None:
        if self._own_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _reset_for_session(self, session: str) -> None:
        if self._session != session:
            self._session = session
            self._recent_tracks.clear()
            self._cw_streams.clear()
            self._modem_streams.clear()
            self._attempts.clear()
            self._provisional_probes.clear()
            self._results.clear()
            self._last_started = -1e9

    def _prune(self, now: float) -> None:
        while self._attempts and now - self._attempts[0][0] >= 60:
            self._attempts.popleft()
        while self._provisional_probes and now - self._provisional_probes[0] >= 60:
            self._provisional_probes.popleft()
        for key, timestamp in list(self._recent_tracks.items()):
            if now - timestamp > TRACK_COOLDOWN_SECONDS * 2:
                del self._recent_tracks[key]
        if len(self._cw_streams) > MAX_CW_STREAM_TRACKS:
            for key in list(self._cw_streams)[:-MAX_CW_STREAM_TRACKS]:
                del self._cw_streams[key]
        if len(self._modem_streams) > MAX_MODEM_STREAM_TRACKS:
            for key in list(self._modem_streams)[:-MAX_MODEM_STREAM_TRACKS]:
                del self._modem_streams[key]

    def submit(self, *, session_id: str, track: dict[str, Any],
               snapshot: dict[str, Any], store: Any) -> dict[str, Any]:
        """Called by live CV/heuristic poll; only copies one bounded PCM crop."""
        track_id = str(track.get("track_id") or "")
        if not track_id.startswith(session_id + ":"):
            return {"status": "not_eligible", "reason": "invalid-track-session"}
        rate = int(snapshot["sample_rate"])
        persistent = track.get("track_state") == "persistent"
        probe_kind = None if persistent else provisional_probe_kind(track, rate)
        if not persistent and probe_kind is None:
            return {"status": "not_eligible", "reason": "no-stable-signal-or-hypothesis"}
        ring_start = int(snapshot["sample_offset"])
        ring_end = ring_start + int(snapshot["window_sample_count"])
        if (snapshot.get("session_id") != session_id or not snapshot.get("active")
                or rate < 8000 or rate > 192000):
            return {"status": "not_eligible"}
        start = int(track.get("sample_start") or 0)
        end = start + int(track.get("sample_count") or 0)
        if start < 0 or end <= start or end > ring_end:
            return {"status": "not_eligible"}
        # CW can need several complete words to overcome a cropped initial
        # Morse symbol. Keep the longer lookback restricted to CW hypotheses,
        # with all other modems still limited to four-second DSP windows.
        modes = candidate_modes(track)[:MAX_MODES_PER_ATTEMPT]
        # Only narrow CW-first tracks get frequent incremental updates.
        # Provisional tracks have an additional hard trial quota below.
        cw_stream = modes[0] == "cw" and (
            float(track.get("frequency_high_hz") or 0)
            - float(track.get("frequency_low_hz") or 0)) <= 80.0
        modem_followup = (persistent and modes[0] in SUPPORTED_MODES
                          and explicit_modem_hypothesis(track) == modes[0])
        max_seconds = MAX_CW_WINDOW_SECONDS if modes[0] == "cw" else MAX_WINDOW_SECONDS
        left = max(ring_start, start - round(.15 * rate))
        right = min(ring_end, end + round(.15 * rate))
        if right - left > max_seconds * rate:
            left = right - max_seconds * rate
        # Bell 202 AX.25 UI packets may be shorter than 400 ms. The AX.25
        # decoder accepts 125-ms input but publishes only CRC-passed frames.
        min_window_seconds = .125 if modes[0] == "ax25" else .4
        if right - left < min_window_seconds * rate:
            return {"status": "not_eligible"}
        lo = float(track.get("frequency_low_hz") or 0)
        hi = float(track.get("frequency_high_hz") or 0)
        if not (0 <= lo < hi <= min(3000.0, rate / 2)):
            return {"status": "not_eligible"}
        carrier = (lo + hi) / 2
        now = time.monotonic()
        with self._lock:
            self._reset_for_session(session_id)
            self._prune(now)
            if self._active:
                self._counts["skipped_busy"] += 1
                return {"status": "skipped_busy"}
            if now - self._last_started < MIN_ATTEMPT_GAP_SECONDS:
                self._counts["skipped_cooldown"] += 1
                return {"status": "skipped_cooldown"}
            cooldown = (CW_STREAM_TRACK_COOLDOWN_SECONDS if cw_stream
                        else MODEM_FOLLOWUP_TRACK_COOLDOWN_SECONDS if modem_followup
                        else TRACK_COOLDOWN_SECONDS)
            if now - self._recent_tracks.get(track_id, -1e9) < cooldown:
                self._counts["skipped_track"] += 1
                return {"status": "skipped_track"}
            attempt_limit = (self._max_cw_attempts if cw_stream
                             else self._max_modem_attempts if modem_followup
                             else self._max_attempts)
            if (len(self._attempts) >= attempt_limit
                    or sum(used for _, used in self._attempts) >= self._max_cpu_ms):
                self._counts["skipped_budget"] += 1
                return {"status": "skipped_budget"}
            if probe_kind and len(self._provisional_probes) >= MAX_PROVISIONAL_PROBES_PER_MINUTE:
                self._counts["skipped_provisional_budget"] += 1
                return {"status": "skipped_provisional_budget"}
            channels = int(snapshot["channels"])
            pcm = np.frombuffer(snapshot["pcm"], dtype="<i2").reshape(-1, channels)
            selected = pcm[left - ring_start:right - ring_start]
            if len(selected) != right - left:
                return {"status": "not_eligible"}
            if channels == 1:
                packed = selected.astype("<i2", copy=False).tobytes()
            else:
                packed = np.clip(np.rint(selected.astype(np.float32).mean(axis=1)),
                                 -32768, 32767).astype("<i2").tobytes()
            self._active = True
            self._recent_tracks[track_id] = now
            self._last_started = now
            self._attempts.append((now, 0.0))
            if probe_kind:
                self._provisional_probes.append(now)
                self._counts["admitted_provisional"] += 1
            if modem_followup:
                self._counts["admitted_modem_followup"] += 1
            self._counts["admitted"] += 1
        try:
            run_id = store.create_decoder_run(
                session_id=session_id, track_id=track_id,
                sample_start=left, sample_count=right - left, sample_rate=rate,
                modulation_hypothesis=str(
                    (track.get("classifier_proposal") or track.get("model_proposal") or {})
                    .get("family") or "unknown")[:80],
                backend="sigdecode-live-auto", modes=list(modes),
                attempted_modes=[], current_mode=modes[0], output_offset=0,
                status="queued", note="Native experimental decoding on a bounded live detector window.",
                carrier_hz=carrier, output_scope="isolated_native_live_window",
                association="live_detector_track_tentative",
                classifier_evidence=(track.get("classifier_proposal")
                                     or track.get("model_proposal") or {}),
                detector_evidence={
                    "source": track.get("detector_source", "live_cv"),
                    "track_id": track_id, "sample_start": start,
                    "sample_count": end - start,
                    "frequency_low_hz": lo, "frequency_high_hz": hi,
                    "confidence": track.get("confidence"),
                    "track_state_at_admission": track.get("track_state"),
                    "provisional_probe_kind": probe_kind,
                    "track_observations": track.get("track_observations"),
                },
            )
            # The explicit Morse hypothesis may be an accepted prediction or
            # a strong *relative ranking*. Never interpret rank as verified.
            keying = cw_stream and _explicit_cw_hypothesis(track)
            worker_args = (packed, rate, left, carrier, tuple(modes))
            if keying:
                worker_args += ((session_id, track_id),)
            future = self._executor.submit(decode_live_candidate, *worker_args)
            future.add_done_callback(lambda done: self._complete(
                done, session_id, track_id, run_id, left, rate, modes, store,
                carrier, cw_stream))
            return {"status": "queued", "run_id": run_id, "track_id": track_id,
                    "modes": list(modes), "admission": probe_kind or "persistent",
                    "modem_followup": modem_followup}
        except Exception:
            with self._lock:
                self._active = False
                self._counts["failed_submission"] += 1
            raise

    def _complete(self, future: Future, session_id: str, track_id: str,
                  run_id: str, sample_start: int, rate: int,
                  modes: tuple[str, ...], store: Any,
                  carrier_hz: float = 0.0, cw_stream: bool = False) -> None:
        try:
            result = future.result()
            attempted = result["attempted"]
            grade = result["grade"]
            text = result["text"]
            selected_mode = result["selected_mode"]
            stream_event = None
            if (cw_stream and selected_mode == "cw"
                    and result.get("decoder_result") is not None):
                stream_event = result["decoder_result"].get("cw_stream")
                if stream_event is None:
                    with self._lock:
                        if self._session == session_id:
                            existing = self._cw_streams.get(track_id)
                            if (existing is None or existing.sample_rate != rate
                                    or abs(existing.carrier_hz-carrier_hz) > 8):
                                existing = CWTokenStream(session_id, track_id, rate, carrier_hz)
                                self._cw_streams[track_id] = existing
                            try:
                                stream_event = existing.feed(
                                    result["decoder_result"], sample_rate=rate,
                                    carrier_hz=carrier_hz)
                            except ValueError:
                                # A detector gap or retune must reset, never
                                # splice unrelated CW fragments into one sentence.
                                existing = CWTokenStream(session_id, track_id, rate, carrier_hz)
                                self._cw_streams[track_id] = existing
                                stream_event = existing.feed(
                                    result["decoder_result"], sample_rate=rate,
                                    carrier_hz=carrier_hz)
            modem_event = None
            if (selected_mode in SUPPORTED_MODES
                    and result.get("decoder_result") is not None):
                # Window demodulation is unchanged. Publish only new sample-
                # indexed text tokens or CRC-passed frames after overlap checks.
                raw = result["decoder_result"]
                acquisition = raw.get("live_acquisition") or {}
                selected_start = int(acquisition.get("selected_sample_start", sample_start))
                selected_count = int(acquisition.get("selected_sample_count", 0))
                if selected_count > 0:
                    key = (track_id, selected_mode)
                    with self._lock:
                        if self._session == session_id:
                            modem_state = self._modem_streams.get(key)
                            if (modem_state is None or not modem_state.compatible(
                                    session_id=session_id, track_id=track_id,
                                    mode=selected_mode, sample_rate=rate,
                                    carrier_hz=carrier_hz)):
                                modem_state = ModemEvidenceStream(
                                    session_id, track_id, selected_mode, rate, carrier_hz)
                                self._modem_streams[key] = modem_state
                            modem_event = modem_state.feed(
                                raw, sample_start=selected_start,
                                sample_count=selected_count, grade=grade,
                                carrier_hz=carrier_hz)
                    if modem_event is not None:
                        text = modem_event["delta_text"]
                        grade = modem_event["grade"]
            if stream_event is not None:
                text = stream_event["delta_text"]
                grade = "coherent-unverified" if text else "none"
            note = ("Decoder frame passed integrity check (not a confirmed mode label)."
                    if grade == "frame-integrity-passed" else
                    "Readable experimental decode; identity unverified."
                    if grade == "coherent-unverified" else
                    "Native decoder candidates abstained or produced no coherent result.")
            if result["decoder_result"] is not None:
                evidence = result["decoder_result"]
                if stream_event is not None:
                    evidence["cw_stream"] = stream_event
                if modem_event is not None:
                    evidence["modem_stream"] = modem_event
                evidence["live_assistance"] = {
                    "grade": grade, "classification_suggestion": selected_mode if grade != "none" else None,
                    "confirmed_identity": False,
                    "sample_start": sample_start, "original_sample_rate": rate,
                    "cpu_ms": result["cpu_ms"], "wall_ms": result["wall_ms"],
                    "mode_timings": [
                        {"mode": a["mode"], "wall_ms": a["wall_ms"], "cpu_ms": a["cpu_ms"],
                         "status": a["status"], "grade": a["grade"],
                         "acquisition": a.get("acquisition")}
                        for a in attempted
                    ],
                }
                store.save_decoder_result(run_id, evidence)
            store.update_decoder_run(run_id, {
                "attempted_modes": [a["mode"] for a in attempted],
                "current_mode": selected_mode, "output_offset": 0,
                "output_end": len(text.encode("utf-8")),
                "text": text, "status": "tentative_output" if grade != "none" else "completed_no_text",
                "note": note,
            })
            event = {
                "run_id": run_id, "session_id": session_id, "track_id": track_id,
                "mode": selected_mode if grade != "none" else None,
                "attempted_modes": [a["mode"] for a in attempted],
                "grade": grade, "text": text, "cpu_ms": result["cpu_ms"],
                "wall_ms": result["wall_ms"], "sample_start": sample_start,
                "confirmed_identity": False,
            }
            if stream_event is not None:
                event["cw_stream_delta_text"] = stream_event["delta_text"]
                event["cw_stream_cumulative_text"] = stream_event["cumulative_text"]
                event["cw_stream_reason"] = stream_event["reason"]
            if modem_event is not None:
                event["modem_stream_delta_text"] = modem_event["delta_text"]
                event["modem_stream_cumulative_text"] = modem_event["cumulative_text"]
                event["modem_stream_new_items"] = modem_event["new_items"]
                event["modem_stream_suppressed_overlaps"] = modem_event["suppressed_overlap_items"]
        except Exception as exc:
            try:
                store.update_decoder_run(run_id, {
                    "attempted_modes": list(modes), "current_mode": modes[0],
                    "status": "failed", "text": "", "note": str(exc)[:300],
                })
            except Exception:
                pass
            event = {"run_id": run_id, "session_id": session_id, "track_id": track_id,
                     "grade": "none", "text": "", "error": str(exc)[:200],
                     "cpu_ms": 0, "wall_ms": 0, "confirmed_identity": False}
        with self._lock:
            self._active = False
            self._counts["finished"] += 1
            if "error" in event:
                self._counts["failed"] += 1
            elif event["grade"] != "none":
                self._counts["coherent"] += 1
            # Account true child process CPU time toward the rolling budget.
            # Session rollover can clear the deque while an older task ends.
            if session_id == self._session and self._attempts:
                last_time, _ = self._attempts[-1]
                self._attempts[-1] = (last_time, float(event["cpu_ms"]))
            if session_id == self._session:
                self._results.append(event)
                for item in result.get("attempted", []) if "error" not in event else []:
                    self._latencies.setdefault(item["mode"], deque(maxlen=100)).append(
                        item["wall_ms"])

    def status(self, session_id: str) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            self._reset_for_session(session_id)
            self._prune(now)
            timings = {
                mode: {"count": len(values), "median_ms": round(statistics.median(values), 2),
                       "p95_ms": round(sorted(values)[min(len(values)-1,
                           math.ceil(.95 * len(values))-1)], 2)}
                for mode, values in self._latencies.items() if values
            }
            return {
                "session_id": session_id, "busy": self._active,
                "attempts_last_minute": len(self._attempts),
                "provisional_probes_last_minute": len(self._provisional_probes),
                "max_provisional_probes_per_minute": MAX_PROVISIONAL_PROBES_PER_MINUTE,
                "cpu_ms_last_minute": round(sum(cost for _, cost in self._attempts), 2),
                "cpu_pct_single_core_last_minute": round(
                    sum(cost for _, cost in self._attempts) / 600, 2),
                "max_attempts_per_minute": self._max_attempts,
                "max_cw_stream_attempts_per_minute": self._max_cw_attempts,
                "max_modem_followup_attempts_per_minute": self._max_modem_attempts,
                "max_cpu_ms_per_minute": self._max_cpu_ms,
                "cw_stream_tracks": len(self._cw_streams),
                "modem_stream_tracks": len(self._modem_streams),
                "modem_stream_units": sum(x.emitted_units for x in self._modem_streams.values()),
                "modem_stream_frames": sum(x.emitted_frames for x in self._modem_streams.values()),
                "counts": dict(self._counts),
                "latency_by_mode": timings,
                "recent": [dict(x) for x in reversed(self._results)
                           if x["session_id"] == session_id][:12],
                "unverified_by_default": True,
            }

    def for_track(self, session_id: str, track_id: str) -> dict[str, Any] | None:
        with self._lock:
            if session_id != self._session:
                return None
            return next((dict(x) for x in reversed(self._results)
                         if x["track_id"] == track_id and x["session_id"] == session_id),
                        None)
