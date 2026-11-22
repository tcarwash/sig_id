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

MAX_WINDOW_SECONDS = 4
MAX_MODES_PER_ATTEMPT = 2
MAX_ATTEMPTS_PER_MINUTE = 5
MAX_CPU_MS_PER_MINUTE = 3000
TRACK_COOLDOWN_SECONDS = 24
MIN_ATTEMPT_GAP_SECONDS = 3
MAX_RECENT_RESULTS = 40
MAX_TEXT_LENGTH = 2048


def candidate_modes(track: dict[str, Any]) -> tuple[str, ...]:
    """Classifier and spectrum propose *hypotheses*, not proven identities."""
    proposal = track.get("classifier_proposal") or track.get("model_proposal") or {}
    if proposal.get("abstained") or not proposal.get("available"):
        proposal = {}
    identity = " ".join(str(proposal.get(k) or "").lower()
                        for k in ("method", "method_id", "family", "top_candidate"))
    low = float(track.get("frequency_low_hz", 0.0))
    high = float(track.get("frequency_high_hz", 0.0))
    width = high - low
    center = (low + high) / 2

    if "ax.25" in identity or "ax25" in identity or "aprs" in identity:
        return ("ax25",)
    if "rtty" in identity or "baudot" in identity:
        return ("rtty",)
    if "bpsk31" in identity or "psk31" in identity:
        return ("bpsk31",)
    if "morse" in identity or " cw " in f" {identity} ":
        return ("cw",)
    if "psk" in identity:
        return ("bpsk31", "cw")
    if "fsk" in identity:
        return ("rtty", "ax25")

    # Fallback even if the classifier abstained. No spurious model confidence
    # is generated: these are frequency/width-based experimental suggestions.
    if width > 600:
        return ("ax25", "rtty")
    if width > 170:
        return ("rtty", "bpsk31")
    if center >= 1600:
        return ("bpsk31", "cw")
    return ("cw", "rtty")


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
                          carrier_hz: float, modes: tuple[str, ...]) -> dict[str, Any]:
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
    if not len(values) or len(values) > MAX_WINDOW_SECONDS * sample_rate:
        raise ValueError("live decoder window must be bounded")
    if not np.isfinite(carrier_hz) or not 20 <= carrier_hz < sample_rate / 2:
        raise ValueError("invalid audio carrier")
    attempts = []
    best = None
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    for mode in modes:
        started = time.perf_counter()
        cpu_started = time.process_time()
        try:
            result = _worker_library.decode(
                mode, DecodeWindow(values, sample_rate, sample_start=sample_start,
                                   carrier_hz=(1200.0 if mode == "ax25" else carrier_hz)))
            grade, text = _evidence_grade(result)
            item = {
                "mode": mode, "status": result.status, "grade": grade,
                "text": text, "result": result.to_dict(),
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
        self._attempts: deque[tuple[float, float]] = deque(maxlen=64)
        self._results: deque[dict[str, Any]] = deque(maxlen=MAX_RECENT_RESULTS)
        self._counts: Counter[str] = Counter()
        self._latencies: dict[str, deque[float]] = {}
        self._max_attempts = max_attempts_per_minute
        self._max_cpu_ms = max_cpu_ms_per_minute

    def close(self) -> None:
        if self._own_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _reset_for_session(self, session: str) -> None:
        if self._session != session:
            self._session = session
            self._recent_tracks.clear()
            self._attempts.clear()
            self._results.clear()
            self._last_started = -1e9

    def _prune(self, now: float) -> None:
        while self._attempts and now - self._attempts[0][0] >= 60:
            self._attempts.popleft()
        for key, timestamp in list(self._recent_tracks.items()):
            if now - timestamp > TRACK_COOLDOWN_SECONDS * 2:
                del self._recent_tracks[key]

    def submit(self, *, session_id: str, track: dict[str, Any],
               snapshot: dict[str, Any], store: Any) -> dict[str, Any]:
        """Called by live CV/heuristic poll; only copies one bounded PCM crop."""
        track_id = str(track.get("track_id") or "")
        if (track.get("track_state") != "persistent"
                or not track_id.startswith(session_id + ":")):
            return {"status": "not_eligible"}
        rate = int(snapshot["sample_rate"])
        ring_start = int(snapshot["sample_offset"])
        ring_end = ring_start + int(snapshot["window_sample_count"])
        if (snapshot.get("session_id") != session_id or not snapshot.get("active")
                or rate < 8000 or rate > 192000):
            return {"status": "not_eligible"}
        start = int(track.get("sample_start") or 0)
        end = start + int(track.get("sample_count") or 0)
        if start < 0 or end <= start or end > ring_end:
            return {"status": "not_eligible"}
        # Add short guard intervals to retain modem acquisition/termination,
        # but never read outside the live ring or exceed 4 seconds.
        left = max(ring_start, start - round(.15 * rate))
        right = min(ring_end, end + round(.15 * rate))
        if right - left > MAX_WINDOW_SECONDS * rate:
            left = right - MAX_WINDOW_SECONDS * rate
        if right - left < .4 * rate:
            return {"status": "not_eligible"}
        lo = float(track.get("frequency_low_hz") or 0)
        hi = float(track.get("frequency_high_hz") or 0)
        if not (0 <= lo < hi <= min(3000.0, rate / 2)):
            return {"status": "not_eligible"}
        carrier = (lo + hi) / 2
        modes = candidate_modes(track)[:MAX_MODES_PER_ATTEMPT]
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
            if now - self._recent_tracks.get(track_id, -1e9) < TRACK_COOLDOWN_SECONDS:
                self._counts["skipped_track"] += 1
                return {"status": "skipped_track"}
            if (len(self._attempts) >= self._max_attempts
                    or sum(used for _, used in self._attempts) >= self._max_cpu_ms):
                self._counts["skipped_budget"] += 1
                return {"status": "skipped_budget"}
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
                },
            )
            future = self._executor.submit(
                decode_live_candidate, packed, rate, left, carrier, tuple(modes))
            future.add_done_callback(lambda done: self._complete(
                done, session_id, track_id, run_id, left, rate, modes, store))
            return {"status": "queued", "run_id": run_id, "track_id": track_id,
                    "modes": list(modes)}
        except Exception:
            with self._lock:
                self._active = False
                self._counts["failed_submission"] += 1
            raise

    def _complete(self, future: Future, session_id: str, track_id: str,
                  run_id: str, sample_start: int, rate: int,
                  modes: tuple[str, ...], store: Any) -> None:
        try:
            result = future.result()
            attempted = result["attempted"]
            grade = result["grade"]
            text = result["text"]
            selected_mode = result["selected_mode"]
            note = ("Decoder frame passed integrity check (not a confirmed mode label)."
                    if grade == "frame-integrity-passed" else
                    "Readable experimental decode; identity unverified."
                    if grade == "coherent-unverified" else
                    "Native decoder candidates abstained or produced no coherent result.")
            if result["decoder_result"] is not None:
                evidence = result["decoder_result"]
                evidence["live_assistance"] = {
                    "grade": grade, "classification_suggestion": selected_mode if grade != "none" else None,
                    "confirmed_identity": False,
                    "sample_start": sample_start, "original_sample_rate": rate,
                    "cpu_ms": result["cpu_ms"], "wall_ms": result["wall_ms"],
                    "mode_timings": [
                        {"mode": a["mode"], "wall_ms": a["wall_ms"], "cpu_ms": a["cpu_ms"],
                         "status": a["status"], "grade": a["grade"]}
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
                "cpu_ms_last_minute": round(sum(cost for _, cost in self._attempts), 2),
                "cpu_pct_single_core_last_minute": round(
                    sum(cost for _, cost in self._attempts) / 600, 2),
                "max_attempts_per_minute": self._max_attempts,
                "max_cpu_ms_per_minute": self._max_cpu_ms,
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
