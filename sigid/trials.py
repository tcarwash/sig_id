"""Bounded, receive-only Fldigi mode trials for live provisional candidates."""

from __future__ import annotations

from collections import deque
from threading import RLock
import time
from typing import Any

from .decoder_backend import TextReceiveBackend


VALIDATED_TRIAL_MODES = ("BPSK31", "RTTY", "THOR16")
MODULATION_TRIAL_ORDER = {
    "FSK-like": ("RTTY", "THOR16", "BPSK31"),
    "MFSK-like": ("THOR16", "RTTY", "BPSK31"),
    "single-tone / unknown": VALIDATED_TRIAL_MODES,
}


class DecoderTrialRunner:
    """Advance a small tested-mode list against a receive backend.

    The runner cannot establish that audio routing is correct. New receive bytes
    end a trial sweep as tentative output and are never treated as ground truth.
    """

    def __init__(
        self,
        client: TextReceiveBackend,
        *,
        dwell_seconds: float = 6.0,
        max_trials_per_candidate: int = 3,
        max_starts_per_minute: int = 6,
        clock: Any = time.monotonic,
    ) -> None:
        if dwell_seconds <= 0 or max_trials_per_candidate < 1 or max_starts_per_minute < 1:
            raise ValueError("dwell and trial limits must be positive")
        self.client = client
        self.dwell_seconds = dwell_seconds
        self.max_trials_per_candidate = max_trials_per_candidate
        self.max_starts_per_minute = max_starts_per_minute
        self.clock = clock
        self._lock = RLock()
        self._starts: deque[float] = deque()
        self._tried_tracks: dict[str, float] = {}
        self._state: dict[str, Any] = {"status": "idle"}

    def start(self, *, track_id: str, modulation_hypothesis: str, live: bool) -> dict[str, Any]:
        if not live:
            raise ValueError("automatic decoder trials require an active live capture")
        if not track_id or len(track_id) > 128:
            raise ValueError("track_id is required")
        proposed = MODULATION_TRIAL_ORDER.get(modulation_hypothesis, VALIDATED_TRIAL_MODES)
        with self._lock:
            now = self.clock()
            while self._starts and now - self._starts[0] >= 60.0:
                self._starts.popleft()
            if self._state.get("status") == "running":
                raise ValueError("a decoder trial is already running")
            if track_id in self._tried_tracks:
                raise ValueError("this live track has already had its automatic trial")
            if len(self._starts) >= self.max_starts_per_minute:
                raise ValueError("automatic decoder trial rate limit reached")

            catalog = set(self.client.mode_names())
            modes = [mode for mode in proposed if mode in catalog][: self.max_trials_per_candidate]
            if not modes:
                raise ValueError("no loopback-tested trial modes are present in the Fldigi catalog")
            selected = self.client.select_mode(modes[0])
            if selected != modes[0]:
                raise RuntimeError(f"Fldigi selected {selected!r} instead of {modes[0]!r}")
            baseline = self.client.receive_text_length()
            self._starts.append(now)
            self._tried_tracks[track_id] = now
            if len(self._tried_tracks) > 1024:
                oldest = next(iter(self._tried_tracks))
                del self._tried_tracks[oldest]
            self._state = {
                "status": "running",
                "track_id": track_id,
                "modulation_hypothesis": modulation_hypothesis,
                "modes": modes,
                "attempted_modes": [selected],
                "current_mode": selected,
                "mode_started_at": now,
                "output_offset": baseline,
                "text": "",
                "note": "Tentative receive trial; verify the audio route and review decoded text.",
            }
            return dict(self._state)

    def advance(self) -> dict[str, Any]:
        with self._lock:
            state = self._state
            if state.get("status") != "running":
                return dict(state)
            end = int(self.client.receive_text_length())
            offset = int(state["output_offset"])
            if end > offset:
                raw = self.client.receive_text(start=offset, length=min(end - offset, 8192))
                state["status"] = "tentative_output"
                state["text"] = raw.decode("utf-8", errors="replace")
                state["output_end"] = offset + len(raw)
                state["note"] = "New Fldigi text observed. Review it; no protocol label was assigned."
                return dict(state)

            now = self.clock()
            if now - float(state["mode_started_at"]) < self.dwell_seconds:
                return dict(state)
            remaining = [mode for mode in state["modes"] if mode not in state["attempted_modes"]]
            if not remaining:
                state["status"] = "exhausted"
                state["note"] = "No receive text observed in the bounded tested-mode sweep."
                return dict(state)

            next_mode = remaining[0]
            selected = self.client.select_mode(next_mode)
            if selected != next_mode:
                state["status"] = "failed"
                state["note"] = f"Fldigi selected {selected!r} instead of {next_mode!r}."
                return dict(state)
            state["current_mode"] = selected
            state["attempted_modes"].append(selected)
            state["mode_started_at"] = now
            state["output_offset"] = int(self.client.receive_text_length())
            return dict(state)

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            if self._state.get("status") == "running":
                self._state["status"] = "cancelled"
                self._state["note"] = "Decoder trial stopped by manual control."
            return dict(self._state)


# Kept as an alias for callers that used the original Fldigi-specific name.
FldigiTrialRunner = DecoderTrialRunner
