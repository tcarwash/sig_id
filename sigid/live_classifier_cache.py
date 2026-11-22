"""Bounded, sample-provenance-aware memoization of live CV classifier evidence.

The cache is deliberately *not* a classifier. A reused result remains tied to
the original PCM crop, model identity and frequency region. This reduces
expensive repeat scoring of a persistent track while keeping the last
classification clearly identified as stale/reused rather than new evidence.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from typing import Any

MAX_TRACKS = 128
REFRESH_SECONDS = 6.0
MAX_FREQUENCY_DRIFT_HZ = 50.0


@dataclass(frozen=True)
class CachedClassification:
    model_id: str
    sample_rate: int
    scored_at_sample: int
    low_hz: float
    high_hz: float
    proposal: dict[str, Any]


class LiveClassifierEvidenceCache:
    def __init__(self, *, refresh_seconds: float = REFRESH_SECONDS,
                 max_tracks: int = MAX_TRACKS):
        if refresh_seconds <= 0 or not 1 <= max_tracks <= 1024:
            raise ValueError("invalid classifier cache limits")
        self._refresh_seconds = refresh_seconds
        self._max_tracks = max_tracks
        self._lock = Lock()
        self._session: str | None = None
        self._entries: OrderedDict[str, CachedClassification] = OrderedDict()
        self._scored = 0
        self._reused = 0

    def _reset(self, session_id: str):
        if session_id != self._session:
            self._session = session_id
            self._entries.clear()
            self._scored = self._reused = 0

    def lookup(self, *, session_id: str, track_id: str, model_id: str,
               sample_rate: int, sample_position: int,
               low_hz: float, high_hz: float) -> dict[str, Any] | None:
        with self._lock:
            self._reset(session_id)
            cached = self._entries.get(track_id)
            if cached is None or cached.model_id != model_id:
                return None
            if cached.sample_rate != sample_rate:
                return None
            age = (sample_position - cached.scored_at_sample) / sample_rate
            if not 0 <= age < self._refresh_seconds:
                return None
            # A CV box can drift slightly as its bounds improve. Large
            # frequency/width changes must cause a fresh classification.
            old_width = cached.high_hz - cached.low_hz
            new_width = high_hz - low_hz
            drift = min(MAX_FREQUENCY_DRIFT_HZ, max(12.0, 0.1 * old_width))
            if (abs((low_hz + high_hz) - (cached.low_hz + cached.high_hz)) / 2 > drift
                    or abs(new_width - old_width) > max(25.0, 0.2 * old_width)):
                return None
            self._entries.move_to_end(track_id)
            self._reused += 1
            result = deepcopy(cached.proposal)
            result["reuse_info"] = {
                "reused": True,
                "evidence_age_seconds": round(age, 3),
                "original_sample_start": cached.proposal.get("sample_start"),
                "original_sample_count": cached.proposal.get("sample_count"),
                "note": "Reused classifier evidence on a persistent live track; not a new observation.",
            }
            return result

    def remember(self, *, session_id: str, track_id: str, model_id: str,
                 sample_rate: int, sample_position: int,
                 low_hz: float, high_hz: float, proposal: dict[str, Any]) -> None:
        with self._lock:
            self._reset(session_id)
            self._scored += 1
            if not track_id or not proposal.get("available"):
                return
            self._entries[track_id] = CachedClassification(
                model_id=model_id, sample_rate=sample_rate,
                scored_at_sample=sample_position, low_hz=low_hz,
                high_hz=high_hz, proposal=deepcopy(proposal),
            )
            self._entries.move_to_end(track_id)
            while len(self._entries) > self._max_tracks:
                self._entries.popitem(last=False)

    def status(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            # A read-only status query for a stale/unknown session must not
            # erase the active recording's expensive classifier evidence.
            active = session_id == self._session
            scored = self._scored if active else 0
            reused = self._reused if active else 0
            total = scored + reused
            return {
                "session_id": session_id, "scored": scored,
                "reused": reused, "entries": len(self._entries) if active else 0,
                "reuse_fraction": round(reused / total, 3) if total else 0.0,
                "refresh_seconds": self._refresh_seconds,
                "max_entries": self._max_tracks,
            }
