"""Bounded association of provisional detections across live analysis windows."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(slots=True)
class _Track:
    track_id: str
    center_hz: float
    bandwidth_hz: float
    first_sample: int
    last_sample: int
    observations: int = 1


class LiveTrackRegistry:
    """Assign stable IDs to nearby detections and expire stale tracks.

    Confirmation means only that the same spectral candidate persisted across
    analysis polls. It is not a verified modulation or protocol identification.
    The registry is deliberately per capture session and capped in size.
    """

    def __init__(
        self,
        session_id: str,
        *,
        max_tracks: int = 64,
        expiry_seconds: float = 5.0,
        confirmation_observations: int = 3,
    ) -> None:
        if not session_id or max_tracks < 1 or expiry_seconds <= 0 or confirmation_observations < 1:
            raise ValueError("session_id, positive max_tracks/expiry_seconds and confirmation count are required")
        self.session_id = session_id
        self.max_tracks = max_tracks
        self.expiry_seconds = expiry_seconds
        self.confirmation_observations = confirmation_observations
        self._tracks: dict[str, _Track] = {}
        self._next_id = 1
        self._lock = RLock()

    def update(
        self,
        detections: list[dict[str, Any]],
        *,
        sample_position: int,
        sample_rate: int,
    ) -> list[dict[str, Any]]:
        if sample_position < 0 or sample_rate <= 0:
            raise ValueError("sample_position must be non-negative and sample_rate positive")
        if len(detections) > self.max_tracks:
            detections = sorted(detections, key=lambda item: float(item.get("peak_snr_db", 0)), reverse=True)[: self.max_tracks]

        with self._lock:
            expiry_samples = int(self.expiry_seconds * sample_rate)
            expired = [
                track_id
                for track_id, track in self._tracks.items()
                if sample_position - track.last_sample > expiry_samples
            ]
            for track_id in expired:
                del self._tracks[track_id]

            available = set(self._tracks)
            observed_this_update: set[str] = set()
            result: list[dict[str, Any]] = []
            for detection in sorted(detections, key=lambda item: float(item.get("center_hz", 0))):
                center = float(detection["center_hz"])
                bandwidth = max(1.0, float(detection["bandwidth_hz"]))
                match_id = None
                match_cost = float("inf")
                for track_id in available:
                    track = self._tracks[track_id]
                    tolerance = max(150.0, 1.5 * max(track.bandwidth_hz, bandwidth))
                    delta = abs(center - track.center_hz)
                    if delta <= tolerance and delta < match_cost:
                        match_id, match_cost = track_id, delta

                if match_id is None:
                    if len(self._tracks) >= self.max_tracks:
                        evictable = [track for track in self._tracks.values() if track.track_id not in observed_this_update]
                        if not evictable:
                            continue
                        weakest = min(evictable, key=lambda item: item.last_sample)
                        del self._tracks[weakest.track_id]
                        available.discard(weakest.track_id)
                    match_id = f"{self.session_id}:{self._next_id:04d}"
                    self._next_id += 1
                    self._tracks[match_id] = _Track(match_id, center, bandwidth, sample_position, sample_position)
                else:
                    available.remove(match_id)
                    track = self._tracks[match_id]
                    track.center_hz = center
                    track.bandwidth_hz = bandwidth
                    if sample_position > track.last_sample:
                        track.last_sample = sample_position
                        track.observations += 1

                track = self._tracks[match_id]
                observed_this_update.add(match_id)
                item = dict(detection)
                item["track_id"] = match_id
                item["track_observations"] = track.observations
                item["track_state"] = (
                    "persistent"
                    if track.observations >= self.confirmation_observations
                    else "provisional"
                )
                item["track_age_seconds"] = (sample_position - track.first_sample) / sample_rate
                result.append(item)
            return result
