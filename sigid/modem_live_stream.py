"""Bounded, sample-indexed receive-only event assembly for live modems.

The first-party RTTY, BPSK31, and AX.25 demodulators are currently bounded
window decoders. This assembler does NOT attempt to reconstruct symbols across
window boundaries. It suppresses duplicate *evidence* returned for overlapping
source intervals, while retaining repeated messages at distinct sample times.

Only an intact modem decode may contribute tokens, and only independently
CRC-passed frames may be counted as packet events. All evidence is tentative
with respect to signal/mode/station identity.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any
import math

SUPPORTED_MODES = frozenset({"rtty", "bpsk31", "ax25"})
MAX_RECENT_ITEMS = 160
MAX_DELTA_ITEMS = 32
MAX_CUMULATIVE_TEXT = 1024
MAX_FRAME_EVENTS = 32
MAX_WINDOW_SECONDS = 8
CARRIER_TOLERANCE_HZ = {"rtty": 25.0, "bpsk31": 12.0, "ax25": 80.0}


def _span(item: dict, lo: int, hi: int) -> tuple[int, int] | None:
    start, end = item.get("start_sample"), item.get("end_sample")
    if (type(start) is not int or type(end) is not int
            or not lo <= start < end <= hi):
        return None
    return start, end


def _same_occurrence(a: tuple[int, int], b: tuple[int, int],
                     sample_rate: int) -> bool:
    """Near-identical absolute time, not merely identical text or checksum."""
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if not overlap:
        return False
    shortest = min(a[1]-a[0], b[1]-b[0])
    mid_delta = abs(a[0]+a[1]-b[0]-b[1])/2
    return overlap >= .53*shortest and mid_delta <= max(.35*shortest, .035*sample_rate)


@dataclass
class ModemEvidenceStream:
    session_id: str
    track_id: str
    mode: str
    sample_rate: int
    carrier_hz: float
    cumulative_text: str = ""
    emitted_units: int = 0
    emitted_frames: int = 0
    _seen: deque = field(default_factory=lambda: deque(maxlen=MAX_RECENT_ITEMS), repr=False)
    _frames: deque = field(default_factory=lambda: deque(maxlen=MAX_RECENT_ITEMS), repr=False)
    _last_text_end: int = -1
    _last_frame_end: int = -1

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_MODES:
            raise ValueError("unsupported modem stream mode")
        if not 8000 <= self.sample_rate <= 192000:
            raise ValueError("unsupported modem stream sample rate")
        if not math.isfinite(self.carrier_hz):
            raise ValueError("invalid stream carrier")

    def compatible(self, *, session_id: str, track_id: str,
                   mode: str, sample_rate: int, carrier_hz: float) -> bool:
        return (session_id == self.session_id and track_id == self.track_id
                and mode == self.mode and sample_rate == self.sample_rate
                and math.isfinite(carrier_hz)
                and abs(self.carrier_hz-carrier_hz) <= CARRIER_TOLERANCE_HZ[self.mode])

    def feed(self, result: dict[str, Any], *,
             sample_start: int, sample_count: int, grade: str,
             carrier_hz: float) -> dict[str, Any]:
        if (type(sample_start) is not int or sample_start < 0
                or type(sample_count) is not int
                or not 0 < sample_count <= MAX_WINDOW_SECONDS*self.sample_rate):
            raise ValueError("invalid bounded modem source window")
        end = sample_start + sample_count
        if not self.compatible(session_id=self.session_id, track_id=self.track_id,
                               mode=self.mode, sample_rate=self.sample_rate,
                               carrier_hz=carrier_hz):
            raise ValueError("carrier or stream discontinuity: reset required")
        self.carrier_hz = carrier_hz
        emitted: list[dict] = []
        suppressed = 0
        preceding_text_end = self._last_text_end
        if (result.get("decoder_id") == self.mode
                and result.get("status") == "decoded"
                and grade == ("frame-integrity-passed" if self.mode == "ax25"
                              else "coherent-unverified")):
            if self.mode == "ax25":
                for frame in result.get("frames", [])[:MAX_DELTA_ITEMS]:
                    if not isinstance(frame, dict):
                        continue
                    span = _span(frame, sample_start, end)
                    if span is None:
                        continue
                    checks = frame.get("integrity", [])
                    if not isinstance(checks, list) or not any(
                            isinstance(check, dict)
                            and check.get("kind") in ("crc", "checksum")
                            and check.get("outcome") == "passed"
                            for check in checks):
                        continue
                    payload = frame.get("payload_base64")
                    if not isinstance(payload, str) or not payload or len(payload) > 12000:
                        continue
                    if any(prev_payload == payload
                           and _same_occurrence(prev_span, span, self.sample_rate)
                           for prev_span, prev_payload in self._frames):
                        suppressed += 1
                        continue
                    if span[1] < self._last_frame_end - self.sample_rate:
                        suppressed += 1
                        continue
                    self._frames.append((span, payload))
                    self._last_frame_end = max(self._last_frame_end, span[1])
                    emitted.append(frame)
                self.emitted_frames += len(emitted)
                # A frame is an independent payload. The local decoder currently
                # emits one printable projection per FCS-passed packet.
                delta = result.get("text", "") if emitted else ""
                if not isinstance(delta, str):
                    delta = ""
            else:
                raw = result.get("tokens", [])
                if not isinstance(raw, list):
                    raw = []
                for token in raw[:MAX_DELTA_ITEMS]:
                    if not isinstance(token, dict):
                        continue
                    span = _span(token, sample_start, end)
                    ch = token.get("text")
                    if span is None or not isinstance(ch, str) or len(ch) != 1:
                        continue
                    if any(prev_text == ch and _same_occurrence(prev_span, span, self.sample_rate)
                           for prev_span, prev_text in self._seen):
                        suppressed += 1
                        continue
                    # Avoid resurrecting a cropped old message when a later
                    # overlapping window has already advanced the transcript.
                    if span[1] < self._last_text_end - self.sample_rate//5:
                        suppressed += 1
                        continue
                    self._seen.append((span, ch))
                    self._last_text_end = max(self._last_text_end, span[1])
                    emitted.append(token)
                self.emitted_units += len(emitted)
                delta = "".join(t["text"] for t in emitted)
        else:
            delta = ""
        if delta:
            separator = ""
            if self.cumulative_text:
                if self.mode == "ax25":
                    separator = " | "
                elif (emitted and preceding_text_end >= 0 and
                      emitted[0]["start_sample"] - preceding_text_end > .7*self.sample_rate):
                    # A later burst is distinct even when it repeats identical
                    # text. Delimit it rather than run two QSOs together.
                    separator = "\n"
            self.cumulative_text = (self.cumulative_text+separator+delta)[-MAX_CUMULATIVE_TEXT:]
        if self.mode == "ax25":
            quality = "frame-integrity-passed" if emitted else "none"
            reason = "new-crc-passed-frames" if emitted else "no-new-crc-passed-frames"
        else:
            quality = "coherent-unverified" if emitted else "none"
            reason = "new-unverified-text-tokens" if emitted else "no-new-text-tokens"
        return {
            "strategy": "overlap-deduplicated-modem-evidence-v1",
            "session_id": self.session_id, "track_id": self.track_id,
            "mode": self.mode, "delta_text": delta,
            "delta_tokens": emitted if self.mode != "ax25" else [],
            "delta_frames": emitted if self.mode == "ax25" else [],
            "cumulative_text": self.cumulative_text,
            "new_items": len(emitted), "suppressed_overlap_items": suppressed,
            "emitted_units": self.emitted_units,
            "emitted_frames": self.emitted_frames,
            "last_emitted_end_sample": max(self._last_text_end, self._last_frame_end),
            "grade": quality, "reason": reason,
            "carrier_hz": carrier_hz,
            "confirmed_identity": False, "not_verified": True,
        }
