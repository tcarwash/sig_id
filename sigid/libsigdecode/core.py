"""Engine-neutral contracts for independently implemented local decoders.

A decoder consumes bounded normalized mono samples. It never owns hardware,
a tuning connection, a subprocess, a file path, or a transmitter. No external
radio application is imported here. Decoding is not classification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .artifacts import (
    MAX_FRAMES, MAX_SOFT_SPANS, MAX_SOFT_VALUES, DecodedFrame, SoftBitSpan,
)


@dataclass(frozen=True, slots=True)
class DecodeWindow:
    samples: np.ndarray
    sample_rate: int
    sample_start: int = 0
    carrier_hz: float | None = None

    def __post_init__(self) -> None:
        audio = np.asarray(self.samples, dtype=np.float64)
        if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("local decoding requires a nonempty finite mono signal")
        if (not isinstance(self.sample_rate, int) or isinstance(self.sample_rate, bool)
                or not 1000 <= self.sample_rate <= 384000):
            raise ValueError("sample rate must be an integer between 1000 and 384000 Hz")
        if self.sample_start < 0:
            raise ValueError("sample_start cannot be negative")
        if audio.size > 120 * self.sample_rate:
            raise ValueError("one local decoder window cannot exceed 120 seconds")
        if self.carrier_hz is not None and not (20 <= self.carrier_hz < self.sample_rate / 2):
            raise ValueError("carrier must be between 20 Hz and Nyquist")
        object.__setattr__(self, "samples", audio)


@dataclass(frozen=True, slots=True)
class DecoderSpec:
    id: str
    name: str
    family: str
    input_kind: str = "mono-audio"
    stage: str = "experimental"
    version: str = "0.1.0"


@dataclass(frozen=True, slots=True)
class DecodedToken:
    text: str
    start_sample: int
    end_sample: int
    pattern: str
    confidence_hint: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "start_sample": self.start_sample,
            "end_sample": self.end_sample, "pattern": self.pattern,
            "confidence_hint": self.confidence_hint,
        }


@dataclass(frozen=True, slots=True)
class LocalDecodeResult:
    decoder_id: str
    version: str
    status: str  # decoded / partial / no_decode
    text: str = ""
    tokens: tuple[DecodedToken, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    frames: tuple[DecodedFrame, ...] = ()
    soft_bits: tuple[SoftBitSpan, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("decoded", "partial", "no_decode"):
            raise ValueError("invalid local decoder result status")
        if self.status == "no_decode" and self.text:
            raise ValueError("no_decode results must not publish decoded text")
        if len(self.frames) > MAX_FRAMES or not all(isinstance(f, DecodedFrame) for f in self.frames):
            raise ValueError("decoder result contains too many or invalid frames")
        if (len(self.soft_bits) > MAX_SOFT_SPANS
                or not all(isinstance(s, SoftBitSpan) for s in self.soft_bits)
                or sum(len(s.values) for s in self.soft_bits) > MAX_SOFT_VALUES):
            raise ValueError("decoder result contains too much or invalid soft-bit evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoder_id": self.decoder_id, "version": self.version,
            "status": self.status, "text": self.text,
            "tokens": [token.to_dict() for token in self.tokens],
            "frames": [frame.to_dict() for frame in self.frames],
            "soft_bits": [span.to_dict() for span in self.soft_bits],
            "evidence": self.evidence, "note": self.note,
            "source": "sigid-independent-implementation",
        }


class Decoder(Protocol):
    spec: DecoderSpec

    def decode(self, window: DecodeWindow) -> LocalDecodeResult: ...


class DecoderLibrary:
    """Small explicit registry; external adapters remain at the API boundary."""

    def __init__(self, decoders: tuple[Decoder, ...] = ()) -> None:
        self._decoders: dict[str, Decoder] = {}
        for decoder in decoders:
            self.register(decoder)

    def register(self, decoder: Decoder) -> None:
        key = decoder.spec.id
        if not key or key != key.lower() or not all(c.islower() or c.isdigit() or c in "-_" for c in key):
            raise ValueError("decoder IDs must be lowercase ASCII slugs")
        if key in self._decoders:
            raise ValueError(f"decoder already registered: {key}")
        self._decoders[key] = decoder

    def modes(self) -> tuple[DecoderSpec, ...]:
        return tuple(decoder.spec for decoder in self._decoders.values())

    def decode(self, mode_id: str, window: DecodeWindow) -> LocalDecodeResult:
        decoder = self._decoders.get(mode_id)
        if decoder is None:
            raise KeyError(f"unknown local decoder: {mode_id}")
        result = decoder.decode(window)
        if result.decoder_id != mode_id:
            raise ValueError("decoder result ID differs from requested decoder")
        window_end = window.sample_start + len(window.samples)
        for frame in result.frames:
            if frame.start_sample < window.sample_start or frame.end_sample > window_end:
                raise ValueError("decoded frame extends beyond original capture window")
        for span in result.soft_bits:
            if span.start_sample < window.sample_start or span.end_sample > window_end:
                raise ValueError("soft-bit evidence extends beyond original capture window")
        return result
