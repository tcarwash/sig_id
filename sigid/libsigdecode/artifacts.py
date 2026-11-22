"""Bounded, typed packet and soft-decision evidence for first-party decoders.

These types describe independent decoder output; they never assert on-air
protocol identity or human-verified payload content.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import math
from typing import Literal

IntegrityKind = Literal["crc", "fec", "checksum", "parity"]
IntegrityOutcome = Literal["passed", "failed", "not_checked"]
SoftBitEncoding = Literal["llr_positive_is_one", "probability_one"]

MAX_FRAMES = 32
MAX_FRAME_BYTES = 8192
MAX_SOFT_SPANS = 16
MAX_SOFT_VALUES = 8192


def _check_span(start: int, end: int) -> None:
    if (type(start) is not int or type(end) is not int
            or start < 0 or end <= start):
        raise ValueError("evidence sample span must be ordered, nonnegative integers")


@dataclass(frozen=True, slots=True)
class IntegrityCheck:
    """A specific computed check, not an assertion of decoded mode identity."""

    kind: IntegrityKind
    algorithm: str
    outcome: IntegrityOutcome
    corrected_bits: int | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("crc", "fec", "checksum", "parity"):
            raise ValueError("unknown integrity-check kind")
        if self.outcome not in ("passed", "failed", "not_checked"):
            raise ValueError("unknown integrity-check outcome")
        if not isinstance(self.algorithm, str) or not 1 <= len(self.algorithm) <= 120:
            raise ValueError("integrity algorithm name must be 1-120 characters")
        if self.corrected_bits is not None and (
            self.kind != "fec" or type(self.corrected_bits) is not int
            or self.corrected_bits < 0
        ):
            raise ValueError("corrected_bits applies only to FEC checks and must be nonnegative")
        if self.outcome == "not_checked" and self.corrected_bits is not None:
            raise ValueError("not_checked FEC cannot claim corrected_bits")
        if len(self.note) > 500:
            raise ValueError("integrity-check note is too long")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "algorithm": self.algorithm,
            "outcome": self.outcome, "corrected_bits": self.corrected_bits,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """Candidate binary frame, source-indexed to the original audio capture.

    payload bytes hold MSB-first bits; bit_length says how many leading bits
    are meaningful when the last byte is not complete. CRC/FEC checks have
    separate, explicit outcomes and do not imply operator verification.
    """

    start_sample: int
    end_sample: int
    payload: bytes
    bit_length: int | None = None
    protocol_hint: str = ""
    integrity: tuple[IntegrityCheck, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        _check_span(self.start_sample, self.end_sample)
        if not isinstance(self.payload, bytes) or len(self.payload) > MAX_FRAME_BYTES:
            raise ValueError("frame payload must be bytes of at most 8192 bytes")
        length = len(self.payload) * 8 if self.bit_length is None else self.bit_length
        if type(length) is not int or not 0 <= length <= 8 * len(self.payload):
            raise ValueError("frame bit_length must fit its payload")
        if len(self.integrity) > 8 or not all(isinstance(i, IntegrityCheck) for i in self.integrity):
            raise ValueError("frames support at most eight typed integrity checks")
        if len(self.protocol_hint) > 100 or len(self.note) > 500:
            raise ValueError("frame hint or note is too long")
        object.__setattr__(self, "bit_length", length)

    def to_dict(self) -> dict:
        return {
            "start_sample": self.start_sample, "end_sample": self.end_sample,
            "payload_base64": base64.b64encode(self.payload).decode("ascii"),
            "encoding": "base64", "byte_count": len(self.payload),
            "bit_length": self.bit_length, "protocol_hint": self.protocol_hint,
            "integrity": [item.to_dict() for item in self.integrity],
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class SoftBitSpan:
    """Uncalibrated soft evidence; values are not generic confidence scores.

    llr_positive_is_one = log(P(bit=1)/P(bit=0)); probability_one is in [0,1].
    """

    start_sample: int
    end_sample: int
    values: tuple[float, ...]
    encoding: SoftBitEncoding
    note: str = ""

    def __post_init__(self) -> None:
        _check_span(self.start_sample, self.end_sample)
        if self.encoding not in ("llr_positive_is_one", "probability_one"):
            raise ValueError("unknown soft-bit encoding")
        values = tuple(self.values)
        if not 1 <= len(values) <= MAX_SOFT_VALUES:
            raise ValueError("soft-bit evidence requires 1-8192 values")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("soft-bit evidence must contain finite numeric values")
            if self.encoding == "probability_one" and not 0 <= value <= 1:
                raise ValueError("probability_one values must be in [0, 1]")
        if len(self.note) > 500:
            raise ValueError("soft-bit note is too long")
        object.__setattr__(self, "values", values)

    def to_dict(self) -> dict:
        return {
            "start_sample": self.start_sample, "end_sample": self.end_sample,
            "encoding": self.encoding, "values": list(self.values),
            "bit_count": len(self.values), "note": self.note,
        }
