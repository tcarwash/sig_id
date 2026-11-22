"""In-project, independently authored protocol decoder library.

The library has no dependency on Fldigi, WSJT-X, OnAir, a receiver,
an external decoder executable or a transmitting device.
"""
from .core import (
    DecodeWindow, DecodedToken, DecoderLibrary, DecoderSpec, LocalDecodeResult,
)
from .cw import CWDecoder, CWSettings
from .rtty import RTTYDecoder, RTTYSettings
from .bpsk31 import BPSK31Decoder, BPSK31Settings
from .ax25 import AX25Decoder
from .artifacts import DecodedFrame, IntegrityCheck, SoftBitSpan


def create_library() -> DecoderLibrary:
    return DecoderLibrary((CWDecoder(), RTTYDecoder(), BPSK31Decoder(), AX25Decoder()))


__all__ = [
    "CWDecoder", "CWSettings", "DecodeWindow", "DecodedToken",
    "DecoderLibrary", "DecoderSpec", "LocalDecodeResult", "create_library",
    "DecodedFrame", "IntegrityCheck", "SoftBitSpan",
    "RTTYDecoder", "RTTYSettings", "BPSK31Decoder", "BPSK31Settings", "AX25Decoder",
]
