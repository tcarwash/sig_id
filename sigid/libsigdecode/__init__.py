"""In-project, independently authored protocol decoder library.

The library has no dependency on Fldigi, WSJT-X, OnAir, a receiver,
an external decoder executable or a transmitting device.
"""
from .core import (
    DecodeWindow, DecodedToken, DecoderLibrary, DecoderSpec, LocalDecodeResult,
)
from .cw import CWDecoder, CWSettings


def create_library() -> DecoderLibrary:
    return DecoderLibrary((CWDecoder(),))


__all__ = [
    "CWDecoder", "CWSettings", "DecodeWindow", "DecodedToken",
    "DecoderLibrary", "DecoderSpec", "LocalDecodeResult", "create_library",
]
