"""Pinned RTTY suite exercises positive decoding and mark-only rejection."""
from pathlib import Path

from sigid.conformance import evaluate_suite
from sigid.libsigdecode import DecoderLibrary
from sigid.libsigdecode.rtty import RTTYDecoder


MANIFEST = Path(__file__).parent / "fixtures/decoder_conformance/rtty/manifest.json"
MANIFEST_57 = Path(__file__).parent / "fixtures/decoder_conformance/rtty/manifest-57baud.json"


def test_rtty_pinned_positive_and_negative_controls():
    report = evaluate_suite(MANIFEST, library=DecoderLibrary((RTTYDecoder(),)))

    assert report["passed"], report["cases"]
    assert report["quality_gate"] == "passed"
    assert report["synthetic_only"] is True


def test_rtty_57_baud_14_stop_positive_and_negative_controls():
    from sigid.libsigdecode.rtty import RTTYSettings

    decoder = RTTYDecoder(RTTYSettings(baud=57.0, shift_hz=170.0, stop_bits=1.4))
    report = evaluate_suite(MANIFEST_57, library=DecoderLibrary((decoder,)))

    assert report["passed"], report["cases"]
    assert report["quality_gate"] == "passed"
    assert report["synthetic_only"] is True
