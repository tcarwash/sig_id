"""Pinned BPSK31 suite exercises printable text and carrier abstention."""
from pathlib import Path

from sigid.conformance import evaluate_suite
from sigid.libsigdecode import DecoderLibrary
from sigid.libsigdecode.bpsk31 import BPSK31Decoder


MANIFEST = Path(__file__).parent / "fixtures/decoder_conformance/bpsk31/manifest.json"


def test_bpsk31_pinned_positive_and_negative_controls():
    report = evaluate_suite(MANIFEST, library=DecoderLibrary((BPSK31Decoder(),)))

    assert report["passed"], report["cases"]
    assert report["quality_gate"] == "passed"
    assert report["synthetic_only"] is True
