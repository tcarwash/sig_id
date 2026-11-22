"""Pinned, independent AX.25 audio conformance vectors (synthetic only)."""
from pathlib import Path

from sigid.conformance import evaluate_suite
from sigid.libsigdecode import DecoderLibrary
from sigid.libsigdecode.ax25 import AX25Decoder


def test_ax25_audio_fcs_positive_and_negative_controls():
    manifest = Path(__file__).parent / "fixtures/decoder_conformance/ax25/manifest.json"
    report = evaluate_suite(manifest, library=DecoderLibrary((AX25Decoder(),)))
    assert report["passed"], report["cases"]
    assert report["quality_gate"] == "passed"
    assert report["synthetic_only"] is True
    assert report["summary"]["positive_cases"] >= 1
    assert report["summary"]["negative_cases"] >= 1
