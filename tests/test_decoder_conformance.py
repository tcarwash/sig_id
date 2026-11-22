"""Regression tests for portable, provenance-aware decoder golden vectors.

A synthetic passing test is NOT evidence of successful over-the-air decoding.
"""
from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path
import shutil
import wave

import numpy as np
import pytest

from sigid.conformance import character_errors, evaluate_suite, main
from sigid.libsigdecode import (
    DecodeWindow, DecoderLibrary, DecoderSpec, DecodedFrame, IntegrityCheck,
    LocalDecodeResult, SoftBitSpan,
)
ROOT = Path(__file__).resolve().parents[1]
build_fixtures = runpy.run_path(str(ROOT / "tools/build_decoder_conformance_cw_fixtures.py"))["build_fixtures"]
CW_MANIFEST = ROOT / "tests/fixtures/decoder_conformance/cw/manifest.json"


def copy_cw_suite(tmp_path: Path) -> Path:
    reference = json.loads(CW_MANIFEST.read_text())
    for fixture in reference["fixtures"]:
        filename = fixture["audio"]["path"]
        shutil.copyfile(CW_MANIFEST.parent / filename, tmp_path / filename)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(reference, indent=2) + "\n")
    return manifest


def test_independently_pinned_cw_synthetic_goldens_and_abstentions():
    report = evaluate_suite(CW_MANIFEST)
    assert report["passed"], report
    assert report["synthetic_only"]
    assert set(report["by_provenance"]) == {"synthetic"}
    assert report["summary"]["cases"] == 5
    assert report["summary"]["positive_cases"] == 2
    assert report["summary"]["negative_cases"] == 3
    assert report["summary"]["negative_abstention_rate"] == 1.0
    assert report["summary"]["negative_false_positive_rate"] == 0.0
    assert report["summary"]["character_error_rate"] == 0.0
    assert report["quality_gate"] == "passed"
    assert report["manifest_sha256"] == hashlib.sha256(CW_MANIFEST.read_bytes()).hexdigest()
    assert report["summary"]["decode_ms_median"] >= 0.0
    assert report["summary"]["decode_ms_p95"] >= 0.0
    assert report["summary"]["realtime_factor_median"] >= 0.0
    assert all("audio_sha256" in item and "decode_elapsed_ms" in item["metrics"]
               for item in report["cases"])
    assert all(item["passed"] is True for item in report["cases"])
    assert all(item["result_version"] == "cw-envelope-v1" for item in report["cases"])
    assert all(item["sample_start"] == 17000 for item in report["cases"])


def test_generated_waveform_bytes_reproduce_pinned_hashes(tmp_path):
    generated = build_fixtures(tmp_path)
    pinned = json.loads(CW_MANIFEST.read_text())
    assert generated == pinned
    assert {case["audio"]["sha256"] for case in generated["fixtures"]} == {
        hashlib.sha256((CW_MANIFEST.parent / case["audio"]["path"]).read_bytes()).hexdigest()
        for case in pinned["fixtures"]
    }


def test_failed_sha_or_path_traversal_cannot_be_scored(tmp_path):
    manifest = copy_cw_suite(tmp_path)
    data = json.loads(manifest.read_text())
    (tmp_path / data["fixtures"][0]["audio"]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_suite(manifest)
    data["fixtures"][0]["audio"]["path"] = "../outside.wav"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="traverse"):
        evaluate_suite(manifest)


def test_provenance_cannot_pretend_unreviewed_captures_are_verified(tmp_path):
    manifest = copy_cw_suite(tmp_path)
    data = json.loads(manifest.read_text())
    data["fixtures"][0]["provenance"]["kind"] = "reviewed_air"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="reviewer, truth_basis, source_reference"):
        evaluate_suite(manifest)
    data["fixtures"][0]["provenance"] = {
        "kind": "unreviewed_air", "description": "Only a guessed label",
    }
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unreviewed_air must be observe only"):
        evaluate_suite(manifest)
    data["fixtures"][0]["expected"] = {"kind": "observe"}
    manifest.write_text(json.dumps(data))
    observed = evaluate_suite(manifest)
    assert observed["passed"]
    assert not observed["synthetic_only"]
    assert observed["summary"]["observations"] == 1
    assert observed["summary"]["graded_cases"] == 4
    assert observed["quality_gate"] == "passed"
    observation = next(row for row in observed["cases"] if row["fixture_id"] == "cw_cq_test")
    assert observation["passed"] is None
    assert observation["expected_kind"] == "observe"


class PacketFixtureDecoder:
    spec = DecoderSpec(
        id="packet_fixture", name="Test fixture — not a modem",
        family="test", version="fixture-v1",
    )

    def __init__(self, *, status: str = "partial", crc: str = "passed", text: str = ""):
        self.status, self.crc, self.text = status, crc, text

    def decode(self, window: DecodeWindow) -> LocalDecodeResult:
        start = window.sample_start + 125
        return LocalDecodeResult(
            "packet_fixture", "fixture-v1", self.status, text=self.text,
            frames=(DecodedFrame(
                start, start + 250, bytes.fromhex("aaff"),
                integrity=(IntegrityCheck("crc", "test-crc", self.crc),),
            ),),
            soft_bits=(SoftBitSpan(
                start, start + 200, (0.1, 0.5, 0.9), "probability_one",
            ),),
        )


def packet_manifest(tmp_path: Path, *, negative: bool = False) -> Path:
    audio = tmp_path / "packet-no-protocol.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(12000)
        wav.writeframes(np.zeros(16000, dtype="<i2").tobytes())
    data = {
        "schema_version": 1, "suite_id": "packet-fixture-test",
        "fixtures": [{
            "id": "frame_case", "decoder_id": "packet_fixture",
            "provenance": {
                "kind": "synthetic", "description": "Synthetic packet fixture, no actual modem",
            },
            "audio": {
                "path": audio.name,
                "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "file_start_sample": 2000, "window_offset_samples": 100,
                "sample_count": 12000,
            },
            "expected": ({"kind": "negative"} if negative else {
                "kind": "positive", "allowed_statuses": ["partial"],
                "frames": [{
                    "payload_hex": "aaff", "bit_length": 16,
                    "integrity": [{
                        "kind": "crc", "algorithm": "test-crc", "outcome": "passed",
                    }],
                }],
            }),
        }],
    }
    path = tmp_path / "packet-manifest.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def test_packet_frame_integrity_is_measured_with_absolute_sample_bounds(tmp_path):
    path = packet_manifest(tmp_path)
    report = evaluate_suite(path, library=DecoderLibrary((PacketFixtureDecoder(),)))
    assert report["passed"], report
    assert report["by_mode"]["packet_fixture"]["synthetic"]["frames_matched"] == 1
    case = report["cases"][0]
    assert case["sample_start"] == 2100
    assert case["sample_count"] == 12000
    assert case["metrics"]["frames_expected"] == 1
    assert case["metrics"]["frames_matched"] == 1
    assert case["metrics"]["frames_extra"] == 0
    bad_crc = evaluate_suite(
        path, library=DecoderLibrary((PacketFixtureDecoder(crc="failed"),)),
    )
    assert not bad_crc["passed"]
    assert bad_crc["cases"][0]["metrics"]["frames_matched"] == 0
    assert bad_crc["cases"][0]["metrics"]["frames_extra"] == 1
    assert "integrity" in bad_crc["cases"][0]["failures"][0]


def test_negative_false_decodes_are_explicit_failures(tmp_path):
    path = packet_manifest(tmp_path, negative=True)
    report = evaluate_suite(path, library=DecoderLibrary((PacketFixtureDecoder(),)))
    assert not report["passed"]
    assert report["summary"]["negative_abstention_rate"] == 0.0
    assert report["summary"]["negative_false_positive_rate"] == 1.0
    assert report["cases"][0]["passed"] is False
    assert any("negative control produced" in reason for reason in report["cases"][0]["failures"])


def test_decoder_exceptions_are_not_misreported_as_abstentions(tmp_path):
    class BrokenDecoder(PacketFixtureDecoder):
        def decode(self, window):
            raise RuntimeError("deliberate fixture error")
    path = packet_manifest(tmp_path, negative=True)
    report = evaluate_suite(path, library=DecoderLibrary((BrokenDecoder(),)))
    assert not report["passed"]
    assert report["cases"][0]["result_status"] == "exception"
    assert report["summary"]["negative_abstention_rate"] == 0.0


def test_expected_frame_duplicate_cannot_match_single_result_frame(tmp_path):
    path = packet_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    expected = manifest["fixtures"][0]["expected"]
    expected["frames"] = expected["frames"] * 2
    path.write_text(json.dumps(manifest))
    report = evaluate_suite(path, library=DecoderLibrary((PacketFixtureDecoder(),)))
    assert not report["passed"]
    assert report["cases"][0]["metrics"]["frames_matched"] == 1
    assert report["cases"][0]["metrics"]["frames_expected"] == 2


def test_observation_only_suite_cannot_claim_passing_conformance(tmp_path):
    path = packet_manifest(tmp_path, negative=True)
    manifest = json.loads(path.read_text())
    manifest["fixtures"][0]["provenance"]["kind"] = "unreviewed_air"
    manifest["fixtures"][0]["expected"] = {"kind": "observe"}
    path.write_text(json.dumps(manifest))
    report = evaluate_suite(path, library=DecoderLibrary((PacketFixtureDecoder(),)))
    assert report["quality_gate"] == "ungraded"
    assert not report["passed"]
    assert report["summary"]["graded_cases"] == 0
    assert report["summary"]["observations"] == 1
    assert report["cases"][0]["passed"] is None


def test_manifest_rejects_ambiguous_carrier_and_empty_positive(tmp_path):
    path = packet_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["fixtures"][0]["carrier_hz"] = "audio?"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="carrier_hz"):
        evaluate_suite(path)
    manifest["fixtures"][0].pop("carrier_hz")
    manifest["fixtures"][0]["expected"] = {"kind": "positive", "text": ""}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="empty output"):
        evaluate_suite(path)


def test_text_character_distance_is_deterministic_and_not_success_rate():
    assert character_errors("CQ TEST", "CQ TEST") == 0
    assert character_errors("CQ TEST", "CQ TENT") == 1
    assert character_errors("CQ TEST", "") == 7
    assert character_errors("", "CQ") == 2
    assert character_errors("SOS", "SSOS") == 1


def test_cli_reports_machine_readable_results_and_exit_status(tmp_path):
    destination = tmp_path / "report.json"
    assert main([str(CW_MANIFEST), "--json-out", str(destination)]) == 0
    report = json.loads(destination.read_text())
    assert report["passed"] and report["synthetic_only"]
    assert main([str(tmp_path / "no-such-manifest.json")]) == 2
