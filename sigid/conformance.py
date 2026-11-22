"""Deterministic, receive-only conformance runs for independently owned decoders.

Fixture ground truth must be independently established. Success here never
establishes an authentic over-the-air protocol identity or a validated decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import wave
from typing import Any

import numpy as np

from .libsigdecode import DecodeWindow, DecoderLibrary, create_library

MAX_FIXTURES = 256
MAX_AUDIO_BYTES = 100 * 1024 * 1024
HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
SOURCE_KINDS = ("synthetic", "unreviewed_air", "reviewed_air")
VALID_STATUSES = ("decoded", "partial", "no_decode")


def _nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _fixture_path(directory: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("fixture WAV must use a relative path")
    if any(segment in ("", ".", "..") for segment in relative.split("/")):
        raise ValueError("fixture WAV path cannot traverse directories")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("fixture WAV must remain under the manifest directory")
    if not path.is_file():
        raise ValueError(f"fixture WAV does not exist: {relative}")
    if path.stat().st_size > MAX_AUDIO_BYTES:
        raise ValueError("fixture WAV exceeds 100 MiB bound")
    return path


def _checked_manifest(path: Path) -> tuple[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("conformance manifest schema_version must be 1")
    suite_id = data.get("suite_id")
    if not isinstance(suite_id, str) or not SLUG.fullmatch(suite_id):
        raise ValueError("suite_id must be a lowercase slug")
    fixtures = data.get("fixtures")
    if not isinstance(fixtures, list) or not 1 <= len(fixtures) <= MAX_FIXTURES:
        raise ValueError("conformance manifest requires 1-256 fixtures")
    identifiers: set[str] = set()
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise ValueError("fixture entries must be objects")
        identifier = fixture.get("id")
        mode = fixture.get("decoder_id")
        if not isinstance(identifier, str) or not SLUG.fullmatch(identifier):
            raise ValueError("fixture id must be a lowercase slug")
        if identifier in identifiers:
            raise ValueError(f"duplicate fixture id: {identifier}")
        identifiers.add(identifier)
        if not isinstance(mode, str) or not SLUG.fullmatch(mode):
            raise ValueError(f"{identifier}: decoder_id must be a lowercase slug")
        provenance = fixture.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("kind") not in SOURCE_KINDS:
            raise ValueError(f"{identifier}: provenance.kind must be explicit")
        if not isinstance(provenance.get("description"), str) or not provenance["description"].strip():
            raise ValueError(f"{identifier}: provenance.description is required")
        if provenance["kind"] == "reviewed_air" and not all(
            isinstance(provenance.get(k), str) and provenance[k].strip()
            for k in ("reviewer", "truth_basis", "source_reference")
        ):
            raise ValueError(f"{identifier}: reviewed_air requires reviewer, truth_basis, source_reference")
        audio = fixture.get("audio")
        if not isinstance(audio, dict) or not HEX_DIGEST.fullmatch(str(audio.get("sha256", ""))):
            raise ValueError(f"{identifier}: audio.sha256 is required")
        _fixture_path(path.parent, audio.get("path"))
        carrier = fixture.get("carrier_hz")
        if carrier is not None and (
            isinstance(carrier, bool) or not isinstance(carrier, (int, float))
            or not math.isfinite(carrier) or carrier <= 0
        ):
            raise ValueError(f"{identifier}: carrier_hz must be a finite audio frequency or null")
        _nonnegative_int(audio.get("window_offset_samples", 0), "window_offset_samples")
        _nonnegative_int(audio.get("file_start_sample", 0), "file_start_sample")
        if "sample_count" in audio and _nonnegative_int(audio["sample_count"], "sample_count") == 0:
            raise ValueError(f"{identifier}: sample_count cannot be zero")
        expected = fixture.get("expected")
        if not isinstance(expected, dict) or expected.get("kind") not in ("positive", "negative", "observe"):
            raise ValueError(f"{identifier}: expected.kind must be positive, negative, or observe")
        if (provenance["kind"] == "unreviewed_air") != (expected["kind"] == "observe"):
            raise ValueError(f"{identifier}: unreviewed_air must be observe only; graded controls require independent truth")
        if expected["kind"] == "positive":
            statuses = expected.get("allowed_statuses", ["decoded"])
            if (not isinstance(statuses, list) or not statuses or
                    any(status not in ("decoded", "partial") for status in statuses)):
                raise ValueError(f"{identifier}: positive statuses must be decoded/partial")
            if "text" not in expected and "frames" not in expected:
                raise ValueError(f"{identifier}: positive fixture requires expected text or frames")
            if "text" in expected and not isinstance(expected["text"], str):
                raise ValueError(f"{identifier}: expected text must be a string")
            if expected.get("text") == "" and not expected.get("frames"):
                raise ValueError(f"{identifier}: a positive control cannot expect only empty output")
            if "max_character_error_rate" in expected and (
                isinstance(expected["max_character_error_rate"], bool)
                or not isinstance(expected["max_character_error_rate"], (int, float))
                or not math.isfinite(expected["max_character_error_rate"])
                or not 0 <= expected["max_character_error_rate"] <= 1
            ):
                raise ValueError(f"{identifier}: invalid max_character_error_rate")
            if "frames" in expected:
                frames = expected["frames"]
                if not isinstance(frames, list) or not 1 <= len(frames) <= 32:
                    raise ValueError(f"{identifier}: expected frames must be a nonempty bounded list")
                for frame in frames:
                    if not isinstance(frame, dict) or not isinstance(frame.get("payload_hex"), str):
                        raise ValueError(f"{identifier}: frame requires hex payload")
                    if not re.fullmatch(r"(?:[0-9a-fA-F]{2})*", frame["payload_hex"]):
                        raise ValueError(f"{identifier}: invalid frame payload_hex")
                    if len(frame["payload_hex"]) > 16384:
                        raise ValueError(f"{identifier}: frame payload_hex exceeds 8192 bytes")
                    if "bit_length" in frame:
                        _nonnegative_int(frame["bit_length"], "bit_length")
                        if frame["bit_length"] > len(bytes.fromhex(frame["payload_hex"])) * 8:
                            raise ValueError(f"{identifier}: frame bit_length exceeds bytes")
                    if (not isinstance(frame.get("integrity", []), list)
                            or len(frame.get("integrity", [])) > 8):
                        raise ValueError(f"{identifier}: invalid integrity list")
                    for check in frame.get("integrity", []):
                        if (not isinstance(check, dict)
                                or check.get("kind") not in ("crc", "fec", "checksum", "parity")
                                or check.get("outcome") not in ("passed", "failed", "not_checked")
                                or not isinstance(check.get("algorithm"), str)
                                or not check["algorithm"].strip()):
                            raise ValueError(f"{identifier}: invalid integrity expectation")
        elif "text" in expected or "frames" in expected:
            raise ValueError(f"{identifier}: negative/observe fixtures cannot assert decoded payloads")
    return suite_id, fixtures


def _audio_window(manifest_dir: Path, fixture: dict[str, Any]) -> DecodeWindow:
    audio = fixture["audio"]
    path = _fixture_path(manifest_dir, audio["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != audio["sha256"]:
        raise ValueError(f"{fixture['id']}: WAV SHA-256 mismatch")
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError(f"{fixture['id']}: expected mono uncompressed PCM16 WAV")
        count = wav.getnframes()
        offset = audio.get("window_offset_samples", 0)
        selected = audio.get("sample_count", count - offset)
        if selected <= 0 or offset + selected > count:
            raise ValueError(f"{fixture['id']}: audio window extends beyond WAV")
        if selected > wav.getframerate() * 120:
            raise ValueError(f"{fixture['id']}: local decode window exceeds 120 seconds")
        wav.setpos(offset)
        samples = np.frombuffer(wav.readframes(selected), dtype="<i2").astype(np.float64) / 32768.0
        if len(samples) != selected:
            raise ValueError(f"{fixture['id']}: incomplete selected WAV samples")
        return DecodeWindow(
            samples, wav.getframerate(),
            sample_start=audio.get("file_start_sample", 0) + offset,
            carrier_hz=fixture.get("carrier_hz"),
        )


def character_errors(reference: str, actual: str) -> int:
    """Levenshtein edits with deterministic O(min(m,n)) memory."""
    if len(actual) > len(reference):
        reference, actual = actual, reference
    previous = list(range(len(actual) + 1))
    for i, expected_char in enumerate(reference, start=1):
        current = [i] + [0] * len(actual)
        for j, actual_char in enumerate(actual, start=1):
            current[j] = min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (expected_char != actual_char),
            )
        previous = current
    return previous[-1]


def _compare_frame(expected: dict[str, Any], actual: Any) -> bool:
    if actual.payload != bytes.fromhex(expected["payload_hex"]):
        return False
    if "bit_length" in expected and actual.bit_length != expected["bit_length"]:
        return False
    for required in expected.get("integrity", []):
        if not any(check.kind == required["kind"]
                   and check.algorithm == required["algorithm"]
                   and check.outcome == required["outcome"]
                   for check in actual.integrity):
            return False
    return True


def _measure(fixture: dict[str, Any], result: Any, window: DecodeWindow) -> dict[str, Any]:
    expected = fixture["expected"]
    is_negative = expected["kind"] == "negative"
    is_observation = expected["kind"] == "observe"
    failures: list[str] = []
    if result.status not in VALID_STATUSES:
        failures.append(f"unknown decoder status: {result.status}")
    if is_negative:
        if result.status != "no_decode":
            failures.append(f"negative control produced {result.status} rather than no_decode")
        if result.text:
            failures.append("negative control published decoded text")
        # no_decode may carry *candidate* frames as evidence; this is not a successful frame.
    elif not is_observation and result.status not in expected.get("allowed_statuses", ["decoded"]):
        failures.append(f"unexpected result status: {result.status}")
    metrics: dict[str, Any] = {}
    if not is_negative and "text" in expected:
        target = expected["text"]
        errors = character_errors(target, result.text)
        metrics["character_errors"] = errors
        metrics["reference_characters"] = len(target)
        metrics["character_error_rate"] = errors / max(1, len(target))
        metrics["exact_text_match"] = target == result.text
        if metrics["character_error_rate"] > expected.get("max_character_error_rate", 0):
            failures.append("character error rate exceeds fixture threshold")
    if not is_negative and "frames" in expected:
        references = expected["frames"]
        # One-to-one and order-independent, preventing a single decoded
        # frame from satisfying multiple expected frames.
        available = list(result.frames)
        matched = 0
        for target in references:
            index = next((i for i, frame in enumerate(available)
                          if _compare_frame(target, frame)), None)
            if index is not None:
                available.pop(index)
                matched += 1
        metrics["frames_expected"] = len(references)
        metrics["frames_matched"] = matched
        metrics["frames_extra"] = len(available)
        metrics["frame_recall"] = matched / len(references)
        if matched != len(references) or available:
            failures.append("candidate frames or integrity checks differ from reference")
    for token in result.tokens:
        if token.start_sample < window.sample_start or token.end_sample > window.sample_start + len(window.samples):
            failures.append("token span extends outside selected audio")
            break
    return {
        "fixture_id": fixture["id"],
        "decoder_id": fixture["decoder_id"],
        "provenance_kind": fixture["provenance"]["kind"],
        "expected_kind": expected["kind"],
        "result_status": result.status,
        "result_text": result.text,
        "result_frame_count": len(result.frames),
        "result_version": result.version,
        "sample_start": window.sample_start,
        "sample_count": len(window.samples),
        "metrics": metrics,
        "passed": (None if is_observation and not failures else not failures),
        "failures": failures,
    }


def _aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [item for item in items if item["expected_kind"] == "positive"]
    negatives = [item for item in items if item["expected_kind"] == "negative"]
    with_text = [item for item in positives if "character_errors" in item["metrics"]]
    errors = sum(item["metrics"]["character_errors"] for item in with_text)
    length = sum(item["metrics"]["reference_characters"] for item in with_text)
    times = [item["metrics"]["decode_elapsed_ms"] for item in items
             if "decode_elapsed_ms" in item["metrics"]]
    realtime = [item["metrics"]["realtime_factor"] for item in items
                if "realtime_factor" in item["metrics"]]
    return {
        "cases": len(items),
        "graded_cases": sum(item["passed"] is not None for item in items),
        "observations": sum(item["passed"] is None for item in items),
        "passed": sum(item["passed"] is True for item in items),
        "failed": sum(item["passed"] is False for item in items),
        "positive_cases": len(positives),
        "negative_cases": len(negatives),
        "positive_status_rate": (sum(item["result_status"] in ("decoded", "partial")
                                    for item in positives) / len(positives)) if positives else None,
        "negative_abstention_rate": (sum(item["result_status"] == "no_decode"
                                        for item in negatives) / len(negatives)) if negatives else None,
        "negative_false_positive_rate": (sum(item["result_status"] != "no_decode"
                                            for item in negatives) / len(negatives)) if negatives else None,
        "character_error_rate": errors / max(1, length) if with_text else None,
        "frames_expected": sum(item["metrics"].get("frames_expected", 0) for item in positives),
        "frames_matched": sum(item["metrics"].get("frames_matched", 0) for item in positives),
        "decode_ms_median": round(float(np.median(times)), 3) if times else None,
        "decode_ms_p95": round(float(np.percentile(times, 95)), 3) if times else None,
        "realtime_factor_median": round(float(np.median(realtime)), 4) if realtime else None,
    }


def evaluate_suite(manifest_path: str | Path, *, library: DecoderLibrary | None = None) -> dict[str, Any]:
    """Run a pinned suite; agents can pass a temporary, unregistered decoder library."""
    manifest = Path(manifest_path).resolve()
    suite_id, fixtures = _checked_manifest(manifest)
    decoder_library = library if library is not None else create_library()
    modes = {mode.id: mode for mode in decoder_library.modes()}
    cases: list[dict[str, Any]] = []
    for fixture in fixtures:
        decoder_id = fixture["decoder_id"]
        if decoder_id not in modes:
            raise ValueError(f"{fixture['id']}: decoder {decoder_id} is not registered in supplied library")
        if modes[decoder_id].input_kind != "mono-audio":
            raise ValueError(f"{fixture['id']}: decoder requires unsupported {modes[decoder_id].input_kind} fixture input")
        window = _audio_window(manifest.parent, fixture)
        # Algorithm exceptions are real conformance failures, never "no_decode".
        start_time = time.perf_counter()
        try:
            result = decoder_library.decode(decoder_id, window)
        except Exception as exc:
            cases.append({
                "fixture_id": fixture["id"], "decoder_id": decoder_id,
                "provenance_kind": fixture["provenance"]["kind"],
                "expected_kind": fixture["expected"]["kind"],
                "result_status": "exception", "result_text": "",
                "result_frame_count": 0, "result_version": modes[decoder_id].version,
                "sample_start": window.sample_start, "sample_count": len(window.samples),
                "metrics": {}, "passed": False,
                "failures": [f"decoder raised {type(exc).__name__}: {str(exc)[:240]}"],
            })
        else:
            try:
                cases.append(_measure(fixture, result, window))
            except Exception as exc:
                cases.append({
                    "fixture_id": fixture["id"], "decoder_id": decoder_id,
                    "provenance_kind": fixture["provenance"]["kind"],
                    "expected_kind": fixture["expected"]["kind"],
                    "result_status": "invalid_result", "result_text": "",
                    "result_frame_count": 0, "result_version": modes[decoder_id].version,
                    "sample_start": window.sample_start, "sample_count": len(window.samples),
                    "metrics": {}, "passed": False,
                    "failures": [f"could not evaluate decoder result: {type(exc).__name__}: {str(exc)[:240]}"],
                })
        elapsed = max(0.0, time.perf_counter() - start_time)
        row = cases[-1]
        row["audio_sha256"] = fixture["audio"]["sha256"]
        row["carrier_hz"] = fixture.get("carrier_hz")
        row["metrics"]["decode_elapsed_ms"] = round(elapsed * 1000, 3)
        row["metrics"]["realtime_factor"] = round(
            elapsed / (len(window.samples) / window.sample_rate), 6,
        )
    provenance = {kind: _aggregate([item for item in cases if item["provenance_kind"] == kind])
                  for kind in SOURCE_KINDS
                  if any(item["provenance_kind"] == kind for item in cases)}
    by_mode = {mode: {kind: _aggregate([item for item in cases if item["decoder_id"] == mode
                                                and item["provenance_kind"] == kind])
                      for kind in SOURCE_KINDS
                      if any(item["decoder_id"] == mode and item["provenance_kind"] == kind
                             for item in cases)}
               for mode in sorted({item["decoder_id"] for item in cases})}
    return {
        "schema_version": 1, "suite_id": suite_id,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "passed": (any(item["passed"] is not None for item in cases)
                   and all(item["passed"] is not False for item in cases)),
        "quality_gate": ("ungraded" if not any(item["passed"] is not None for item in cases)
                         else "passed" if all(item["passed"] is not False for item in cases)
                         else "failed"),
        "summary": _aggregate(cases),
        "by_provenance": provenance,
        "by_mode": by_mode,
        "synthetic_only": set(provenance) == {"synthetic"},
        "cases": cases,
        "disclaimer": "Synthetic/black-box agreement does not establish reviewed on-air protocol validation.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate pinned decoder golden and negative WAV fixtures.")
    parser.add_argument("manifest", type=Path, help="Path to decoder conformance manifest JSON")
    parser.add_argument("--json-out", type=Path, help="Write full JSON report to this path")
    parser.add_argument("--report-only", action="store_true", help="Do not fail exit status for failed fixtures")
    args = parser.parse_args(argv)
    try:
        report = evaluate_suite(args.manifest)
    except (OSError, ValueError, json.JSONDecodeError, wave.Error) as exc:
        print(f"Invalid conformance suite: {exc}", file=sys.stderr)
        return 2
    document = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.json_out:
        args.json_out.write_text(document, encoding="utf-8")
    else:
        sys.stdout.write(document)
    return 0 if report["passed"] or args.report_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
