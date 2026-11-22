"""Ground-truthed, receive-only Signal Lab SNR/acquisition sweep.

Detect, decode, and noise controls have distinct denominators. This is
synthetic evidence, never an on-air accuracy or identity validation claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

from sigid import signal_lab
from sigid.native_live_decode import decode_live_candidate, _evidence_grade
from sigid.libsigdecode import create_library, DecodeWindow
from sigid.spectrum import waterfall_samples
from sigid.detection import detect_tracks

DEFAULT_SNRS = (-15, -10, -5, 0, 10, 25)
MODES = ("cw", "rtty", "bpsk31", "ax25")


def detection_matches(box: dict, truth: dict, rate: int) -> bool:
    """Positive is spatiotemporal coverage, *not* mode identification."""
    a = box["sample_start"]
    b = a + box["sample_count"]
    c = truth["start_sample"]
    d = c + truth["sample_count"]
    time_overlap = max(0, min(b, d) - max(a, c))
    if time_overlap < min(b-a, d-c, 0.30 * rate):
        return False
    f_lo = max(box["frequency_low_hz"], truth["occupied_low_hz"])
    f_hi = min(box["frequency_high_hz"], truth["occupied_high_hz"])
    width = truth["occupied_high_hz"] - truth["occupied_low_hz"]
    return f_hi - f_lo >= min(20, 0.25 * width)


def payload_correct(result: dict, expected: str, mode: str) -> bool:
    if result.get("grade") == "none" or result.get("text") != expected:
        return False
    if mode == "ax25":
        return result.get("grade") == "frame-integrity-passed"
    return result.get("grade") == "coherent-unverified"


def read_mono(path: str) -> tuple[np.ndarray, int, str]:
    with wave.open(path, "rb") as f:
        if f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise ValueError("expected mono PCM16")
        rate = f.getframerate()
        raw = f.readframes(f.getnframes())
    return np.frombuffer(raw, dtype="<i2"), rate, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cropped_pcm(pcm: np.ndarray, begin: int, end: int) -> bytes:
    return np.ascontiguousarray(pcm[max(0, begin):min(end, len(pcm))]).tobytes()


def decode_attempt(pcm: np.ndarray, rate: int, start: int, end: int,
                   carrier: float, mode: str) -> dict:
    if not 0 <= start < end <= len(pcm):
        raise ValueError("invalid attempt bounds")
    return decode_live_candidate(cropped_pcm(pcm, start, end),
                                 rate, start, carrier, (mode,))


def narrow_tone_metrics(pcm: np.ndarray, rate: int, *, start: int,
                        stop: int, truth: dict | None) -> dict:
    view = waterfall_samples(
        pcm[start:stop].astype(np.float64) / 32768.0,
        rate, fft_size=4096, hop_size=512, max_rows=320,
        sample_offset=start, total_sample_count=stop,
    )
    t0 = time.perf_counter()
    proposals = detect_tracks(
        view, threshold_db=18.0, minimum_peak_db=-32.0,
        maximum_tone_gap_hz=35.0,
    )
    # Exact UI parity when CV is preferred: only strongly observed
    # one-tone narrow candidates survive. This is *not* Morse ID.
    narrow = [x for x in proposals
              if x["tone_count_candidate"] == 1
              and x["bandwidth_hz"] <= 45
              and x["peak_snr_db"] >= 14]
    matched = False
    if truth is not None:
        matched = any(
            x["sample_start"] < truth["start_sample"] + truth["sample_count"]
            and x["sample_start"] + x["sample_count"] > truth["start_sample"]
            and truth["occupied_low_hz"] <= x["center_hz"] <= truth["occupied_high_hz"]
            for x in narrow
        )
    return {"matched_truth": matched, "all_tracks": len(proposals),
            "narrow_candidates": len(narrow),
            "wall_ms": round((time.perf_counter()-t0)*1000,2)}


def score_case(*, mode: str, snr_db: float, seed: int, output_dir: Path,
               detector=None, duration: int = 30, burst_seconds: float = 9,
               rate: int = 12000, compare_legacy: bool = False) -> dict:
    path = output_dir / ("lab-" + mode + "-" + str(snr_db).replace("-", "neg") + ".wav")
    manifest = signal_lab.render(
        modes=[mode], output=path, duration=duration, sample_rate=rate,
        traffic="sparse", max_active=1, bursts_per_mode=1,
        burst_seconds=burst_seconds, stereo=False, qrm=False,
        seed=seed, snr_db=snr_db, snr_reference_hz=2500,
    )
    pcm, rate, wav_sha = read_mono(manifest["mono_wav"])
    [burst] = manifest["transmissions"]
    [first, *rest] = burst["embedded_messages"]
    last = rest[-1] if rest else first
    full_start, full_end = first["sample_start"], first["sample_start"] + first["sample_count"]
    carrier = burst["carrier_hz"]
    # Oracle fixture window: provides an upper bound on modem performance.
    oracle = decode_attempt(pcm, rate, full_start, full_end, carrier, mode)
    # Live track crop is no more than four seconds, starts *mid-transmission*
    # and has no access to the exact message boundaries.
    live_start = first["sample_start"] + round(0.35 * rate)
    live_end = min(live_start + 4 * rate, len(pcm))
    full_messages_inside_crop = sum(
        message["sample_start"] >= live_start
        and message["sample_start"] + message["sample_count"] <= live_end
        for message in burst["embedded_messages"]
    )
    live = decode_attempt(pcm, rate, live_start, live_end, carrier, mode)
    # A separate noise-only 4s interval after all bursts.
    noise_start = len(pcm) - 4 * rate
    assert burst["start_sample"] + burst["sample_count"] < noise_start
    noise = decode_attempt(pcm, rate, noise_start, len(pcm), carrier, mode)
    legacy = None
    if compare_legacy:
        # Same normalized input and modem implementation, but exactly the
        # pre-segmentation worker's single whole-window decode behavior.
        old_start = time.perf_counter()
        old_result = create_library().decode(mode, DecodeWindow(
            pcm[live_start:live_end].astype(np.float64) / 32768.0,
            rate, sample_start=live_start,
            carrier_hz=1200.0 if mode == "ax25" else carrier,
        ))
        old_grade, old_text = _evidence_grade(old_result)
        legacy = {
            "status": old_result.status, "grade": old_grade,
            "text": old_text[:160],
            "correct_payload": bool(first["expected_text"] in old_text)
                if mode != "ax25" else bool(old_grade == "frame-integrity-passed"
                                            and first["expected_text"] in old_text),
            "wall_ms": round((time.perf_counter()-old_start)*1000, 2),
        }
    info = {
        "mode": mode, "snr_db": snr_db, "seed": seed, "sample_rate": rate,
        "mono_sha256": wav_sha, "fixture_sha256": burst["fixture_sha256"],
        "carrier_hz": carrier,
        "message_count": len(burst["embedded_messages"]),
        "message_text": first["expected_text"],
        "oracle": {
            "status": oracle["attempted"][0]["status"],
            "grade": oracle["grade"], "text": oracle["text"][:160],
            "correct": payload_correct(oracle, first["expected_text"], mode),
            "wall_ms": oracle["wall_ms"], "cpu_ms": oracle["cpu_ms"],
        },
        "midburst_four_second": {
            "start_sample": live_start, "sample_count": live_end-live_start,
            "fully_embedded_messages": full_messages_inside_crop,
            "status": live["attempted"][0]["status"],
            "grade": live["grade"], "text": live["text"][:160],
            # Exact message matching is conservative: partial/interleaved
            # fragments are not called correct even if readable.
            "correct_payload": bool(first["expected_text"] in live["text"])
                if mode != "ax25" else bool(live["grade"] == "frame-integrity-passed"
                                           and first["expected_text"] in live["text"]),
            "nonempty_tentative": live["grade"] != "none",
            "wall_ms": live["wall_ms"], "cpu_ms": live["cpu_ms"],
        },
        "noise_only": {
            "start_sample": noise_start, "sample_count": len(pcm)-noise_start,
            "grade": noise["grade"], "status": noise["attempted"][0]["status"],
            "false_output": noise["grade"] != "none",
            "text": noise["text"][:160],
        },
        "cv": None,
        "legacy_four_second": legacy,
        "heuristic": {
            "active": narrow_tone_metrics(
                pcm, rate, start=rate, stop=11*rate, truth=burst),
            "noise_only": narrow_tone_metrics(
                pcm, rate, start=len(pcm)-10*rate, stop=len(pcm), truth=None),
        },
    }
    if detector is not None:
        # A 10-second window with most of the active burst, plus a separate
        # 10-second noise-only window. Count matched detections and unpaired
        # boxes; these are geometrical detections, not classified identities.
        active_stop = min(len(pcm), 11 * rate)
        active_window = pcm[active_stop-10*rate:active_stop].astype(np.float64)/32768
        t0 = time.perf_counter()
        active = detector.detect_live_partial(active_window, sample_start=rate,
                         total_sample_count=active_stop, sample_rate=rate)
        active_ms = round((time.perf_counter()-t0)*1000,2)
        background = pcm[-10*rate:].astype(np.float64)/32768
        t0 = time.perf_counter()
        negative = detector.detect_live_partial(background, sample_start=len(pcm)-10*rate,
                         total_sample_count=len(pcm), sample_rate=rate)
        noise_ms = round((time.perf_counter()-t0)*1000,2)
        matches = [d for d in active if detection_matches(d, burst, rate)]
        info["cv"] = {
            "model_sha256": detector.model_sha256,
            "matched_truth": bool(matches),
            "matched_boxes": len(matches),
            "active_boxes": len(active),
            "unmatched_active_boxes": len(active)-len(matches),
            "noise_only_boxes": len(negative),
            "active_wall_ms": active_ms, "noise_wall_ms": noise_ms,
        }
    return info


def summary(rows: list[dict]) -> dict:
    by_snr = {}
    by_mode = {}
    for field, destination in (("snr_db", by_snr), ("mode", by_mode)):
        for key in sorted({row[field] for row in rows}):
            group = [r for r in rows if r[field] == key]
            cv = [r["cv"] for r in group if r["cv"] is not None]
            destination[str(key)] = {
                "count": len(group),
                "oracle_exact": sum(r["oracle"]["correct"] for r in group),
                "four_sec_contains_payload": sum(r["midburst_four_second"]["correct_payload"] for r in group),
                "four_sec_evaluable_complete_message": sum(
                    r["midburst_four_second"]["fully_embedded_messages"] > 0 for r in group),
                "legacy_four_sec_correct": sum(
                    bool(r["legacy_four_second"]["correct_payload"]) for r in group
                    if r.get("legacy_four_second") is not None),
                "legacy_four_sec_scored": sum(
                    r.get("legacy_four_second") is not None for r in group),
                "noise_false_outputs": sum(r["noise_only"]["false_output"] for r in group),
                "cv_detected": sum(r["matched_truth"] for r in cv),
                "cv_scored": len(cv),
                "cv_noise_boxes": sum(r["noise_only_boxes"] for r in cv),
                "narrow_tone_matched": sum(r["heuristic"]["active"]["matched_truth"] for r in group),
                "narrow_tone_noise_candidates": sum(r["heuristic"]["noise_only"]["narrow_candidates"] for r in group),
            }
    return {"by_snr": by_snr, "by_mode": by_mode}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--snrs", default=",".join(map(str, DEFAULT_SNRS)))
    parser.add_argument("--seed", type=int, default=71062)
    parser.add_argument("--sample-rate", type=int, choices=(12000, 24000, 48000), default=12000,
                        help="48 kHz matches the normal Signal Lab live audio stream")
    parser.add_argument("--cv-model", type=Path, help="optional pinned YOLO model file")
    parser.add_argument("--compare-legacy", action="store_true",
                        help="also run original single-full-window decoding on identical PCM")
    parser.add_argument("--keep-wavs", type=Path, help="keep generated WAV files here")
    parser.add_argument("--output", type=Path, help="JSON report")
    args = parser.parse_args()
    modes = args.modes.split(",")
    snrs = [float(x) for x in args.snrs.split(",")]
    if not modes or len(modes) != len(set(modes)) or any(x not in MODES for x in modes):
        parser.error("choose distinct modes from cw,rtty,bpsk31,ax25")
    if not snrs or len(snrs) > 20 or any(not np.isfinite(s) or s < -30 or s > 55 for s in snrs):
        parser.error("SNRs must be finite, -30 to 55 dB (at most twenty)")
    detector = None
    if args.cv_model:
        from sigid.cv_detector import YoloSignalDetector
        detector = YoloSignalDetector(args.cv_model)
    if args.keep_wavs:
        directory = args.keep_wavs.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        tmp = None
    else:
        tmp = tempfile.TemporaryDirectory(prefix="sigid-snr-benchmark-")
        directory = Path(tmp.name)
    try:
        rows = []
        for mode in modes:
            for i,snr in enumerate(snrs):
                row = score_case(mode=mode, snr_db=snr, seed=args.seed + i,
                                 output_dir=directory, detector=detector,
                                 rate=args.sample_rate, compare_legacy=args.compare_legacy)
                rows.append(row)
                print("%-6s %5.1f dB oracle=%s four-sec=%s noise-fp=%s CV=%s" %
                      (mode, snr, row["oracle"]["correct"], row["midburst_four_second"]["correct_payload"],
                       row["noise_only"]["false_output"],
                       None if row["cv"] is None else row["cv"]["matched_truth"]), flush=True)
        report = {
            "version": 1, "provenance": "synthetic-signal-lab-only",
            "not_on_air_validation": True,
            "reference_noise_bandwidth_hz": 2500,
            "seed": args.seed, "sample_rate_hz": args.sample_rate,
            "legacy_comparison_enabled": args.compare_legacy,
            "cv_model_sha256": detector.model_sha256 if detector else None,
            "summary": summary(rows), "cases": rows,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report["summary"], indent=2))
        return 0
    finally:
        if tmp:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
