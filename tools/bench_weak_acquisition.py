"""Controlled, receive-only weak RTTY and CW lookback acquisition sweep.

Benchmarks first-party native worker against an identical raw full-window
decoder, with deterministic Signal Lab noise controls and sample provenance.
No sound hardware or live server access. Not a field error-rate estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from sigid.libsigdecode import DecodeWindow, create_library
from sigid.native_live_decode import _evidence_grade, decode_live_candidate
from sigid.signal_lab import render
from tools.bench_signal_chain_snr import read_mono


def score_case(*, mode: str, snr_db: float, seed: int, sample_rate: int,
               output_dir: Path) -> dict:
    if mode not in ("cw", "rtty"):
        raise ValueError("this sweep evaluates CW and RTTY only")
    seconds = 8 if mode == "cw" else 4
    data = render(
        modes=[mode],
        output=output_dir / f"{mode}-{sample_rate}-{snr_db}-{seed}.wav",
        duration=30, sample_rate=sample_rate, seed=seed,
        snr_db=snr_db, snr_reference_hz=2500,
        traffic="sparse", bursts_per_mode=1,
        burst_seconds=16 if mode == "cw" else 9,
        qrm=False, stereo=False,
    )
    pcm, rate, wav_sha = read_mono(data["mono_wav"])
    [burst] = data["transmissions"]
    [first] = burst["embedded_messages"][:1]
    expected = first["expected_text"]
    start = first["sample_start"] + round(.35 * rate)
    end = start + seconds * rate
    assert end <= len(pcm)
    full_inside = sum(
        m["sample_start"] >= start
        and m["sample_start"] + m["sample_count"] <= end
        for m in burst["embedded_messages"]
    )
    assert full_inside > 0
    crop = pcm[start:end]
    new = decode_live_candidate(
        crop.tobytes(), rate, start, burst["carrier_hz"], (mode,)
    )
    noise_start = len(pcm) - seconds * rate
    assert burst["start_sample"] + burst["sample_count"] < noise_start
    noise = decode_live_candidate(
        pcm[noise_start:].tobytes(), rate, noise_start,
        burst["carrier_hz"], (mode,),
    )
    old_window = crop[-4*rate:] if mode == "cw" else crop
    old_start = end - len(old_window)
    t0 = time.perf_counter()
    old = create_library().decode(
        mode, DecodeWindow(
            old_window.astype(np.float64)/32768.0,
            rate, sample_start=old_start, carrier_hz=burst["carrier_hz"],
        )
    )
    old_grade, old_text = _evidence_grade(old)
    old_ms = round((time.perf_counter()-t0)*1000, 2)
    # The initial 4s crop begins 350 ms after the first message began.
    # It is not the same as the 4s *ending* at the longer window's edge;
    # retain both controls rather than choosing the more flattering baseline.
    initial = None
    if mode == "cw":
        old_front = create_library().decode(mode, DecodeWindow(
            crop[:4*rate].astype(np.float64)/32768.0,
            rate, sample_start=start, carrier_hz=burst["carrier_hz"],
        ))
        front_grade, front_text = _evidence_grade(old_front)
        initial = {"grade": front_grade, "text": front_text,
                   "expected_present": expected in front_text}

    if mode == "cw":
        # CW is a stream of characters, not packet framing; retaining a full
        # repeated message is more realistic than requiring *only* that text.
        correct_new = expected in new["text"]
        correct_old = expected in old_text
    else:
        correct_new = new["text"] == expected
        correct_old = old_text == expected
    return {
        "mode": mode, "snr_db": snr_db, "seed": seed,
        "sample_rate": rate, "source_sha256": wav_sha,
        "fixture_sha256": burst["fixture_sha256"],
        "expected_text": expected,
        "carrier_hz": burst["carrier_hz"],
        "sample_start": start, "sample_count": len(crop),
        "complete_messages_inside_crop": full_inside,
        "initial_four_seconds": initial,
        "legacy": {
            "sample_start": old_start, "sample_count": len(old_window),
            "grade": old_grade, "text": old_text,
            "expected_present": correct_old, "wall_ms": old_ms,
        },
        "updated": {
            "grade": new["grade"], "text": new["text"],
            "expected_present": correct_new,
            "incorrect_coherent": new["grade"] != "none" and not correct_new,
            "wall_ms": new["wall_ms"], "cpu_ms": new["cpu_ms"],
            "acquisition": new["attempted"][0].get("acquisition"),
        },
        "noise": {
            "grade": noise["grade"], "text": noise["text"],
            "false_output": noise["grade"] != "none",
            "cpu_ms": noise["cpu_ms"],
        },
        "synthetic_only": True,
    }


def summarize(rows: list[dict]) -> dict:
    groups = {}
    for mode in ("cw", "rtty"):
        for snr in sorted({r["snr_db"] for r in rows}):
            case_rows = [r for r in rows if r["mode"] == mode and r["snr_db"] == snr]
            if not case_rows:
                continue
            groups[f"{mode}:{snr}"] = {
                "cases": len(case_rows),
                "legacy_expected_present": sum(r["legacy"]["expected_present"] for r in case_rows),
                "updated_expected_present": sum(r["updated"]["expected_present"] for r in case_rows),
                "incorrect_coherent_unverified": sum(r["updated"]["incorrect_coherent"] for r in case_rows),
                "noise_false_outputs": sum(r["noise"]["false_output"] for r in case_rows),
                "updated_cpu_ms_median": round(float(np.median(
                    [r["updated"]["cpu_ms"] for r in case_rows])), 2),
            }
    return groups


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modes", default="cw,rtty")
    p.add_argument("--snrs", default="-10,-5,0,10")
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--seed-start", type=int, default=71062)
    p.add_argument("--sample-rate", type=int, choices=(12000,48000),default=48000)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    modes = args.modes.split(",")
    snrs = [float(x) for x in args.snrs.split(",")]
    if any(mode not in ("cw","rtty") for mode in modes) or len(modes)!=len(set(modes)):
        p.error("choose distinct cw and/or rtty")
    if not snrs or any(not np.isfinite(s) or not -30 <= s <= 55 for s in snrs):
        p.error("SNRs must be finite -30..55")
    if not 1 <= args.seeds <= 30:
        p.error("seeds must be 1..30")
    with tempfile.TemporaryDirectory(prefix="sigid-weak-sweep-") as tmp:
        rows = []
        for seed in range(args.seed_start,args.seed_start+args.seeds):
            for mode in modes:
                for snr in snrs:
                    result = score_case(
                        mode=mode,snr_db=snr,seed=seed,
                        sample_rate=args.sample_rate,output_dir=Path(tmp),
                    )
                    rows.append(result)
                    print(f"{mode:<5} snr={snr:5.1f} seed={seed} "
                          f"old={result['legacy']['expected_present']} "
                          f"new={result['updated']['expected_present']} "
                          f"noise-fp={result['noise']['false_output']}",flush=True)
        report = {
            "schema_version":1,"provenance":"synthetic-signal-lab-only",
            "not_on_air_validation":True,
            "sample_rate":args.sample_rate,"seed_start":args.seed_start,
            "seed_count":args.seeds,"snrs":snrs,
            "summary":summarize(rows),"cases":rows,
        }
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report["summary"],indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
