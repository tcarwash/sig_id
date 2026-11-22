#!/usr/bin/env python3
"""Synthetic API benchmark for native live-decoder admission (never opens audio)."""
from __future__ import annotations
import argparse
import asyncio
import json
import statistics
import time
import wave
from pathlib import Path

import httpx

from sigid.api import create_app


def pct(values, fraction):
    values = sorted(values)
    return round(values[min(len(values)-1, max(0, int(len(values)*fraction+.999)-1))], 2)


async def benchmark(workspace: Path, iterations: int):
    manifest = json.loads((Path(__file__).resolve().parents[1] /
        "tests/fixtures/decoder_conformance/rtty/manifest.json").read_text())
    fixture = next(c for c in manifest["fixtures"] if c["expected"]["kind"] == "positive")
    with wave.open(str(Path(__file__).resolve().parents[1] /
             "tests/fixtures/decoder_conformance/rtty" / fixture["audio"]["path"]), "rb") as wav:
        samples = wav.readframes(wav.getnframes())
        rate = wav.getframerate()
    app = create_app(workspace)
    app.state.classifier = None

    class Detector:
        model_sha256 = "a" * 64
        model_version = "benchmark"
        def detect_live_partial(self, mono, *, sample_start, total_sample_count,
                                sample_rate, confidence=None):
            return [{"sample_start": 0, "sample_count": len(mono),
                     "frequency_low_hz": fixture["carrier_hz"] - 110,
                     "frequency_high_hz": fixture["carrier_hz"] + 110,
                     "confidence": .8}]

    app.state.cv_detector = Detector()
    snapshot = {
        "pcm": samples, "active": True, "session_id": "perf-native",
        "sample_offset": 0, "sample_count": len(samples)//2,
        "window_sample_count": len(samples)//2, "channels": 1, "sample_rate": rate,
    }
    app.state.live_capture.analysis_audio = lambda: snapshot
    base = {"session_id": "perf-native", "include_classifier": "false"}
    results = {"baseline_ms": [], "native_enabled_ms": []}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://bench", timeout=45) as client:
        for enabled, name in ((False, "baseline_ms"), (True, "native_enabled_ms")):
            for index in range(iterations):
                snapshot["sample_count"] = len(samples) // 2 + index
                t = time.perf_counter()
                reply = await client.get("/api/live/cv-detections", params={
                    **base, "auto_native_decode": str(enabled).lower()})
                reply.raise_for_status()
                results[name].append((time.perf_counter()-t)*1000)
                if enabled:
                    await asyncio.sleep(.025)
        deadline = time.monotonic()+15
        while time.monotonic() < deadline:
            result = await client.get("/api/live/native-decode/status",
                                      params={"session_id": "perf-native"})
            result.raise_for_status()
            metrics = result.json()
            if not metrics["busy"]:
                break
            await asyncio.sleep(.025)
        else:
            raise TimeoutError("native decoder did not finish")
    app.state.native_live_runner.close()
    summary = {
        "description": "In-process API CV request times, synthetic RTTY, no classifier or real audio device",
        "iterations_per_configuration": iterations,
        "baseline_median_ms": round(statistics.median(results["baseline_ms"]), 2),
        "baseline_p95_ms": pct(results["baseline_ms"], .95),
        "native_enabled_median_ms": round(statistics.median(results["native_enabled_ms"]), 2),
        "native_enabled_p95_ms": pct(results["native_enabled_ms"], .95),
        "worker_cpu_ms_last_minute": metrics["cpu_ms_last_minute"],
        "worker_single_core_percent_last_minute": metrics["cpu_pct_single_core_last_minute"],
        "worker_stats": metrics,
        "user_stream_tested": False,
    }
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=None)
    parser.add_argument("--iterations",type=int,default=14)
    parser.add_argument("--state",type=Path,default=Path("/tmp/sigid-native-live-bench"))
    args=parser.parse_args()
    if not 5 <= args.iterations <= 50:
        parser.error("iterations 5..50 required")
    report=asyncio.run(benchmark(args.state,args.iterations))
    text=json.dumps(report,indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text+"\n")


if __name__=="__main__":
    main()
