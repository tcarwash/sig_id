#!/usr/bin/env python3
"""Deterministic API scheduling benchmark, synthetic CV + synthetic slow classifier.

This measures the *orchestration* improvement, not real YOLO/classifier accuracy
or on-air decoding. It never reads an audio device or starts RF equipment.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import statistics
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np

from sigid.live_classifier_cache import LiveClassifierEvidenceCache

api_module = importlib.import_module("sigid.api")


class FakeCv:
    model_sha256 = "a" * 64

    def detect_live_partial(self, samples, *, sample_start,
                            total_sample_count, sample_rate, confidence=None):
        return [{
            "sample_start": total_sample_count - 2 * sample_rate,
            "sample_count": 2 * sample_rate,
            "frequency_low_hz": 200 + index * 180,
            "frequency_high_hz": 270 + index * 180,
            "confidence": round(.99 - .01 * index, 3),
        } for index in range(12)]


async def scenario(root: Path, *, cache_enabled: bool, polls: int,
                   delay_ms: float) -> dict:
    real_cache = api_module.LiveClassifierEvidenceCache
    try:
        api_module.LiveClassifierEvidenceCache = (
            lambda: LiveClassifierEvidenceCache(
                refresh_seconds=6.0 if cache_enabled else 0.000001))
        app = api_module.create_app(root)
    finally:
        api_module.LiveClassifierEvidenceCache = real_cache
    app.state.cv_detector = FakeCv()
    app.state.classifier = {"metadata": {
        "model_sha256": "c" * 64,
        "classifier_window_seconds": 1.0,
    }}
    rate = 12000
    audio = np.zeros(rate * 8, dtype="<i2")
    snapshot = {
        "pcm": audio[:rate * 3].tobytes(), "sample_rate": rate,
        "channels": 1, "sample_offset": 0, "sample_count": rate * 3,
        "window_sample_count": rate * 3,
        "session_id": "bench", "active": True,
    }
    app.state.live_capture.analysis_audio = lambda: snapshot
    count = [0]
    def mock_classifier(classifier, crop, sample_rate, low_hz, high_hz):
        count[0] += 1
        time.sleep(delay_ms / 1000)
        return {"available": True, "abstained": False, "method": "BPSK31",
                "family": "PSK", "confidence": 0.75}
    real_classify = api_module._classify_signal_crop
    api_module._classify_signal_crop = mock_classifier
    timings = []
    scans = []
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://bench", timeout=30) as client:
            for i in range(polls):
                # Advances original sample position but keeps a stable CV
                # geometry and track, as happens with overlapping live scans.
                frames = rate * 3 + i * 1000
                snapshot.update(
                    pcm=audio[:frames].tobytes(),
                    sample_count=frames, window_sample_count=frames)
                t = time.perf_counter()
                response = await client.get("/api/live/cv-detections",
                    params={"session_id": "bench", "include_classifier": "true"})
                response.raise_for_status()
                delta_ms = (time.perf_counter() - t) * 1000
                timings.append(delta_ms)
                j = response.json()
                scans.append({
                    "latency_ms": round(delta_ms, 2),
                    "scored": j["classifier_scored_this_scan"],
                    "reused": j["classifier_reused_this_scan"],
                    "deferred": j["classifier_deferred_this_scan"],
                })
        cache_stats = app.state.classifier_evidence_cache.status("bench")
    finally:
        api_module._classify_signal_crop = real_classify
        app.state.native_live_runner.close()

    return {
        "cache_enabled": cache_enabled,
        "polls": polls,
        "synthetic_scoring_delay_ms": delay_ms,
        "classifier_calls": count[0],
        "median_http_ms": round(statistics.median(timings), 2),
        "total_http_ms": round(sum(timings), 2),
        "cache_stats": cache_stats,
        "scans": scans,
    }


async def main(polls: int, delay_ms: float):
    with tempfile.TemporaryDirectory(prefix="sigid-live-chain-bench-") as temp:
        base = Path(temp)
        without = await scenario(base / "baseline", cache_enabled=False,
                                 polls=polls, delay_ms=delay_ms)
        with_cache = await scenario(base / "optimized", cache_enabled=True,
                                    polls=polls, delay_ms=delay_ms)
    return {
        "scope": "synthetic HTTP ASGI orchestration; no hardware, no real model inference",
        "baseline": without, "optimized": with_cache,
        "classifier_calls_saved": without["classifier_calls"] - with_cache["classifier_calls"],
        "total_http_speedup_factor": round(
            without["total_http_ms"] / max(1, with_cache["total_http_ms"]), 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--polls", type=int, default=6)
    parser.add_argument("--scorer-delay-ms", type=float, default=8.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 3 <= args.polls <= 30 or not 0 <= args.scorer_delay_ms <= 100:
        parser.error("polls 3–30; delay 0–100ms")
    report = asyncio.run(main(args.polls, args.scorer_delay_ms))
    result = json.dumps(report, indent=2) + "\n"
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
