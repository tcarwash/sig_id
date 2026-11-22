"""Independently synthesized textbook Morse and negative audio conformance fixtures.

A fixtures-only generator: it never imports the decoder being evaluated.
Regenerate to verify pinned, deterministic PCM16 WAV SHA-256 values.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

import numpy as np

RATE = 12_000
CARRIER = 1100.0
# Independent hand-transcribed International Morse patterns needed by examples.
SYMBOLS = {
    "C": "-.-.", "Q": "--.-", "T": "-", "E": ".", "S": "...", "O": "---",
}


def keyed_morse(message: str, wpm: float) -> np.ndarray:
    dot = 1.2 / wpm
    events: list[tuple[float, bool]] = [(0.25, False)]
    words = message.split(" ")
    for wi, word in enumerate(words):
        for ci, char in enumerate(word):
            code = SYMBOLS[char]
            for si, symbol in enumerate(code):
                events.append((dot * (3 if symbol == "-" else 1), True))
                if si != len(code) - 1:
                    events.append((dot, False))
            if ci != len(word) - 1:
                events.append((dot * 3, False))
        if wi != len(words) - 1:
            events.append((dot * 7, False))
    events.append((0.30, False))
    gate = np.concatenate([
        np.full(int(round(seconds * RATE)), pressed, dtype=np.float64)
        for seconds, pressed in events
    ])
    # A smooth ~7ms key edge avoids RF-shaped clicks. This is not an RX algorithm.
    kernel = np.full(85, 1.0 / 85)
    gate = np.convolve(gate, kernel, mode="same")
    times = np.arange(len(gate), dtype=np.float64) / RATE
    signal = 0.5 * gate * np.sin(2 * np.pi * CARRIER * times)
    signal += 0.18 * np.sin(2 * np.pi * (CARRIER + 210) * times)
    signal += np.random.default_rng(1201).normal(0, 0.003, len(gate))
    return signal


def constant_carrier() -> np.ndarray:
    times = np.arange(8 * RATE, dtype=np.float64) / RATE
    return 0.4 * np.sin(2 * np.pi * CARRIER * times)


def noise_floor() -> np.ndarray:
    return np.random.default_rng(23).normal(0, 0.001, 8 * RATE)


def hopped_tones() -> np.ndarray:
    # FSK-like repetitive tone activity, not Morse.
    count = 8 * RATE
    t = np.arange(count, dtype=np.float64) / RATE
    rng = np.random.default_rng(51)
    tones = rng.integers(0, 8, 52)
    freq = CARRIER + 6.25 * tones[np.minimum((t / 0.16).astype(int), 51)]
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    return 0.3 * np.sin(phase) + np.random.default_rng(52).normal(0, 0.001, count)


def write_fixture(out: Path, filename: str, values: np.ndarray) -> str:
    path = out / filename
    pcm = np.rint(np.clip(values, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixtures(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cases = [
        ("cw_cq_test", "cw-cq-test.wav", keyed_morse("CQ TEST", 20.0),
         {"kind": "positive", "text": "CQ TEST", "allowed_statuses": ["decoded"]}),
        ("cw_sos", "cw-sos.wav", keyed_morse("SOS", 30.0),
         {"kind": "positive", "text": "SOS", "allowed_statuses": ["decoded"]}),
        ("cw_constant_carrier", "cw-carrier.wav", constant_carrier(),
         {"kind": "negative"}),
        ("cw_noise_floor", "cw-noise.wav", noise_floor(),
         {"kind": "negative"}),
        ("cw_unrelated_hopping", "cw-hopping.wav", hopped_tones(),
         {"kind": "negative"}),
    ]
    fixtures = []
    for key, filename, values, expected in cases:
        digest = write_fixture(out, filename, values)
        fixtures.append({
            "id": key,
            "decoder_id": "cw",
            "provenance": {
                "kind": "synthetic",
                "description": (
                    "Independent textbook Morse generator, 1:3:7 timing with 7ms key ramp, "
                    "plus adjacent tone and seeded AWGN."
                    if expected["kind"] == "positive"
                    else "Independent synthetic negative audio; no transmitted Morse message."
                ),
                "generator": "tools/build_decoder_conformance_cw_fixtures.py",
                "generator_version": "1",
            },
            "audio": {
                "path": filename, "sha256": digest,
                "window_offset_samples": 0,
                "file_start_sample": 17000,
            },
            "carrier_hz": CARRIER,
            "expected": expected,
        })
    manifest = {
        "schema_version": 1, "suite_id": "cw-synthetic-v1",
        "description": "Deterministic synthetic vectors; not a reviewed on-air benchmark.",
        "fixtures": fixtures,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "decoder_conformance" / "cw"
    result = build_fixtures(target)
    print(f"Wrote {len(result['fixtures'])} deterministic fixtures to {target}")
