"""Deterministic, labeled baseband audio fixtures for modulation research."""

from __future__ import annotations

import json
import math
from pathlib import Path
import argparse
import wave


def generate_2fsk_wav(
    output: str | Path,
    bits: str,
    *,
    sample_rate: int = 12_000,
    symbol_rate: int = 100,
    tone_low_hz: float = 1_000.0,
    tone_high_hz: float = 1_200.0,
    amplitude: float = 0.6,
) -> dict[str, object]:
    """Write a clean phase-continuous 2-FSK WAV and generation manifest.

    This represents binary FSK only; it carries no named protocol, framing,
    whitening, error correction, or bit polarity convention.
    """
    output = Path(output)
    if not bits or any(bit not in "01" for bit in bits):
        raise ValueError("bits must be a non-empty string containing only 0 and 1")
    if sample_rate <= 0 or symbol_rate <= 0 or sample_rate % symbol_rate:
        raise ValueError("sample_rate must be a positive integer multiple of symbol_rate")
    if not (0.0 < tone_low_hz < tone_high_hz < sample_rate / 2):
        raise ValueError("tones must satisfy 0 < low < high < Nyquist")
    if not (0.0 < amplitude <= 1.0):
        raise ValueError("amplitude must be in (0, 1]")

    samples_per_symbol = sample_rate // symbol_rate
    phase = 0.0
    pcm = bytearray()
    for bit in bits:
        frequency = tone_high_hz if bit == "1" else tone_low_hz
        phase_step = 2.0 * math.pi * frequency / sample_rate
        for _ in range(samples_per_symbol):
            value = round(math.sin(phase) * amplitude * 32767)
            pcm.extend(int(value).to_bytes(2, "little", signed=True))
            phase = (phase + phase_step) % (2.0 * math.pi)

    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "source_kind": "synthetic",
        "generator": "sig-id-2fsk",
        "generator_version": "0.1.0",
        "deterministic": True,
        "waveform": output.name,
        "sample_rate": sample_rate,
        "sample_count": len(bits) * samples_per_symbol,
        "sample_width_bytes": 2,
        "channels": 1,
        "labels": {
            "modulation_family": "FSK",
            "modulation_method": "binary frequency shift keying",
            "protocol": None,
            "variant": None,
        },
        "parameters": {
            "symbol_rate": symbol_rate,
            "tone_count": 2,
            "tone_low_hz": tone_low_hz,
            "tone_high_hz": tone_high_hz,
            "amplitude": amplitude,
            "bit_mapping": {"0": tone_low_hz, "1": tone_high_hz},
            "phase_continuous": True,
        },
        "reference_bits": bits,
        "impairments": [],
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a clean labeled binary-FSK WAV fixture")
    parser.add_argument("output", type=Path)
    parser.add_argument("bits", help="reference bit sequence, for example 00110110")
    parser.add_argument("--sample-rate", type=int, default=12_000)
    parser.add_argument("--symbol-rate", type=int, default=100)
    parser.add_argument("--tone-low-hz", type=float, default=1_000.0)
    parser.add_argument("--tone-high-hz", type=float, default=1_200.0)
    parser.add_argument("--amplitude", type=float, default=0.6)
    args = parser.parse_args()
    manifest = generate_2fsk_wav(
        args.output,
        args.bits,
        sample_rate=args.sample_rate,
        symbol_rate=args.symbol_rate,
        tone_low_hz=args.tone_low_hz,
        tone_high_hz=args.tone_high_hz,
        amplitude=args.amplitude,
    )
    print(json.dumps(manifest, indent=2))
