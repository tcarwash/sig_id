"""Deterministic, labeled baseband audio fixtures for modulation research."""

from __future__ import annotations

import json
import math
from pathlib import Path
import argparse
import wave
import random
from collections.abc import Sequence


def generate_noise_wav(
    output: str | Path,
    *,
    seed: int,
    sample_rate: int = 12_000,
    sample_count: int = 24_000,
    amplitude: float = 0.25,
) -> dict[str, object]:
    """Write deterministic uniform noise as an explicitly unlabeled negative."""
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate < 8000:
        raise ValueError("sample_rate must be an integer of at least 8000 Hz")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 1024:
        raise ValueError("sample_count must be an integer of at least 1024")
    if not 0 < amplitude <= 1:
        raise ValueError("amplitude must be in (0, 1]")
    output = Path(output)
    pcm = bytearray()
    rng = random.Random(seed)
    for _ in range(sample_count):
        value = round(rng.uniform(-amplitude, amplitude) * 32767)
        pcm.extend(int(value).to_bytes(2, "little", signed=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source_kind": "synthetic",
        "generator": "sig-id-noise",
        "generator_version": "0.1.0",
        "deterministic": True,
        "waveform": output.name,
        "sample_rate": sample_rate,
        "sample_count": sample_count,
        "sample_width_bytes": 2,
        "channels": 1,
        "labels": {"modulation_family": None, "modulation_method": None, "protocol": None, "variant": None},
        "parameters": {"seed": seed, "amplitude": amplitude, "distribution": "uniform", "negative_kind": "noise_only"},
        "impairments": [],
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


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
    if (
        not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or not isinstance(symbol_rate, int)
        or isinstance(symbol_rate, bool)
        or sample_rate <= 0
        or symbol_rate <= 0
        or sample_rate % symbol_rate
    ):
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


def generate_bpsk_wav(
    output: str | Path,
    bits: str,
    *,
    sample_rate: int = 12_000,
    symbol_rate: int = 100,
    carrier_hz: float = 1_500.0,
    amplitude: float = 0.6,
    initial_phase_radians: float = 0.0,
) -> dict[str, object]:
    """Generate a deterministic uncoded BPSK passband fixture and manifest."""
    output = Path(output)
    if not bits or any(bit not in "01" for bit in bits):
        raise ValueError("bits must be a non-empty string containing only 0 and 1")
    if (
        not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or not isinstance(symbol_rate, int)
        or isinstance(symbol_rate, bool)
        or sample_rate <= 0
        or symbol_rate <= 0
        or sample_rate % symbol_rate
    ):
        raise ValueError("sample_rate must be a positive integer multiple of symbol_rate")
    if not math.isfinite(carrier_hz) or not (0 < carrier_hz < sample_rate / 2):
        raise ValueError("carrier_hz must be finite and between 0 and Nyquist")
    if not (0.0 < amplitude <= 1.0) or not math.isfinite(initial_phase_radians):
        raise ValueError("amplitude must be in (0, 1] and phase must be finite")

    samples_per_symbol = sample_rate // symbol_rate
    phase = initial_phase_radians
    phase_step = 2.0 * math.pi * carrier_hz / sample_rate
    pcm = bytearray()
    for bit in bits:
        bit_phase = math.pi if bit == "1" else 0.0
        for _ in range(samples_per_symbol):
            value = round(math.sin(phase + bit_phase) * amplitude * 32767)
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
        "generator": "sig-id-bpsk",
        "generator_version": "0.1.0",
        "deterministic": True,
        "waveform": output.name,
        "sample_rate": sample_rate,
        "sample_count": len(bits) * samples_per_symbol,
        "sample_width_bytes": 2,
        "channels": 1,
        "labels": {
            "modulation_family": "PSK",
            "modulation_method": "binary phase-shift keying",
            "protocol": None,
            "variant": None,
        },
        "parameters": {
            "symbol_rate": symbol_rate,
            "carrier_hz": carrier_hz,
            "amplitude": amplitude,
            "phase_mapping": {"0": 0.0, "1": math.pi},
            "initial_phase_radians": initial_phase_radians,
            "phase_continuous_carrier": True,
            "protocol_framing": None,
        },
        "reference_bits": bits,
        "impairments": [],
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def generate_qpsk_wav(
    output: str | Path,
    bits: str,
    *,
    sample_rate: int = 12_000,
    symbol_rate: int = 100,
    carrier_hz: float = 1_500.0,
    amplitude: float = 0.6,
    initial_phase_radians: float = 0.0,
) -> dict[str, object]:
    """Generate uncoded Gray-mapped QPSK passband audio with exact source bits."""
    output = Path(output)
    if not bits or any(bit not in "01" for bit in bits) or len(bits) % 2:
        raise ValueError("bits must be a non-empty even-length binary string")
    if (
        not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or not isinstance(symbol_rate, int)
        or isinstance(symbol_rate, bool)
        or sample_rate <= 0
        or symbol_rate <= 0
        or sample_rate % symbol_rate
    ):
        raise ValueError("sample_rate must be a positive integer multiple of symbol_rate")
    if not math.isfinite(carrier_hz) or not (0 < carrier_hz < sample_rate / 2):
        raise ValueError("carrier_hz must be finite and between 0 and Nyquist")
    if not (0.0 < amplitude <= 1.0) or not math.isfinite(initial_phase_radians):
        raise ValueError("amplitude must be in (0, 1] and phase must be finite")

    mapping = {"00": 0.0, "01": math.pi / 2, "11": math.pi, "10": 3 * math.pi / 2}
    samples_per_symbol = sample_rate // symbol_rate
    phase = initial_phase_radians
    phase_step = 2.0 * math.pi * carrier_hz / sample_rate
    pcm = bytearray()
    for index in range(0, len(bits), 2):
        symbol = bits[index : index + 2]
        symbol_phase = mapping[symbol]
        for _ in range(samples_per_symbol):
            value = round(math.sin(phase + symbol_phase) * amplitude * 32767)
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
        "generator": "sig-id-qpsk",
        "generator_version": "0.1.0",
        "deterministic": True,
        "waveform": output.name,
        "sample_rate": sample_rate,
        "sample_count": len(bits) // 2 * samples_per_symbol,
        "sample_width_bytes": 2,
        "channels": 1,
        "labels": {
            "modulation_family": "PSK",
            "modulation_method": "quadrature phase-shift keying",
            "protocol": None,
            "variant": None,
        },
        "parameters": {
            "symbol_rate": symbol_rate,
            "bits_per_symbol": 2,
            "carrier_hz": carrier_hz,
            "amplitude": amplitude,
            "phase_mapping": mapping,
            "initial_phase_radians": initial_phase_radians,
            "phase_continuous_carrier": True,
            "protocol_framing": None,
        },
        "reference_bits": bits,
        "reference_symbols": [bits[index : index + 2] for index in range(0, len(bits), 2)],
        "impairments": [],
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def generate_mfsk_wav(
    output: str | Path,
    symbols: Sequence[int],
    *,
    tone_count: int = 4,
    sample_rate: int = 12_000,
    symbol_rate: int = 100,
    base_tone_hz: float = 1_000.0,
    tone_spacing_hz: float = 200.0,
    amplitude: float = 0.6,
) -> dict[str, object]:
    """Generate clean phase-continuous M-FSK with exact source symbols.

    This creates an uncoded generic modulation example, not a compliant Thor,
    Olivia, Contestia, or other protocol waveform. Symbols use natural binary
    tone indices (zero maps to the base tone). For power-of-two M, the sidecar
    also includes the exact fixed-width bit sequence implied by that mapping.
    """
    output = Path(output)
    if not isinstance(symbols, Sequence) or isinstance(symbols, (str, bytes)) or not symbols:
        raise ValueError("symbols must be a non-empty sequence of tone indices")
    if not isinstance(tone_count, int) or isinstance(tone_count, bool) or not 2 <= tone_count <= 64:
        raise ValueError("tone_count must be from 2 to 64")
    if any(not isinstance(symbol, int) or isinstance(symbol, bool) or not 0 <= symbol < tone_count for symbol in symbols):
        raise ValueError("every symbol must be an integer tone index within tone_count")
    if (
        not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or not isinstance(symbol_rate, int)
        or isinstance(symbol_rate, bool)
        or sample_rate <= 0
        or symbol_rate <= 0
        or sample_rate % symbol_rate
    ):
        raise ValueError("sample_rate must be a positive integer multiple of symbol_rate")
    if (
        not math.isfinite(base_tone_hz)
        or not math.isfinite(tone_spacing_hz)
        or base_tone_hz <= 0
        or tone_spacing_hz <= 0
    ):
        raise ValueError("base_tone_hz and tone_spacing_hz must be positive")
    highest_tone_hz = base_tone_hz + (tone_count - 1) * tone_spacing_hz
    if highest_tone_hz >= sample_rate / 2:
        raise ValueError("all tones must be below Nyquist")
    if not (0.0 < amplitude <= 1.0):
        raise ValueError("amplitude must be in (0, 1]")

    samples_per_symbol = sample_rate // symbol_rate
    phase = 0.0
    pcm = bytearray()
    tone_frequencies = [base_tone_hz + index * tone_spacing_hz for index in range(tone_count)]
    for symbol in symbols:
        phase_step = 2.0 * math.pi * tone_frequencies[symbol] / sample_rate
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

    bits_per_symbol = int(math.log2(tone_count)) if tone_count & (tone_count - 1) == 0 else None
    reference_bits = (
        "".join(f"{symbol:0{bits_per_symbol}b}" for symbol in symbols)
        if bits_per_symbol is not None
        else None
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source_kind": "synthetic",
        "generator": "sig-id-mfsk",
        "generator_version": "0.2.0",
        "deterministic": True,
        "waveform": output.name,
        "sample_rate": sample_rate,
        "sample_count": len(symbols) * samples_per_symbol,
        "sample_width_bytes": 2,
        "channels": 1,
        "labels": {
            "modulation_family": "FSK" if tone_count == 2 else "MFSK",
            "modulation_method": "binary frequency shift keying" if tone_count == 2 else f"{tone_count}-ary frequency shift keying",
            "protocol": None,
            "variant": None,
        },
        "parameters": {
            "symbol_rate": symbol_rate,
            "tone_count": tone_count,
            "base_tone_hz": base_tone_hz,
            "tone_spacing_hz": tone_spacing_hz,
            "tone_frequencies_hz": tone_frequencies,
            "amplitude": amplitude,
            "symbol_mapping": {str(index): frequency for index, frequency in enumerate(tone_frequencies)},
            "bits_per_symbol": bits_per_symbol,
            "bit_order": "most-significant bit first" if bits_per_symbol is not None else None,
            "phase_continuous": True,
            "protocol_framing": None,
        },
        "reference_symbols": list(symbols),
        "reference_bits": reference_bits,
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


def mfsk_main() -> None:
    parser = argparse.ArgumentParser(description="Generate a clean labeled generic M-FSK WAV fixture")
    parser.add_argument("output", type=Path)
    parser.add_argument("symbols", help="comma-separated natural-binary tone indices, e.g. 0,1,3,2")
    parser.add_argument("--tone-count", type=int, default=4)
    parser.add_argument("--sample-rate", type=int, default=12_000)
    parser.add_argument("--symbol-rate", type=int, default=100)
    parser.add_argument("--base-tone-hz", type=float, default=1_000.0)
    parser.add_argument("--tone-spacing-hz", type=float, default=200.0)
    parser.add_argument("--amplitude", type=float, default=0.6)
    args = parser.parse_args()
    try:
        symbols = [int(item) for item in args.symbols.split(",")]
    except ValueError as exc:
        parser.error(f"symbols must be comma-separated integers: {exc}")
    manifest = generate_mfsk_wav(
        args.output,
        symbols,
        tone_count=args.tone_count,
        sample_rate=args.sample_rate,
        symbol_rate=args.symbol_rate,
        base_tone_hz=args.base_tone_hz,
        tone_spacing_hz=args.tone_spacing_hz,
        amplitude=args.amplitude,
    )
    print(json.dumps(manifest, indent=2))


def bpsk_main() -> None:
    parser = argparse.ArgumentParser(description="Generate a clean labeled binary-PSK WAV fixture")
    parser.add_argument("output", type=Path)
    parser.add_argument("bits", help="reference bit sequence, for example 00110110")
    parser.add_argument("--sample-rate", type=int, default=12_000)
    parser.add_argument("--symbol-rate", type=int, default=100)
    parser.add_argument("--carrier-hz", type=float, default=1_500.0)
    parser.add_argument("--amplitude", type=float, default=0.6)
    parser.add_argument("--initial-phase-radians", type=float, default=0.0)
    args = parser.parse_args()
    manifest = generate_bpsk_wav(
        args.output,
        args.bits,
        sample_rate=args.sample_rate,
        symbol_rate=args.symbol_rate,
        carrier_hz=args.carrier_hz,
        amplitude=args.amplitude,
        initial_phase_radians=args.initial_phase_radians,
    )
    print(json.dumps(manifest, indent=2))


def qpsk_main() -> None:
    parser = argparse.ArgumentParser(description="Generate a clean labeled quadrature-PSK WAV fixture")
    parser.add_argument("output", type=Path)
    parser.add_argument("bits", help="even-length reference bit sequence, for example 00110110")
    parser.add_argument("--sample-rate", type=int, default=12_000)
    parser.add_argument("--symbol-rate", type=int, default=100)
    parser.add_argument("--carrier-hz", type=float, default=1_500.0)
    parser.add_argument("--amplitude", type=float, default=0.6)
    parser.add_argument("--initial-phase-radians", type=float, default=0.0)
    args = parser.parse_args()
    manifest = generate_qpsk_wav(
        args.output,
        args.bits,
        sample_rate=args.sample_rate,
        symbol_rate=args.symbol_rate,
        carrier_hz=args.carrier_hz,
        amplitude=args.amplitude,
        initial_phase_radians=args.initial_phase_radians,
    )
    print(json.dumps(manifest, indent=2))
