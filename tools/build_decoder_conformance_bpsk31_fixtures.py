"""Build pinned BPSK31 vectors from ARRL-published Varicode entries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "tests/fixtures/decoder_conformance/bpsk31"
RATE, CARRIER, BAUD = 8000, 1000.0, 31.25
# Independent literals for "cq de" from the published printable Varicode.
MESSAGE_BITS = "10111100" + "11011111100" + "100" + "10110100" + "1100"


def _bpsk() -> np.ndarray:
    bits = "00000000" + MESSAGE_BITS + "00000"
    sps = round(RATE / BAUD)
    phases = [0.0]
    for bit in bits:
        phases.append(phases[-1] + (np.pi if bit == "0" else 0.0))
    symbols = np.exp(1j * np.asarray(phases))
    base = np.repeat(symbols, sps).astype(np.complex128)
    width = max(4, sps // 16)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(width * 2) / (width * 2))
    for index in range(1, len(symbols)):
        if abs(symbols[index] - symbols[index - 1]) > 1e-9:
            center = index * sps
            base[center-width:center+width] = (
                (1-ramp) * base[center-width-1:center+width-1]
                + ramp * base[center-width:center+width]
            )
    t = np.arange(base.size) / RATE
    return 0.55 * np.real(base * np.exp(2j * np.pi * CARRIER * t))


def _write(name: str, samples: np.ndarray) -> str:
    path = DEST / name
    pcm = np.clip(np.rint(samples * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(RATE)
        out.writeframes(pcm.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    positive = _write("bpsk31-cq-de.wav", _bpsk())
    t = np.arange(RATE * 4) / RATE
    negative = _write("bpsk31-carrier.wav", 0.55 * np.sin(2 * np.pi * CARRIER * t))
    base = {"decoder_id": "bpsk31", "carrier_hz": CARRIER,
            "provenance": {"kind": "synthetic",
                           "description": "Independent BPSK31 phase and published Varicode mapping; synthetic, not an air capture.",
                           "generator": "tools/build_decoder_conformance_bpsk31_fixtures.py",
                           "generator_version": "1"}}
    manifest = {"schema_version": 1, "suite_id": "bpsk31-synthetic-v1",
                "description": "Pinned BPSK31 positive and constant-carrier negative controls.",
                "fixtures": [
                    {**base, "id": "bpsk31_cq_de", "audio": {"path": "bpsk31-cq-de.wav", "sha256": positive}, "expected": {"kind": "positive", "text": "cq de"}},
                    {**base, "id": "bpsk31_constant_carrier", "provenance": {**base["provenance"], "description": "Unmodulated selected-frequency carrier without BPSK symbols."}, "audio": {"path": "bpsk31-carrier.wav", "sha256": negative}, "expected": {"kind": "negative"}}]}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
