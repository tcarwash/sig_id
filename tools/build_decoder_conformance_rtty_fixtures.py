"""Build pinned RTTY vectors from the ITU-T S.1 ITA2 letter assignments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "tests/fixtures/decoder_conformance/rtty"
RATE = 12_000
BAUD = 45.45
CARRIER = 1_500.0
SHIFT = 170.0
LETTERS = {"A": 3, "C": 14, "D": 9, "E": 1, "G": 26, "I": 6,
           "J": 11, "L": 18, "N": 12, "O": 24, "Q": 23, "R": 10,
           "S": 5, "T": 16, "U": 7, "W": 19, "X": 29}
LTRS, SPACE = 31, 4


def _wave(text: str, *, baud: float = BAUD, shift: float = SHIFT,
          stop_bits: float = 1.5) -> np.ndarray:
    codes = [LTRS]
    for char in text:
        codes.append(SPACE if char == " " else LETTERS[char])
    durations = [(1, 4 / baud)]
    for code in codes:
        durations.append((0, 1 / baud))
        durations.extend((((code >> bit) & 1, 1 / baud) for bit in range(5)))
        durations.append((1, stop_bits / baud))
    durations.append((1, 3 / baud))
    data = np.concatenate([np.full(round(duration * RATE), bit, dtype=float)
                           for bit, duration in durations])
    freqs = np.where(data > 0.5, CARRIER + shift / 2, CARRIER - shift / 2)
    phase = 2 * np.pi * np.cumsum(freqs) / RATE
    return 0.62 * np.sin(phase)


def _write(name: str, samples: np.ndarray) -> str:
    pcm = np.clip(np.rint(samples * 32767), -32768, 32767).astype("<i2")
    path = DEST / name
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(RATE)
        out.writeframes(pcm.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    positive = _write("rtty-cq-test.wav", _wave("CQ TEST"))
    t = np.arange(RATE * 5) / RATE
    negative = _write("rtty-mark-only.wav", 0.5 * np.sin(2 * np.pi * 1585 * t))
    common = {"decoder_id": "rtty", "carrier_hz": CARRIER,
              "provenance": {"kind": "synthetic",
                             "description": "Independent ITA2/45.45-baud 170-Hz-shift waveform from S.1 letter assignments; not on-air audio.",
                             "generator": "tools/build_decoder_conformance_rtty_fixtures.py",
                             "generator_version": "1"}}
    manifest = {"schema_version": 1, "suite_id": "rtty-ita2-synthetic-v1",
                "description": "Pinned RTTY positive and mark-only abstention controls; synthetic only.",
                "fixtures": [
                    {**common, "id": "rtty_cq_test",
                     "audio": {"path": "rtty-cq-test.wav", "sha256": positive},
                     "expected": {"kind": "positive", "text": "CQ TEST"}},
                    {**common, "id": "rtty_mark_only",
                     "provenance": {**common["provenance"], "description": "Continuous RTTY mark tone without data transitions."},
                     "audio": {"path": "rtty-mark-only.wav", "sha256": negative},
                     "expected": {"kind": "negative"}}]}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # ARRL lists 57-baud Baudot as an alternate speed and 1.4 stop units as
    # a permitted stop length. Keep it in a separately injected suite because
    # each fixture suite uses one fixed decoder configuration.
    baud57_common = {**common, "provenance": {
        **common["provenance"],
        "description": "Independent ITA2/57-baud/170-Hz-shift/1.4-stop waveform; not on-air audio.",
    }}
    pos57 = _write("rtty-57baud-cq-test.wav",
                   _wave("CQ TEST", baud=57.0, shift=170.0, stop_bits=1.4))
    neg57 = _write("rtty-57baud-mark-only.wav",
                   0.5 * np.sin(2 * np.pi * 1585 * t))
    suite57 = {"schema_version": 1, "suite_id": "rtty-ita2-57baud-synthetic-v1",
               "description": "Pinned 57-baud Baudot positive and mark-only abstention controls; synthetic only.",
               "fixtures": [
                   {**baud57_common, "id": "rtty_57baud_cq_test",
                    "audio": {"path": "rtty-57baud-cq-test.wav", "sha256": pos57},
                    "expected": {"kind": "positive", "text": "CQ TEST"}},
                   {**baud57_common, "id": "rtty_57baud_mark_only",
                    "provenance": {**baud57_common["provenance"],
                                   "description": "Continuous mark tone without data transitions; 57-baud suite."},
                    "audio": {"path": "rtty-57baud-mark-only.wav", "sha256": neg57},
                    "expected": {"kind": "negative"}},
               ]}
    (DEST / "manifest-57baud.json").write_text(json.dumps(suite57, indent=2) + "\n")


if __name__ == "__main__":
    main()
