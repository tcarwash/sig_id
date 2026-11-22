"""Reproducible, standalone, receive-only command line for in-project decoders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import wave

import numpy as np

from . import DecodeWindow, create_library


def main(argv: list[str] | None = None) -> int:
    library = create_library()
    parser = argparse.ArgumentParser(
        description="Decode a bounded PCM WAV region with Signal ID's in-project library.",
    )
    parser.add_argument("wav", type=Path)
    parser.add_argument("--mode", choices=[m.id for m in library.modes()], default="cw")
    parser.add_argument("--carrier-hz", type=float, required=True,
                        help="Audio-frequency center of the selected narrowband signal")
    parser.add_argument("--start-seconds", type=float, default=0)
    parser.add_argument("--duration-seconds", type=float, default=30)
    arguments = parser.parse_args(argv)
    if not (0 <= arguments.start_seconds and 0 < arguments.duration_seconds <= 120):
        parser.error("start must be >= 0 and duration must be 0–120 seconds")
    try:
        with wave.open(str(arguments.wav), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
                parser.error("input must be uncompressed mono 16-bit PCM WAV")
            rate = wav.getframerate()
            start = round(arguments.start_seconds * rate)
            if start >= wav.getnframes():
                parser.error("selected region starts past the end of the file")
            count = min(round(arguments.duration_seconds * rate), wav.getnframes() - start)
            wav.setpos(start)
            audio = np.frombuffer(wav.readframes(count), dtype="<i2").astype(np.float64) / 32768
    except (OSError, EOFError, wave.Error) as exc:
        parser.error(str(exc))
    try:
        decoded = library.decode(arguments.mode, DecodeWindow(
            audio, rate, start, arguments.carrier_hz,
        ))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(decoded.to_dict(), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
