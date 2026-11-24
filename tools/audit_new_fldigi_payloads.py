"""Read-only independent native-decoder checks of newly generated Fldigi WAVs.

Decoding is fallible. This does NOT certify source text, CW timing, or
protocol integrity and never changes source labels or acceptance state.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
import wave

import numpy as np

from sigid.dataset import CaptureStore
from sigid.libsigdecode import DecodeWindow, CWDecoder, RTTYDecoder, BPSK31Decoder


def audit_generated_payloads(data_dir: str | Path) -> dict:
    store = CaptureStore(Path(data_dir) / "dataset.sqlite3")
    decoders = {"CW": CWDecoder(), "RTTY": RTTYDecoder(), "BPSK31": BPSK31Decoder()}
    cases = []
    for capture in store.list_captures():
        meta = capture["source_metadata"]
        params = meta.get("parameters") or {}
        mode = params.get("native_mode")
        if params.get("payload_recipe_version") != "diverse-operator-messages-v1" or mode not in decoders:
            continue
        start = time.monotonic()
        try:
            with wave.open(capture["audio_path"], "rb") as source:
                sr = source.getframerate()
                samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
            result = decoders[mode].decode(DecodeWindow(
                samples, sr, carrier_hz=float(params["carrier_hz"]),
            ))
            wanted = " ".join(params["transmit_payload"].split())
            got = " ".join(result.text.split())
            status = result.status
            exact = got == wanted
            contained = bool(wanted and wanted in got)
            # Human-readable partials are still *not* integrity-checked text.
            case = {
                "waveform_sha256": capture["sha256"], "mode": mode,
                "requested_payload": wanted, "native_decoder_status": status,
                "native_decoder_text": got[:256],
                "exact_payload_match": exact,
                "payload_contained": contained,
                "native_decoding_elapsed_seconds": round(time.monotonic() - start, 3),
                "independent_text_verified": exact or contained,
                "note": "Native source-independent modem decode; no CRC on CW/RTTY/BPSK31.",
            }
        except Exception as exc:
            # One unsupported or numerical decoder condition must be
            # surfaced as failed independent validation, not abort the
            # entire source-quality report.
            case = {
                "waveform_sha256": capture["sha256"], "mode": mode,
                "requested_payload": params["transmit_payload"],
                "error": str(exc), "error_type": type(exc).__name__,
                "independent_text_verified": False,
            }
        cases.append(case)
    by_mode = {}
    for mode in decoders:
        subset = [case for case in cases if case["mode"] == mode]
        by_mode[mode] = {
            "sources": len(subset),
            "verified_exact_or_contained": sum(c.get("independent_text_verified", False) for c in subset),
            "decoder_statuses": dict(Counter(c.get("native_decoder_status", "error") for c in subset)),
        }
    return {
        "source_count": len(cases),
        "by_mode": by_mode,
        "cases": sorted(cases, key=lambda row: (row["mode"], row["waveform_sha256"])),
        "note": "Decoder validation report is supplemental: mode provenance is source manifest; text requested is not itself decoded proof.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_generated_payloads(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "source_count": result["source_count"],
        "by_mode": result["by_mode"],
        "report": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
