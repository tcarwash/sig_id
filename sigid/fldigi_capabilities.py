"""Build an evidence-based capability registry from a Fldigi catalog audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REFERENCE_BACKEND = "fldigi"
REFERENCE_VERSION = "4.2.13"
NON_MODE_ENTRIES = {"RSID_EOT"}
KNOWN_AUDIO_ROUND_TRIPS = {
    "BPSK31", "BPSK63", "BPSK125", "BPSK250", "BPSK500", "BPSK1000", "RTTY", "THOR8", "THOR16", "MFSK16", "MFSK32",
    "MT63-500S", "MT63-1KS", "QPSK31", "QPSK63", "QPSK125", "QPSK250",
    "QPSK500", "8PSK125", "8PSK250", "8PSK500",
}
KNOWN_UNAVAILABLE = {
    "OFDM500F": "The reference backend lists this name but returns to BPSK31 when selection is requested.",
    "OFDM750F": "The reference backend lists this name but returns to BPSK31 when selection is requested.",
    "OFDM3500": "The reference backend lists this name but returns to BPSK31 when selection is requested.",
}
DECODE_LIMITATIONS = {
    "OLIVIA-8/500": "A reference workbench loopback attempt returned only the trailing text 'K COMPLETE 123'; full-message decode remains unresolved.",
}


def build_registry(audit: dict[str, Any], audit_bytes: bytes) -> dict[str, Any]:
    """Return one capability record per audited catalog name; claims require evidence."""
    if audit.get("backend") != REFERENCE_BACKEND or audit.get("version") != REFERENCE_VERSION:
        raise ValueError(f"expected {REFERENCE_BACKEND} {REFERENCE_VERSION} selection audit")
    rows = audit.get("results")
    failures = audit.get("selection_failures")
    if not isinstance(rows, list) or not isinstance(failures, list):
        raise ValueError("selection audit must contain results and selection_failures lists")
    failed_by_mode = {row["mode"]: row for row in failures}
    seen: set[str] = set()
    modes: list[dict[str, Any]] = []
    for row in rows:
        name = row.get("mode")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"invalid or duplicate catalog mode: {name!r}")
        seen.add(name)
        failure = failed_by_mode.get(name)
        selected = bool(row.get("selectable")) and failure is None
        round_trip_tests = (
            (
                f"tests/test_fldigi_audio_integration.py::test_fldigi_payload_reaches_workbench_api_through_virtual_audio[{name}]",
                f"tests/test_fldigi_audio_integration.py::test_isolated_region_replay_decodes_a_known_fldigi_transmission[{name}]",
            )
            if name in KNOWN_AUDIO_ROUND_TRIPS else ()
        )
        if name in NON_MODE_ENTRIES:
            state = "not_a_receive_mode"
            issue = "Catalog marker, not a selectable receive modem."
        elif failure:
            state = "selection_unavailable"
            issue = KNOWN_UNAVAILABLE.get(name, str(failure.get("reason", "Selection failed in the reference audit.")))
        elif round_trip_tests:
            state = "text_decode_validated"
            issue = None
        else:
            state = "selection_validated_decode_unverified"
            issue = DECODE_LIMITATIONS.get(
                name,
                f"No known-message end-to-end decode evidence has been recorded for {name} on Fldigi {REFERENCE_VERSION}.",
            )
        modes.append({
            "backend_name": name,
            "catalog_present": True,
            "selectable": selected,
            "selection_latency_ms": row.get("selection_latency_ms"),
            "decode_validated": bool(round_trip_tests),
            "audio_routed": bool(round_trip_tests),
            "output_handled": bool(round_trip_tests),
            "generation_validated": False,
            "automatic_id_validated": False,
            "validation_state": state,
            "evidence_tests": list(round_trip_tests),
            "unresolved_issue": issue,
        })
    unrepresented_failures = set(failed_by_mode) - seen
    if unrepresented_failures:
        raise ValueError(f"selection failures absent from catalog results: {sorted(unrepresented_failures)}")
    if len(modes) != int(audit.get("catalog_count", -1)):
        raise ValueError("catalog_count does not match the result rows")
    return {
        "schema_version": 1,
        "backend": REFERENCE_BACKEND,
        "version": REFERENCE_VERSION,
        "audit_tested_at_utc": audit.get("tested_at_utc"),
        "source_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "catalog_count": len(modes),
        "selectable_count": sum(mode["selectable"] for mode in modes),
        "decode_validated_count": sum(mode["decode_validated"] for mode in modes),
        "modes": modes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path, help="Fldigi catalog-selection JSON report")
    parser.add_argument("output", type=Path, help="output capability registry JSON")
    args = parser.parse_args()
    raw = args.audit.read_bytes()
    registry = build_registry(json.loads(raw), raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(registry['modes'])} modes ({registry['selectable_count']} selectable, {registry['decode_validated_count']} decode validated)")


if __name__ == "__main__":
    main()
