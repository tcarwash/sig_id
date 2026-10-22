from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigid.fldigi_capabilities import build_registry


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "reports" / "fldigi-4.2.13-catalog-selection.json"


def test_reference_registry_covers_catalog_without_overclaiming_decode_support():
    raw = AUDIT_PATH.read_bytes()
    audit = json.loads(raw)
    report = build_registry(audit, raw)
    modes = {mode["backend_name"]: mode for mode in report["modes"]}

    assert report["catalog_count"] == 169 == len(modes)
    assert report["selectable_count"] == 165
    assert report["decode_validated_count"] == 10
    assert all(mode["catalog_present"] for mode in modes.values())
    assert {name for name, mode in modes.items() if mode["decode_validated"]} == {
        "BPSK31", "BPSK63", "RTTY", "THOR8", "THOR16", "MFSK16",
        "MFSK32", "MT63-500S", "MT63-1KS", "QPSK31",
    }
    for name, mode in modes.items():
        assert mode["output_handled"] == mode["decode_validated"]
        assert mode["audio_routed"] == mode["decode_validated"]
        assert not mode["generation_validated"]
        assert not mode["automatic_id_validated"]
        if mode["selectable"] and not mode["decode_validated"]:
            assert mode["unresolved_issue"]
        if mode["decode_validated"]:
            assert mode["validation_state"] == "text_decode_validated"
            assert mode["unresolved_issue"] is None
    source = (ROOT / "tests" / "test_fldigi_audio_integration.py").read_text()
    assert "def test_fldigi_payload_reaches_workbench_api_through_virtual_audio" in source
    assert "def test_isolated_region_replay_decodes_a_known_fldigi_transmission" in source
    for name in ("BPSK31", "BPSK63", "RTTY", "THOR8", "THOR16", "MFSK16", "MFSK32", "MT63-500S", "MT63-1KS", "QPSK31"):
        assert len(modes[name]["evidence_tests"]) == 2
    assert "trailing text" in modes["OLIVIA-8/500"]["unresolved_issue"]
    assert modes["RSID_EOT"]["validation_state"] == "not_a_receive_mode"
    for name in ("OFDM500F", "OFDM750F", "OFDM3500"):
        assert not modes[name]["selectable"]
        assert modes[name]["validation_state"] == "selection_unavailable"


def test_registry_rejects_audit_with_missing_or_duplicate_catalog_rows():
    audit = json.loads(AUDIT_PATH.read_text())
    audit["results"] = audit["results"][:-1]
    with pytest.raises(ValueError, match="absent from catalog results"):
        build_registry(audit, json.dumps(audit).encode())

    audit = json.loads(AUDIT_PATH.read_text())
    audit["results"].append(dict(audit["results"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        build_registry(audit, json.dumps(audit).encode())


def test_registry_rejects_mismatched_reference_backend():
    audit = json.loads(AUDIT_PATH.read_text())
    audit["version"] = "other"
    with pytest.raises(ValueError, match="expected fldigi 4.2.13"):
        build_registry(audit, json.dumps(audit).encode())
