"""Keep delegation documentation synchronized with the implemented contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sigid.libsigdecode import DecodeWindow, create_library

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_mode_queue_is_unique_and_every_reference_is_documented():
    manifest = json.loads((DOCS / "decoder-work-items.json").read_text())
    assert manifest["schema_version"] == 1
    assert len(manifest["work_items"]) >= 30
    assert (ROOT / manifest["api_contract"]).is_file()
    assert (ROOT / manifest["agent_playbook"]).is_file()
    assert (ROOT / manifest["evidence_index"]).is_file()
    source_ids = set(manifest["references"])
    index = (DOCS / "DECODER_EVIDENCE_INDEX.md").read_text()
    names = [item["id"] for item in manifest["work_items"]]
    assert len(names) == len(set(names))
    stages = {"planned", "experimental_implemented", "benchmarked", "validated"}
    for item in manifest["work_items"]:
        assert item["state"] in stages
        # Missing primary references are legitimate research blockers,
        # not a reason to invent a URL or copy another decoder's code.
        if not item["reference_ids"]:
            assert item["state"] == "planned"
        for key in item["reference_ids"]:
            assert key in source_ids
            assert key in index
        assert item["input_kind"] in {"mono-audio", "iq-proposed"}
        assert item["evidence_file"].startswith("docs/decoder-evidence/")
        assert item["next_step"]
    deps = set(names) | set(manifest["infrastructure_dependencies"])
    for item in manifest["work_items"]:
        assert set(item["depends_on"]) <= deps, item["id"]
        if item["state"] != "planned":
            assert (ROOT / item["evidence_file"]).is_file()
    implemented = {item["id"] for item in manifest["work_items"]
                   if item["state"] != "planned"}
    assert implemented == {mode.id for mode in create_library().modes()}


def test_reference_urls_are_absolute_and_not_source_code_imports():
    manifest = json.loads((DOCS / "decoder-work-items.json").read_text())
    for ref in manifest["references"].values():
        assert ref["url"].startswith("https://")
        assert ref["title"] and ref["kind"]
    assert "Do NOT import/copy/translate" in (DOCS / "DECODER_AGENT_PLAYBOOK.md").read_text()


def test_local_decoder_schema_matches_implemented_serialization():
    schema = json.loads((DOCS / "schemas/local_decode_result.schema.json").read_text())
    example = create_library().decode(
        "cw", DecodeWindow(np.zeros(12000), 12000, carrier_hz=1100),
    ).to_dict()
    assert set(schema["required"]) == set(example)
    assert schema["properties"]["status"]["enum"] == ["decoded", "partial", "no_decode"]
    assert schema["properties"]["source"]["const"] == example["source"]
    assert set(schema["properties"]["tokens"]["items"]["required"]) == {
        "text", "start_sample", "end_sample", "pattern", "confidence_hint",
    }


def test_agent_entry_documents_link_back_to_each_other():
    key_docs = [
        "DECODER_API_SPEC.md", "DECODER_AGENT_PLAYBOOK.md",
        "DECODER_EVIDENCE_INDEX.md", "LIBSIGDECODE.md",
    ]
    for name in key_docs[:3]:
        text = (DOCS / name).read_text()
        for referenced in ["decoder-work-items.json", "DECODER_API_SPEC.md",
                           "DECODER_AGENT_PLAYBOOK.md", "DECODER_EVIDENCE_INDEX.md"]:
            if referenced == name:
                continue
            assert referenced in text, (name, referenced)
