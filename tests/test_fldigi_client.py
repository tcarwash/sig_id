from __future__ import annotations

from types import SimpleNamespace

import pytest

from sigid.fldigi import FldigiClient, ModeCapability
from sigid.fldigi_probe import probe


@pytest.fixture
def client(monkeypatch):
    state = {"mode": "PSK31", "text": b"HELLO\xff"}
    rpc = SimpleNamespace(
        fldigi=SimpleNamespace(name=lambda: "fldigi", version=lambda: "test-1.0"),
        modem=SimpleNamespace(
            get_names=lambda: ["PSK31", "RTTY", "THOR16"],
            get_name=lambda: state["mode"],
            set_by_name=lambda name: state.update(mode=name) or "old",
        ),
        text=SimpleNamespace(
            get_rx_length=lambda: len(state["text"]),
            get_rx=lambda start, length: state["text"][start : start + length],
        ),
    )
    return FldigiClient(rpc_proxy=rpc), state


def test_client_lists_selects_and_reads_arbitrary_bytes(client):
    client, state = client
    assert client.modem_names() == ["PSK31", "RTTY", "THOR16"]
    assert client.select_modem("THOR16") == "THOR16"
    assert state["mode"] == "THOR16"
    assert client.receive_text(start=0) == b"HELLO\xff"
    assert client.receive_text_length() == 6


def test_client_rejects_unknown_modes_and_invalid_offsets(client):
    client, _ = client
    with pytest.raises(ValueError, match="not present"):
        client.select_modem("invented-mode")
    with pytest.raises(ValueError, match="non-negative"):
        client.receive_text(start=-1)


def test_probe_marks_catalog_presence_as_pending_validation(client, monkeypatch):
    client, _ = client
    monkeypatch.setattr("sigid.fldigi_probe.FldigiClient", lambda _: client)
    result = probe("unused")
    assert result["catalog_count"] == 3
    thor = next(mode for mode in result["modes"] if mode["backend_name"] == "THOR16")
    assert thor["selectable"] is True
    assert thor["decode_validated"] is False
    assert thor["generation_validated"] is False


def test_mode_capability_defaults_never_overclaim_support():
    mode = ModeCapability("MFSK16", selectable=True)
    assert not mode.audio_routed
    assert not mode.decode_validated
    assert not mode.output_handled
    assert not mode.generation_validated
    assert not mode.automatic_id_validated


@pytest.mark.skipif(__import__("os").environ.get("SIGID_TEST_FLDIGI") != "1", reason="set SIGID_TEST_FLDIGI=1 for installed backend")
def test_live_fldigi_backend():
    result = probe("http://127.0.0.1:7362/RPC2")
    assert result["backend"]["name"] == "fldigi"
    assert result["catalog_count"] > 20
    assert "PSK31" in [mode["backend_name"] for mode in result["modes"]]
