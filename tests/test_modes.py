import pytest

from sigid.modes import MODE_REGISTRY_VERSION, find_modes, resolve_mode_label


def test_psk31_search_returns_bpsk_and_qpsk_as_distinct_modes():
    matches = find_modes("PSK31")
    assert [item["id"] for item in matches[:2]] == ["bpsk31", "qpsk31"]
    assert matches[0]["family"] == "PSK"
    assert matches[0]["method"] == "BPSK"
    assert matches[0]["protocol"] == "PSK31"
    assert matches[0]["variant"] == "BPSK31"


def test_resolver_derives_taxonomy_and_rejects_contradictory_claims():
    result = resolve_mode_label("bpsk31", {"family": None, "method": None, "protocol": None, "variant": None})
    assert result == {"mode_id": "bpsk31", "family": "PSK", "method": "BPSK", "protocol": "PSK31", "variant": "BPSK31", "registry_version": MODE_REGISTRY_VERSION}
    with pytest.raises(ValueError, match="conflicts"):
        resolve_mode_label("bpsk31", {"family": "FSK", "method": None, "protocol": None, "variant": None})


def test_unknown_and_partial_labels_are_preserved_without_guessing():
    partial = {"family": "FSK", "method": None, "protocol": None, "variant": None}
    assert resolve_mode_label(None, partial)["family"] == "FSK"
    assert resolve_mode_label(None, partial)["mode_id"] is None


def test_alias_resolution_is_case_and_separator_insensitive():
    assert find_modes("bpsk-31")[0]["id"] == "bpsk31"
    assert find_modes("unknown protocol") == []
