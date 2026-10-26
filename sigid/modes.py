"""Canonical, versioned signal-mode definitions and label resolution."""
from __future__ import annotations

import re
import json
from importlib.resources import files
from typing import Any

MODE_REGISTRY_VERSION = "2"


def _entry(mode_id: str, display_name: str, family: str, method: str, protocol: str, variant: str, aliases: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "id": mode_id,
        "display_name": display_name,
        "family": family,
        "method": method,
        "protocol": protocol,
        "variant": variant,
        "aliases": aliases,
        "registry_version": MODE_REGISTRY_VERSION,
    }


_MODE_LIST = [
    _entry("bpsk31", "BPSK31", "PSK", "BPSK", "PSK31", "BPSK31", ("PSK31", "PSK-31", "BPSK-31")),
    _entry("qpsk31", "QPSK31", "PSK", "QPSK", "PSK31", "QPSK31", ("QPSK-31", "PSK31 QPSK")),
    *[_entry(f"bpsk{rate}", f"BPSK{rate}", "PSK", "BPSK", f"PSK{rate}", f"BPSK{rate}", (f"BPSK-{rate}",)) for rate in (63, 125, 250, 500, 1000)],
    _entry("bpsk63f", "BPSK63F", "PSK", "BPSK", "PSK63", "BPSK63F", ("BPSK-63F",)),
    *[_entry(f"qpsk{rate}", f"QPSK{rate}", "PSK", "QPSK", f"PSK{rate}", f"QPSK{rate}", (f"QPSK-{rate}",)) for rate in (63, 125, 250, 500)],
    *[_entry(f"8psk{rate}", f"8PSK{rate}", "PSK", "8PSK", f"PSK{rate}", f"8PSK{rate}", (f"8-PSK{rate}",)) for rate in (125, 250, 500)],
    _entry("rtty", "RTTY", "FSK", "2-FSK", "RTTY", "RTTY", ("radio teletype",)),
    *[_entry(f"thor{tones}", f"THOR{tones}", "MFSK", "MFSK", "THOR", f"THOR{tones}") for tones in (8, 16)],
    *[_entry(f"mfsk{tones}", f"MFSK{tones}", "MFSK", "MFSK", "MFSK", f"MFSK{tones}") for tones in (4, 8, 16, 32)],
    _entry("olivia-8-250", "OLIVIA-8/250", "MFSK", "MFSK", "Olivia", "OLIVIA-8/250", ("Olivia 8 250",)),
    _entry("mt63-500s", "MT63-500S", "OFDM", "OFDM", "MT63", "MT63-500S"),
    _entry("mt63-1ks", "MT63-1KS", "OFDM", "OFDM", "MT63", "MT63-1KS"),
]
MODE_REGISTRY = {item["id"]: item for item in _MODE_LIST}


def _load_collection_catalog() -> None:
    """Add broad decoder and collection labels without claiming identification support."""
    catalog = json.loads(files(__package__).joinpath("mode_catalog.json").read_text(encoding="utf-8"))
    def key(value: str) -> str:
        return _normalize(value)

    for definition in catalog:
        names = {key(value) for value in (definition["id"], definition["display_name"], *definition["aliases"])}
        existing = next(
            (item for item in _MODE_LIST
             if names & {key(value) for value in (item["id"], item["display_name"], *item["aliases"])}),
            None,
        )
        if existing is not None:
            existing.update({field: definition[field] for field in (
                "status", "detector_status", "decoder_status", "backend_name",
                "backend_selectable", "catalog_source",
                "native_backend_id", "native_application", "native_application_url",
                "backend_mode", "source_bandwidth", "source_frequencies",
                "source_modulation", "source_mode", "source_categories",
            ) if field in definition})
            continue
        _MODE_LIST.append(definition)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


_load_collection_catalog()
MODE_REGISTRY = {item["id"]: item for item in _MODE_LIST}


def mode_name_conflicts(value: str, *, additional_modes: list[dict[str, Any]] | None = None) -> bool:
    needle = _normalize(value)
    if not needle:
        return False
    for item in [*MODE_REGISTRY.values(), *(additional_modes or [])]:
        if needle in {_normalize(term) for term in (item["id"], item["display_name"], *item["aliases"])}:
            return True
    return False


def find_modes(
    query: str, *, limit: int = 20, additional_modes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Search canonical names and aliases; exact matches rank before prefix/substring."""
    if limit < 1:
        return []
    needle = _normalize(query.strip())
    if not needle:
        return []
    ranked = []
    entries = [*_MODE_LIST, *(additional_modes or [])]
    for order, item in enumerate(entries):
        terms = (item["id"], item["display_name"], *item["aliases"])
        normalized = [_normalize(term) for term in terms]
        if needle in normalized:
            rank = 0
        elif any(term.startswith(needle) for term in normalized):
            rank = 1
        elif any(needle in term for term in normalized):
            rank = 2
        else:
            continue
        ranked.append((rank, order, item))
    fields = (
        "id", "display_name", "family", "method", "protocol", "variant", "aliases",
        "registry_version", "status", "detector_status", "decoder_status", "backend_name",
        "backend_selectable", "catalog_source", "native_backend_id", "native_application",
        "native_application_url", "backend_mode", "source_bandwidth", "source_frequencies",
        "source_modulation", "source_mode", "source_categories",
    )
    return [{key: item[key] for key in fields if key in item} for _, _, item in sorted(ranked, key=lambda row: (row[0], row[1]))[:limit]]


def resolve_mode_label(
    mode_id: str | None, label: dict[str, Any], *, additional_modes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Derive taxonomy from a canonical selection, preserving valid partial labels."""
    result = {key: label.get(key) for key in ("family", "method", "protocol", "variant")}
    if mode_id is None:
        result.update(mode_id=None, registry_version=MODE_REGISTRY_VERSION)
        return result
    definition = next(
        (item for item in [*MODE_REGISTRY.values(), *(additional_modes or [])]
         if item["id"].casefold() == mode_id.casefold()),
        None,
    )
    if definition is None:
        raise ValueError(f"unknown mode_id: {mode_id}")
    for key in ("family", "method", "protocol", "variant"):
        current = result[key]
        canonical = definition[key]
        if current is not None and current.strip() and current.strip().casefold() != canonical.casefold():
            raise ValueError(f"{key}={current!r} conflicts with selected mode {definition['display_name']} ({canonical})")
        result[key] = canonical
    result.update(mode_id=definition["id"], registry_version=MODE_REGISTRY_VERSION)
    return result


def get_mode(mode_id: str) -> dict[str, Any] | None:
    item = MODE_REGISTRY.get(mode_id.casefold())
    return dict(item) if item else None
