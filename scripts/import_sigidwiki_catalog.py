#!/usr/bin/env python3
"""Refresh the reference-only signal labels from SigIDWiki's Signal category.

Every imported page is a classification label; this does not imply that a local
 detector or decoder can recognize it. Existing catalog entries and their IDs
are preserved, and matching names are enriched with source metadata.
"""
from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.sigidwiki.com/api.php"
SOURCE = "SigIDWiki Database (Category:Signal; refreshed 2026-10-24)"
CATALOG_PATH = Path(__file__).resolve().parents[1] / "sigid" / "mode_catalog.json"
USER_AGENT = "sig-id-mode-catalog/1.0 (local catalog import)"


def api(params: dict[str, str]) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("SigIDWiki API retry loop ended unexpectedly")


def pages_in_signal_category() -> list[dict]:
    pages, continuation = [], None
    while True:
        params = {"action": "query", "list": "categorymembers", "cmtitle": "Category:Signal", "cmlimit": "max", "cmtype": "page"}
        if continuation:
            params.update(continuation)
        result = api(params)
        pages.extend(result["query"]["categorymembers"])
        continuation = result.get("continue") or result.get("query-continue", {}).get("categorymembers")
        if not continuation:
            return pages


def page_templates(pageids: list[int]) -> dict[int, dict[str, str]]:
    result = api({"action": "query", "prop": "revisions", "rvprop": "content", "pageids": "|".join(map(str, pageids))})
    parsed: dict[int, dict[str, str]] = {}
    for pageid, page in result.get("query", {}).get("pages", {}).items():
        revision = (page.get("revisions") or [{}])[0]
        content = revision.get("*", "")
        match = re.search(r"\{\{\s*Signal\s*\n(.*?)\n\}\}", content, re.I | re.S)
        fields: dict[str, str] = {}
        if match:
            for line in match.group(1).splitlines():
                field = re.match(r"\s*\|\s*([^=]+?)\s*=\s*(.*?)\s*$", line)
                if field:
                    fields[field.group(1).strip().casefold()] = field.group(2).strip()
        parsed[int(pageid)] = fields
    return parsed


def clean(value: str) -> str:
    value = re.sub(r"\[\[(?:[^]|]*\|)?([^]]+)\]\]", r"\1", value)
    value = re.sub(r"\[https?://[^ ]+\s*([^]]*)\]", r"\1", value)
    value = re.sub(r"<[^>]+>", "", value)
    return re.sub(r"\s+", " ", value).strip().strip("'")


def slug(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-") or "signal"


def family_for(modulation: str, mode: str) -> str:
    value = (modulation or mode).casefold()
    if "ofdm" in value: return "OFDM"
    if "qam" in value: return "QAM"
    if "psk" in value: return "PSK"
    if any(x in value for x in ("fsk", "gfsk", "msk", "gmsk", "afsk")): return "FSK"
    if any(x in value for x in ("ask", "ook", "pam")): return "ASK/OOK"
    if "morse" in value or value.strip() == "cw": return "Morse"
    if any(x in value for x in ("am", "ssb", "usb", "lsb")): return "AM"
    if any(x in value for x in ("fm", "fmcw")): return "FM"
    if "chirp" in value: return "Chirp"
    if "analog" in value: return "Analog"
    if "digital" in value: return "Digital"
    return "Unclassified"


def catalog_entry(page: dict, fields: dict[str, str], used_ids: set[str]) -> dict:
    title = page["title"]
    mode = clean(fields.get("mode", ""))
    modulation = clean(fields.get("modulation", ""))
    method = modulation or mode or "Unspecified"
    base = "sigidwiki-" + slug(title)
    mode_id, suffix = base, 2
    while mode_id in used_ids:
        mode_id, suffix = f"{base}-{suffix}", suffix + 1
    categories = [clean(part) for part in fields.get("additional categories", "").split(",") if clean(part)]
    frequency = clean(fields.get("frequencies", ""))
    bandwidth = clean(fields.get("bandwidth", ""))
    return {
        "id": mode_id,
        "display_name": title,
        "family": family_for(modulation, mode),
        "method": method,
        "protocol": title,
        "variant": title,
        "aliases": [],
        "registry_version": "2",
        "status": "Classification label · detector and decoder unverified",
        "detector_status": "not validated for automatic identification",
        "decoder_status": "not yet validated",
        "backend_name": None,
        "backend_selectable": False,
        "catalog_source": SOURCE,
        "source_url": "https://www.sigidwiki.com/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe="()'_-"),
        "source_mode": mode,
        "source_modulation": modulation,
        "source_bandwidth": bandwidth,
        "source_frequencies": frequency,
        "source_categories": categories,
        "source_location": clean(fields.get("location", "")),
    }


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def main() -> None:
    current = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    pages = pages_in_signal_category()
    fields_by_id: dict[int, dict[str, str]] = {}
    for start in range(0, len(pages), 50):
        fields_by_id.update(page_templates([p["pageid"] for p in pages[start:start + 50]]))
        time.sleep(0.05)

    # Retain existing definitions and IDs. Enrich exact/punctuation-insensitive
    # name matches with their SigIDWiki source fields.
    existing_names: dict[str, dict] = {}
    used_ids = {item["id"] for item in current}
    for item in current:
        for value in (item["id"], item["display_name"], *item.get("aliases", [])):
            existing_names.setdefault(norm(value), item)
    added = enriched = 0
    for page in pages:
        fields = fields_by_id.get(page["pageid"], {})
        item = catalog_entry(page, fields, used_ids)
        match = existing_names.get(norm(page["title"]))
        if match:
            for key in ("source_url", "source_mode", "source_modulation", "source_bandwidth", "source_frequencies", "source_categories", "source_location"):
                if item[key]:
                    match[key] = item[key]
            match.setdefault("catalog_source", SOURCE)
            enriched += 1
            continue
        current.append(item)
        used_ids.add(item["id"])
        existing_names[norm(item["display_name"])] = item
        added += 1

    CATALOG_PATH.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"SigIDWiki pages: {len(pages)}; metadata records: {len(fields_by_id)}; added labels: {added}; enriched existing labels: {enriched}; total catalog entries: {len(current)}")


if __name__ == "__main__":
    main()
