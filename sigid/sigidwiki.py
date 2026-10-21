"""Metadata-only catalog import from the Signal Identification Wiki.

This importer intentionally does not download sample audio or waterfall images.
Wiki-reported labels remain source claims until an independent review is recorded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://www.sigidwiki.com/api.php"
WIKI_ROOT = "https://www.sigidwiki.com"
DEFAULT_CATEGORIES = ("Signal", "Unidentified")
MAX_PAGE_LIMIT = 500
MAX_CATEGORY_LIMIT = 1000
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_FIELD_LINE = re.compile(r"^\|\s*([^=]+?)\s*=\s*(.*)$")
_SIGNAL_TEMPLATE = re.compile(r"\{\{\s*Signal\s*\n(.*?)\n\s*\}\}", re.IGNORECASE | re.DOTALL)


def _source_fields(wikitext: str) -> dict[str, str]:
    """Extract only the key/value fields from the page's Signal template."""
    match = _SIGNAL_TEMPLATE.search(wikitext)
    if match is None:
        return {}
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        field = _FIELD_LINE.match(line)
        if field:
            key, value = field.groups()
            fields[key.strip()] = value.strip()
    return fields


def _wiki_title_url(title: str) -> str:
    from urllib.parse import quote

    path = quote(title.replace(" ", "_"), safe="()/,:_'-")
    return f"{WIKI_ROOT}/wiki/{path}"


class SigidWikiCatalogImporter:
    """Fetch a bounded set of page metadata through the public MediaWiki API."""

    def __init__(
        self,
        *,
        fetch_json: Callable[[str, dict[str, str]], dict[str, Any]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        request_interval: float = 0.2,
    ) -> None:
        if request_interval < 0:
            raise ValueError("request_interval must be non-negative")
        self._fetch_json = fetch_json or self._http_json
        self._sleep = sleep
        self.request_interval = request_interval
        self._last_request: float | None = None

    @staticmethod
    def _http_json(endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        query = urlencode(params)
        request = Request(
            f"{endpoint}?{query}",
            headers={"User-Agent": "sig-id-workbench/0.1 (metadata catalog; contact via project README)", "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=20) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("Signal Identification Wiki API response exceeds the 8 MiB limit")
            payload = json.loads(raw)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ValueError(f"Signal Identification Wiki API request failed: {exc}") from exc
        if not isinstance(payload, dict) or "error" in payload:
            raise ValueError("Signal Identification Wiki API returned an invalid response")
        return payload

    def _request(self, params: dict[str, str]) -> dict[str, Any]:
        if self._last_request is not None:
            wait = self.request_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        result = self._fetch_json(API_URL, params)
        self._last_request = time.monotonic()
        return result

    def _category_titles(self, category: str, limit: int) -> list[str]:
        titles: list[str] = []
        continuation: str | None = None
        while len(titles) < limit:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmlimit": str(min(100, limit - len(titles))),
                "format": "json",
            }
            if continuation:
                params["cmcontinue"] = continuation
            payload = self._request(params)
            members = payload.get("query", {}).get("categorymembers", [])
            titles.extend(
                member["title"] for member in members
                if isinstance(member, dict) and isinstance(member.get("title"), str)
            )
            continuation = payload.get("continue", {}).get("cmcontinue")
            if not continuation or not members:
                break
        return titles[:limit]

    def _page_records(self, titles: list[str], category: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for first in range(0, len(titles), 10):
            batch = titles[first : first + 10]
            payload = self._request({
                "action": "query",
                "titles": "|".join(batch),
                "prop": "info|revisions|categories|images",
                "rvprop": "ids|timestamp|content",
                "cllimit": "max",
                "imlimit": "max",
                "format": "json",
            })
            pages = payload.get("query", {}).get("pages", {})
            iterable = pages.values() if isinstance(pages, dict) else pages
            for page in iterable:
                revisions = page.get("revisions", [])
                if not revisions:
                    continue
                revision = revisions[0]
                wikitext = revision.get("*", "")
                fields = _source_fields(wikitext)
                images = [item.get("title") for item in page.get("images", []) if isinstance(item, dict) and item.get("title")]
                files: list[dict[str, str]] = []
                for field_name in ("Signal file", "Picture"):
                    filename = fields.get(field_name)
                    if filename:
                        title = filename if filename.startswith("File:") else f"File:{filename}"
                        files.append({"field": field_name, "title": title, "page_url": _wiki_title_url(title)})
                for title in images:
                    if title not in {file["title"] for file in files}:
                        files.append({"field": "page_image", "title": title, "page_url": _wiki_title_url(title)})
                records.append({
                    "title": page.get("title", ""),
                    "page_id": page.get("pageid"),
                    "page_url": _wiki_title_url(page.get("title", "")),
                    "source_category": category,
                    "source_revision_id": revision.get("revid"),
                    "source_revision_timestamp": revision.get("timestamp"),
                    "source_content_sha256": hashlib.sha256(wikitext.encode("utf-8")).hexdigest(),
                    "source_reported_fields": fields,
                    "source_label_status": "unreviewed_external",
                    "asset_reuse_status": "not_assessed",
                    "asset_handling": "link_only",
                    "assets": files,
                })
        return records

    def import_catalog(
        self,
        *,
        categories: tuple[str, ...] = DEFAULT_CATEGORIES,
        limit_per_category: int = 25,
    ) -> dict[str, Any]:
        if not categories or any(not category.strip() for category in categories):
            raise ValueError("at least one non-empty category is required")
        if not isinstance(limit_per_category, int) or isinstance(limit_per_category, bool) or not 1 <= limit_per_category <= MAX_CATEGORY_LIMIT:
            raise ValueError(f"limit_per_category must be between 1 and {MAX_CATEGORY_LIMIT}")
        if limit_per_category * len(categories) > MAX_PAGE_LIMIT:
            raise ValueError(f"requested catalog exceeds {MAX_PAGE_LIMIT} pages")
        records = []
        seen: set[str] = set()
        for category in categories:
            titles = self._category_titles(category, limit_per_category)
            for record in self._page_records(titles, category):
                if record["title"] not in seen:
                    seen.add(record["title"])
                    records.append(record)
        return {
            "schema_version": 1,
            "catalog": "sigidwiki-metadata-only",
            "source_api": API_URL,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "categories": list(categories),
            "limit_per_category": limit_per_category,
            "asset_policy": "No audio or images downloaded; retain source links until per-asset reuse rights are reviewed.",
            "label_policy": "Wiki fields are source-reported and unreviewed; they are not accepted dataset ground truth.",
            "records": records,
        }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
        temporary = Path(file.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Signal Identification Wiki page metadata without copying media")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-category", type=int, default=25)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    args = parser.parse_args()
    try:
        catalog = SigidWikiCatalogImporter().import_catalog(
            categories=tuple(args.categories),
            limit_per_category=args.limit_per_category,
        )
        _write_json(args.output, catalog)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(args.output), "records": len(catalog["records"]), "categories": catalog["categories"]}, indent=2))


if __name__ == "__main__":
    main()
