"""Locally cache and clean selected SigIDWiki waterfall reference images.

The source screenshot remains alongside each derived render for provenance.
These assets stay in the user's data directory and are not packaged with the app.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from .sigidwiki import API_URL, WIKI_ROOT


# Curated examples that show a useful signal pattern rather than a collage or UI.
# File titles are chosen deliberately; we do not use every image on a wiki page.
CURATED_IMAGES: dict[str, dict[str, str]] = {
    "ft8": {"page": "FT8", "file": "FT8.png"},
    "wspr": {"page": "WSPR", "file": "WSPR 1.png"},
    "ais": {"page": "Automatic Identification System (AIS)", "file": "AIS Waterfall.jpg"},
    "fldigi-olivia-16-500": {"page": "Olivia", "file": "16-500.png"},
    "rtty": {"page": "Radio Teletype (RTTY)", "file": "RTTY 170Hz 45.45bd.jpg"},
    "pocsag": {"page": "POCSAG", "file": "POCSAG Waterfallthmb.png"},
    "noaa-apt": {"page": "Automatic Picture Transmission (APT)", "file": "NOAAAPT Waterfall.png"},
}

MAX_IMAGE_BYTES = 8 * 1024 * 1024
USER_AGENT = "sig-id-workbench/0.1 (local signal field guide)"


def _api_json(params: dict[str, str]) -> dict[str, Any]:
    url = f"{API_URL}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=25) as response:
        payload = json.loads(response.read(2 * 1024 * 1024 + 1))
    if not isinstance(payload, dict) or "error" in payload:
        raise ValueError("SigIDWiki returned an invalid MediaWiki API response")
    return payload


def _source_url(page: str, filename: str) -> tuple[str, str]:
    title = f"File:{filename}"
    payload = _api_json({
        "action": "query", "titles": title, "prop": "imageinfo",
        "iiprop": "url|size", "format": "json",
    })
    pages = payload.get("query", {}).get("pages", {})
    record = next(iter(pages.values()), {}) if isinstance(pages, dict) else {}
    info = (record.get("imageinfo") or [{}])[0]
    url = info.get("url")
    if not isinstance(url, str) or not url.startswith(f"{WIKI_ROOT}/images/"):
        raise ValueError(f"Could not resolve the SigIDWiki image for {filename}")
    if int(info.get("size", MAX_IMAGE_BYTES + 1)) > MAX_IMAGE_BYTES:
        raise ValueError(f"SigIDWiki image {filename} exceeds the 8 MiB limit")
    page_slug = quote(page.replace(" ", "_"), safe="()/,:_'-")
    file_slug = quote(filename.replace(" ", "_"), safe="()_'-")
    page_url = f"{WIKI_ROOT}/wiki/{page_slug}"
    file_page_url = f"{WIKI_ROOT}/wiki/File:{file_slug}"
    return url, page_url + "#" + file_page_url


def _clean_signal(source: Image.Image) -> Image.Image:
    """Keep bright signal pixels, discard the dark waterfall background, recolor yellow."""
    rgb_image = source.convert("RGB")
    rgb = np.asarray(rgb_image, dtype=np.float32)
    height, width = rgb.shape[:2]
    corner_size = max(2, min(height, width) // 32)
    corners = np.concatenate((
        rgb[:corner_size, :corner_size].reshape(-1, 3),
        rgb[:corner_size, -corner_size:].reshape(-1, 3),
        rgb[-corner_size:, :corner_size].reshape(-1, 3),
        rgb[-corner_size:, -corner_size:].reshape(-1, 3),
    ), axis=0)
    background = np.median(corners, axis=0)
    distance = np.sqrt(np.sum((rgb - background) ** 2, axis=2))
    # A soft alpha ramp preserves weak blue filter skirts while removing the
    # dark navy/black noise floor. Keep original pixel geometry and edge detail.
    alpha = np.clip((distance - 10.0) * 2.2, 0, 255).astype(np.uint8)
    if int(alpha.max()) < 12:
        raise ValueError("Could not separate a bright signal from the image background")
    active = alpha >= 18
    ys, xs = np.where(active)
    pad_x, pad_y = max(3, width // 80), max(3, height // 80)
    left, right = max(0, int(xs.min()) - pad_x), min(width, int(xs.max()) + pad_x + 1)
    top, bottom = max(0, int(ys.min()) - pad_y), min(height, int(ys.max()) + pad_y + 1)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = (255, 218, 92)
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, "RGBA").crop((left, top, right, bottom))


def import_wiki_images(data_dir: str | Path, entry_ids: list[str] | None = None) -> dict[str, Any]:
    """Download curated source files and write transparent signal-only PNG renders."""
    selected = entry_ids or list(CURATED_IMAGES)
    unknown = sorted(set(selected) - CURATED_IMAGES.keys())
    if unknown:
        raise ValueError(f"No curated SigIDWiki image mapping for: {', '.join(unknown)}")
    target = Path(data_dir).expanduser() / "field-guide" / "wiki-images"
    target.mkdir(parents=True, exist_ok=True)
    assets: dict[str, Any] = {}
    for entry_id in selected:
        spec = CURATED_IMAGES[entry_id]
        image_url, source_reference = _source_url(spec["page"], spec["file"])
        request = Request(image_url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=30) as response:
            raw = response.read(MAX_IMAGE_BYTES + 1)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError(f"SigIDWiki image {spec['file']} exceeds the 8 MiB limit")
        try:
            import io
            image = Image.open(io.BytesIO(raw))
            image.load()
        except Exception as exc:
            raise ValueError(f"SigIDWiki image {spec['file']} is not a supported raster image") from exc
        cleaned = _clean_signal(image)
        source_path = target / f"{entry_id}.source{image.format.lower()}"
        output_path = target / f"{entry_id}.png"
        source_path.write_bytes(raw)
        with tempfile.NamedTemporaryFile(suffix=".png", dir=target, delete=False) as temporary:
            temp_path = Path(temporary.name)
        try:
            cleaned.save(temp_path, format="PNG", optimize=True)
            temp_path.replace(output_path)
        finally:
            temp_path.unlink(missing_ok=True)
        page_slug = quote(spec["page"].replace(" ", "_"), safe="()/,:_'-")
        source_page = f"{WIKI_ROOT}/wiki/{page_slug}"
        assets[entry_id] = {
            "path": output_path.name, "source_file": spec["file"],
            "source_page": source_page,
            "source_image": image_url, "source_attribution_page": source_reference.split("#", 1)[1],
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "asset_reuse_status": "not_assessed",
            "processing": "dark-background removal; signal pixels recolored yellow; tight alpha crop",
            "width": cleaned.width, "height": cleaned.height,
        }
        print(f"[field-guide] cleaned {spec['file']} -> {output_path} ({cleaned.width}x{cleaned.height})")
    manifest_path = target / "manifest.json"
    previous = {}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8")).get("assets", {})
            previous = {key: value for key, value in previous.items() if key in CURATED_IMAGES}
        except (OSError, json.JSONDecodeError):
            pass
    previous.update(assets)
    manifest_path.write_text(json.dumps({
        "schema_version": 1, "source": "SigIDWiki", "assets": previous,
    }, indent=2) + "\n", encoding="utf-8")
    return {"directory": str(target), "assets": assets}


def read_wiki_image_manifest(data_dir: str | Path) -> dict[str, Any]:
    path = Path(data_dir).expanduser() / "field-guide" / "wiki-images" / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return manifest.get("assets", {}) if isinstance(manifest, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache selected SigIDWiki waterfall images as transparent yellow signal renders")
    parser.add_argument("--data-dir", type=Path, default=Path("~/.local/share/sig-id"))
    parser.add_argument("--entries", nargs="+", choices=sorted(CURATED_IMAGES))
    args = parser.parse_args()
    try:
        result = import_wiki_images(args.data_dir, args.entries)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"directory": result["directory"], "count": len(result["assets"])}, indent=2))


if __name__ == "__main__":
    main()
