"""Validated, durable storage for non-text products received by decoder backends."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


MAX_PRODUCT_BYTES = 32 * 1024 * 1024
IMAGE_FORMATS = (
    (b"\x89PNG\r\n\x1a\n", ".png", "image/png"),
    (b"\xff\xd8\xff", ".jpg", "image/jpeg"),
    (b"BM", ".bmp", "image/bmp"),
    (b"II*\x00", ".tif", "image/tiff"),
    (b"MM\x00*", ".tif", "image/tiff"),
)


class ReceiveProductStore:
    """Copy bounded image outputs into the app's managed directory by content hash."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def import_wefax(self, source_path: str | Path) -> dict[str, Any]:
        source = Path(source_path)
        if not source.is_absolute():
            raise ValueError("Fldigi returned a non-absolute WEFAX path")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(source, flags)
        except OSError as exc:
            raise ValueError(f"cannot safely open received WEFAX file: {exc}") from exc
        with os.fdopen(descriptor, "rb") as handle:
            details = os.fstat(handle.fileno())
            if not stat.S_ISREG(details.st_mode):
                raise ValueError("received WEFAX output is not a regular file")
            if details.st_size <= 0 or details.st_size > MAX_PRODUCT_BYTES:
                raise ValueError(f"received image must be between 1 and {MAX_PRODUCT_BYTES} bytes")
            payload = handle.read(MAX_PRODUCT_BYTES + 1)
        if len(payload) != details.st_size or len(payload) > MAX_PRODUCT_BYTES:
            raise ValueError("received image changed or exceeded its size limit while reading")
        image = next((item for item in IMAGE_FORMATS if payload.startswith(item[0])), None)
        if image is None:
            raise ValueError("received WEFAX product is not a supported PNG, JPEG, BMP or TIFF image")
        _, extension, media_type = image
        digest = hashlib.sha256(payload).hexdigest()
        product_id = f"{digest[:24]}"
        image_path = self.directory / f"{product_id}{extension}"
        metadata_path = self.directory / f"{product_id}.json"
        if not image_path.exists():
            temporary = self.directory / f".{product_id}.tmp"
            with temporary.open("xb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, image_path)
        metadata = {
            "id": product_id,
            "kind": "wefax_image",
            "media_type": media_type,
            "size_bytes": len(payload),
            "sha256": digest,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "source_backend": "fldigi",
            "source_filename": source.name,
            "image_url": f"/api/fldigi/products/{product_id}/image",
        }
        if not metadata_path.exists():
            temporary_metadata = self.directory / f".{product_id}.json.tmp"
            temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary_metadata, metadata_path)
        return self._read_metadata(metadata_path)

    def list(self) -> list[dict[str, Any]]:
        products = []
        for path in self.directory.glob("[0-9a-f]*.json"):
            try:
                products.append(self._read_metadata(path))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return sorted(products, key=lambda item: str(item.get("received_at", "")), reverse=True)

    def image_path(self, product_id: str) -> tuple[Path, str]:
        if len(product_id) != 24 or any(character not in "0123456789abcdef" for character in product_id):
            raise KeyError(product_id)
        metadata_path = self.directory / f"{product_id}.json"
        metadata = self._read_metadata(metadata_path)
        extension = next((ext for _, ext, mime in IMAGE_FORMATS if mime == metadata["media_type"]), None)
        if extension is None:
            raise KeyError(product_id)
        path = self.directory / f"{product_id}{extension}"
        if not path.is_file() or path.stat().st_size != metadata["size_bytes"]:
            raise KeyError(product_id)
        return path, metadata["media_type"]

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("kind") != "wefax_image":
            raise ValueError("invalid receive-product metadata")
        return data
