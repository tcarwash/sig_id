from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from sigid.api import create_app
from sigid.fldigi import FldigiClient
from sigid.receive_products import ReceiveProductStore


def test_imports_and_lists_content_addressed_png(tmp_path):
    source = tmp_path / "fax.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\n" + b"test image")
    store = ReceiveProductStore(tmp_path / "products")

    product = store.import_wefax(source)

    assert product["kind"] == "wefax_image"
    assert product["source_filename"] == "fax.png"
    assert store.list() == [product]
    path, media_type = store.image_path(product["id"])
    assert path.read_bytes() == source.read_bytes()
    assert media_type == "image/png"


def test_rejects_unsafe_paths_and_unrecognized_images(tmp_path):
    store = ReceiveProductStore(tmp_path / "products")
    with pytest.raises(ValueError, match="absolute"):
        store.import_wefax("relative.png")
    source = tmp_path / "not-image.png"
    source.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="supported"):
        store.import_wefax(source)
    with pytest.raises(KeyError):
        store.image_path("../etc/passwd")


def test_fldigi_client_reads_wefax_file_and_bounds_wait():
    calls = []
    rpc = SimpleNamespace(wefax=SimpleNamespace(get_received_file=lambda timeout: calls.append(timeout) or "/tmp/fax.png"))
    client = FldigiClient(rpc_proxy=rpc)

    assert client.receive_wefax_file(timeout_seconds=2) == "/tmp/fax.png"
    assert calls == [2]
    with pytest.raises(ValueError, match="0 to 5"):
        client.receive_wefax_file(timeout_seconds=6)


def test_wefax_api_imports_and_serves_received_image(tmp_path):
    source = tmp_path / "fax.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\n" + b"api test")
    fake_fldigi = SimpleNamespace(receive_wefax_file=lambda *, timeout_seconds=0: str(source))
    app = create_app(tmp_path / "app-data", fldigi_client=fake_fldigi)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            imported = await client.post("/api/fldigi/products/wefax/poll")
            assert imported.status_code == 200, imported.text
            product = imported.json()["product"]
            image = await client.get(product["image_url"])
            assert image.status_code == 200
            assert image.headers["content-type"] == "image/png"
            assert image.content == source.read_bytes()
            listing = await client.get("/api/fldigi/products")
            assert listing.json() == [product]

    asyncio.run(exercise())
