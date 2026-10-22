import asyncio

import httpx

from sigid.api import create_app
from sigid.synthetic import generate_bpsk_wav


def test_audio_endpoint_honors_single_byte_ranges_for_media_seeking(tmp_path):
    wav = tmp_path / "source.wav"
    generate_bpsk_wav(wav, "00110110" * 20)
    app = create_app(tmp_path / "data")

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            capture = (await client.post("/api/captures", files={"file": ("source.wav", wav.read_bytes(), "audio/wav")})).json()
            response = await client.get(f"/api/captures/{capture['id']}/audio", headers={"Range": "bytes=44-107"})
            assert response.status_code == 206
            assert response.headers["accept-ranges"] == "bytes"
            assert response.headers["content-range"] == f"bytes 44-107/{wav.stat().st_size}"
            assert len(response.content) == 64
            assert response.content == wav.read_bytes()[44:108]
            suffix = await client.get(f"/api/captures/{capture['id']}/audio", headers={"Range": "bytes=-10"})
            assert suffix.status_code == 206
            assert suffix.headers["content-range"] == f"bytes {wav.stat().st_size - 10}-{wav.stat().st_size - 1}/{wav.stat().st_size}"
            assert suffix.content == wav.read_bytes()[-10:]
            unsatisfiable = await client.get(
                f"/api/captures/{capture['id']}/audio",
                headers={"Range": f"bytes={wav.stat().st_size}-"},
            )
            assert unsatisfiable.status_code == 416
            assert unsatisfiable.headers["content-range"] == f"bytes */{wav.stat().st_size}"

    asyncio.run(run())
