import asyncio

import httpx

from sigid.api import create_app
from sigid.synthetic import generate_bpsk_wav


def test_geometry_correction_appends_revision_and_rejects_stale_bounds(tmp_path):
    wav = tmp_path / "source.wav"
    generate_bpsk_wav(wav, "00110110" * 20)
    app = create_app(tmp_path / "data")

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            capture = (await client.post("/api/captures", files={"file": ("source.wav", wav.read_bytes(), "audio/wav")})).json()
            base = f"/api/captures/{capture['id']}/regions"
            region = (await client.post(base, json={"sample_start": 100, "sample_count": 500, "frequency_low_hz": 500, "frequency_high_hz": 1000})).json()
            geometry_path = f"{base}/{region['id']}/geometry"
            revised = await client.post(geometry_path, json={
                "expected_revision": 1, "sample_start": 120, "sample_count": 420,
                "frequency_low_hz": 550, "frequency_high_hz": 950,
            })
            assert revised.status_code == 201, revised.text
            assert revised.json()["geometry_revision"] == 2
            assert revised.json()["sample_start"] == 120
            stale = await client.post(geometry_path, json={
                "expected_revision": 1, "sample_start": 140, "sample_count": 300,
            })
            assert stale.status_code == 409
            stale_feedback = await client.post(
                f"{base}/{region['id']}/feedback",
                json={"expected_revision": 1, "expected_geometry_revision": 1, "action": "correct"},
            )
            assert stale_feedback.status_code == 409
            out_of_bounds = await client.post(geometry_path, json={
                "expected_revision": 2, "sample_start": 1_000_000, "sample_count": 500,
            })
            assert out_of_bounds.status_code == 422
            current = (await client.get(base)).json()[0]
            assert current["sample_start"] == 120 and current["sample_count"] == 420
            history = (await client.get(geometry_path)).json()
            assert [item["revision"] for item in history] == [1, 2]
            assert history[0]["sample_start"] == 100 and history[1]["frequency_high_hz"] == 950

    asyncio.run(run())
