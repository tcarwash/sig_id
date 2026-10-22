import asyncio

import httpx

from sigid.api import create_app
from sigid.synthetic import generate_bpsk_wav


def test_scene_coverage_revisions_bind_review_to_annotation_set_and_become_stale(tmp_path):
    wav = tmp_path / "source.wav"
    generate_bpsk_wav(wav, "00110110" * 24)
    app = create_app(tmp_path / "data")

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            capture = (await client.post("/api/captures", files={"file": ("source.wav", wav.read_bytes(), "audio/wav")})).json()
            regions_path = f"/api/captures/{capture['id']}/regions"
            region = (await client.post(regions_path, json={
                "sample_start": 100, "sample_count": 600, "frequency_low_hz": 500,
                "frequency_high_hz": 1500, "mode_id": "bpsk31",
            })).json()
            coverage_path = f"/api/captures/{capture['id']}/coverage"
            created = await client.post(coverage_path, json={
                "sample_start": 0, "sample_count": 1800,
                "frequency_low_hz": 300, "frequency_high_hz": 2200,
                "task": "detection", "taxonomy_scope": ["PSK", "FSK"],
                "state": "complete",
                "ignore_regions": [{"sample_start": 1500, "sample_count": 100, "reason": "Known receiver artifact"}],
                "uncertain_regions": [], "notes": "All visible candidates reviewed.",
            })
            assert created.status_code == 201, created.text
            first = created.json()
            assert first["revision"] == 1 and first["is_stale"] is False
            assert len(first["annotation_set_digest"]) == 64
            assert first["ignore_regions"][0]["reason"] == "Known receiver artifact"

            revised_label = await client.post(f"{regions_path}/{region['id']}/labels", json={
                "expected_revision": 1, "mode_id": "rtty", "notes": "Identity correction",
            })
            assert revised_label.status_code == 201, revised_label.text
            current = (await client.get(coverage_path)).json()[0]
            assert current["is_stale"] is True
            assert current["annotation_set_digest"] == first["annotation_set_digest"]
            assert current["current_annotation_set_digest"] != first["annotation_set_digest"]

            updated = await client.post(f"{coverage_path}/{first['id']}", json={
                "expected_revision": 1, "sample_start": 0, "sample_count": 1800,
                "frequency_low_hz": 300, "frequency_high_hz": 2200,
                "task": "detection", "taxonomy_scope": ["FSK"], "state": "partial",
                "ignore_regions": [],
                "uncertain_regions": [{"sample_start": 1600, "sample_count": 100, "reason": "Weak signal at edge"}],
                "notes": "Reopened after label correction.",
            })
            assert updated.status_code == 201, updated.text
            assert updated.json()["revision"] == 2 and updated.json()["is_stale"] is False
            assert updated.json()["uncertain_regions"][0]["reason"] == "Weak signal at edge"
            stale = await client.post(f"{coverage_path}/{first['id']}", json={
                "expected_revision": 1, "sample_start": 0, "sample_count": 1800,
                "task": "detection", "taxonomy_scope": [], "state": "complete",
            })
            assert stale.status_code == 409
            history = (await client.get(f"{coverage_path}/{first['id']}/history")).json()
            assert [item["revision"] for item in history] == [1, 2]

    asyncio.run(run())


def test_scene_coverage_rejects_out_of_window_masks_and_capture_bounds(tmp_path):
    wav = tmp_path / "source.wav"
    generate_bpsk_wav(wav, "00110110" * 24)
    app = create_app(tmp_path / "data")

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            capture = (await client.post("/api/captures", files={"file": ("source.wav", wav.read_bytes(), "audio/wav")})).json()
            base = f"/api/captures/{capture['id']}/coverage"
            outside = await client.post(base, json={
                "sample_start": 0, "sample_count": 1000, "task": "detection", "state": "complete",
                "taxonomy_scope": [],
                "ignore_regions": [{"sample_start": 900, "sample_count": 200, "reason": "outside"}],
            })
            assert outside.status_code == 422
            beyond_capture = await client.post(base, json={
                "sample_start": capture["sample_count"] - 1, "sample_count": 100,
                "task": "detection", "state": "complete", "taxonomy_scope": [],
            })
            assert beyond_capture.status_code == 422

    asyncio.run(run())
