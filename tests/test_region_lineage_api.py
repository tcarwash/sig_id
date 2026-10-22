import asyncio

import httpx

from sigid.api import create_app
from sigid.synthetic import generate_bpsk_wav


def test_split_creates_unknown_children_and_keeps_parent_as_superseded(tmp_path):
    wav = tmp_path / "source.wav"
    generate_bpsk_wav(wav, "00110110" * 20)
    app = create_app(tmp_path / "data")

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            capture = (await client.post("/api/captures", files={"file": ("source.wav", wav.read_bytes(), "audio/wav")})).json()
            base = f"/api/captures/{capture['id']}/regions"
            parent = (await client.post(base, json={
                "sample_start": 100, "sample_count": 1000,
                "frequency_low_hz": 400, "frequency_high_hz": 1800,
                "mode_id": "bpsk31",
            })).json()
            response = await client.post(f"{base}/{parent['id']}/split", json={
                "expected_geometry_revision": 1,
                "children": [
                    {"sample_start": 120, "sample_count": 400, "frequency_low_hz": 500, "frequency_high_hz": 1000},
                    {"sample_start": 550, "sample_count": 500, "frequency_low_hz": 1100, "frequency_high_hz": 1700},
                ],
            })
            assert response.status_code == 201, response.text
            operation = response.json()
            assert operation["relation"] == "split"
            assert len(operation["result_region_ids"]) == 2
            regions = {region["id"]: region for region in (await client.get(base)).json()}
            assert regions[parent["id"]]["lineage_state"] == "superseded"
            children = [regions[region_id] for region_id in operation["result_region_ids"]]
            assert all(child["lineage_state"] == "active" for child in children)
            assert all(child["review_state"] == "pending" and child["mode_id"] is None for child in children)
            assert (await client.get(f"{base}/{parent['id']}/lineage")).json()[0]["result_region_ids"] == operation["result_region_ids"]
            stale = await client.post(f"{base}/{parent['id']}/split", json={
                "expected_geometry_revision": 1,
                "children": [
                    {"sample_start": 100, "sample_count": 300},
                    {"sample_start": 500, "sample_count": 300},
                ],
            })
            assert stale.status_code == 409

    asyncio.run(run())


def test_merge_links_active_regions_without_copying_conflicting_labels(tmp_path):
    wav = tmp_path / "source.wav"
    generate_bpsk_wav(wav, "00110110" * 20)
    app = create_app(tmp_path / "data")

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            capture = (await client.post("/api/captures", files={"file": ("source.wav", wav.read_bytes(), "audio/wav")})).json()
            base = f"/api/captures/{capture['id']}/regions"
            left = (await client.post(base, json={"sample_start": 100, "sample_count": 300, "mode_id": "bpsk31"})).json()
            right = (await client.post(base, json={"sample_start": 450, "sample_count": 250, "mode_id": "rtty"})).json()
            response = await client.post(f"/api/captures/{capture['id']}/regions/merge", json={
                "source_region_ids": [left["id"], right["id"]],
                "expected_geometry_revisions": {left["id"]: 1, right["id"]: 1},
            })
            assert response.status_code == 201, response.text
            operation = response.json()
            merged = next(region for region in (await client.get(base)).json() if region["id"] == operation["result_region_id"])
            assert merged["sample_start"] == 100 and merged["sample_count"] == 600
            assert merged["review_state"] == "pending" and merged["mode_id"] is None
            regions = {region["id"]: region for region in (await client.get(base)).json()}
            assert regions[left["id"]]["lineage_state"] == "superseded"
            assert regions[right["id"]]["lineage_state"] == "superseded"
            history = (await client.get(f"{base}/{operation['result_region_id']}/lineage")).json()
            assert history[0]["relation"] == "merge"
            assert history[0]["source_region_ids"] == [left["id"], right["id"]]
            stale = await client.post(f"/api/captures/{capture['id']}/regions/merge", json={
                "source_region_ids": [left["id"], right["id"]],
                "expected_geometry_revisions": {left["id"]: 1, right["id"]: 1},
            })
            assert stale.status_code == 409

    asyncio.run(run())


def test_lineage_rejects_geometry_outside_parent_and_cross_capture_merge(tmp_path):
    wav = tmp_path / "source.wav"
    generate_bpsk_wav(wav, "00110110" * 20)
    app = create_app(tmp_path / "data")

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first = (await client.post("/api/captures", files={"file": ("one.wav", wav.read_bytes(), "audio/wav")})).json()
            second = (await client.post("/api/captures", files={"file": ("two.wav", wav.read_bytes(), "audio/wav")})).json()
            first_regions = f"/api/captures/{first['id']}/regions"
            second_regions = f"/api/captures/{second['id']}/regions"
            parent = (await client.post(first_regions, json={"sample_start": 100, "sample_count": 500, "frequency_low_hz": 400, "frequency_high_hz": 1200})).json()
            other = (await client.post(second_regions, json={"sample_start": 100, "sample_count": 500})).json()
            outside = await client.post(f"{first_regions}/{parent['id']}/split", json={
                "expected_geometry_revision": 1,
                "children": [{"sample_start": 0, "sample_count": 150}, {"sample_start": 500, "sample_count": 300}],
            })
            assert outside.status_code == 422
            cross_capture = await client.post(f"/api/captures/{first['id']}/regions/merge", json={
                "source_region_ids": [parent["id"], other["id"]],
                "expected_geometry_revisions": {parent["id"]: 1, other["id"]: 1},
            })
            assert cross_capture.status_code == 422

    asyncio.run(run())
