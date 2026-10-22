import asyncio

import httpx

from sigid.api import create_app
from sigid.synthetic import generate_2fsk_wav, generate_bpsk_wav


def test_custom_mode_requires_independent_audio_evidence_and_only_enters_search_after_review(tmp_path):
    first_path = tmp_path / "first.wav"
    second_path = tmp_path / "second.wav"
    generate_bpsk_wav(first_path, "00110110" * 24)
    generate_2fsk_wav(second_path, "01101001" * 24, sample_rate=8000, symbol_rate=100)
    app = create_app(tmp_path / "data")

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            sources = []
            for audio_path in (first_path, second_path):
                capture = (await client.post("/api/captures", files={"file": (audio_path.name, audio_path.read_bytes(), "audio/wav")})).json()
                region = (await client.post(f"/api/captures/{capture['id']}/regions", json={"sample_start": 0, "sample_count": min(2000, capture["sample_count"])})).json()
                sources.append({"capture_id": capture["id"], "region_id": region["id"], "observations": "Stable repeated symbol timing and a distinct tone transition pattern are audible in this independent recording."})
            proposal = {
                "id": "experimental-multitone-1",
                "display_name": "Experimental Multitone X1",
                "family": "MFSK",
                "method": "multi-tone FSK",
                "protocol": "Experimental Multitone",
                "variant": "X1",
                "aliases": ["EMX1", "multitone x1"],
                "rationale": "This proposal identifies a repeatable experimental digital mode heard in multiple independent recordings and characterized using the attached exact sample regions.",
                "distinguishing_evidence": "The observed tone spacing and transition cadence differ from the nearest existing MFSK modes; both recordings show the same measured structure under different capture conditions.",
                "evidence_regions": sources,
            }
            created = await client.post("/api/mode-proposals", json=proposal)
            assert created.status_code == 201, created.text
            assert created.json()["status"] == "pending"
            assert not (await client.get("/api/modes", params={"q": "EMX1"})).json()["modes"]
            label = await client.post(f"/api/captures/{sources[0]['capture_id']}/regions", json={
                "sample_start": 200, "sample_count": 500, "mode_id": proposal["id"],
            })
            assert label.status_code == 422
            decision = await client.post(f"/api/mode-proposals/{proposal['id']}/decision", json={
                "decision": "accepted", "note": "Compared two independent recordings and confirmed the distinguishing tone pattern against the documented reference."}
            )
            assert decision.status_code == 201, decision.text
            matches = (await client.get("/api/modes", params={"q": "EMX1"})).json()["modes"]
            assert matches[0]["id"] == proposal["id"]
            assert matches[0]["status"] == "locally_accepted"
            label = await client.post(f"/api/captures/{sources[0]['capture_id']}/regions", json={
                "sample_start": 200, "sample_count": 500, "mode_id": proposal["id"],
            })
            assert label.status_code == 201, label.text
            assert label.json()["variant"] == "X1"

    asyncio.run(run())


def test_mode_proposal_rejects_duplicate_source_audio_and_inadequate_evidence(tmp_path):
    audio = tmp_path / "source.wav"
    generate_bpsk_wav(audio, "00110110" * 24)
    app = create_app(tmp_path / "data")

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            capture = (await client.post("/api/captures", files={"file": ("source.wav", audio.read_bytes(), "audio/wav")})).json()
            region = (await client.post(f"/api/captures/{capture['id']}/regions", json={"sample_start": 0, "sample_count": 2000})).json()
            other_region = (await client.post(f"/api/captures/{capture['id']}/regions", json={"sample_start": 2500, "sample_count": 2000})).json()
            evidence = [
                {"capture_id": capture["id"], "region_id": region["id"], "observations": "A stable signal structure is audible in this source region and was reviewed."},
                {"capture_id": capture["id"], "region_id": other_region["id"], "observations": "The same mode structure can be heard again later in this same recording."},
            ]
            payload = {
                "id": "proposed-mode-a", "display_name": "Proposed Mode A", "family": "FSK",
                "method": "2-FSK", "protocol": "Mode A", "variant": "A1", "aliases": ["MA1"],
                "rationale": "A repeatable digital signal with a distinctive observed transition pattern.",
                "distinguishing_evidence": "The tone separation and symbol timing differ from the other local mode definitions.",
                "evidence_regions": evidence,
            }
            response = await client.post("/api/mode-proposals", json=payload)
            assert response.status_code == 422

    asyncio.run(run())
