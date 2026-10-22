import asyncio

import httpx

from sigid.api import create_app
from sigid.synthetic import generate_bpsk_wav


def test_operator_feedback_is_append_only_and_bound_to_exact_label_revision(tmp_path):
    wav = tmp_path / "source.wav"
    generate_bpsk_wav(wav, "00110110" * 20)
    app = create_app(tmp_path / "data")

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            capture = (await client.post("/api/captures", files={"file": ("source.wav", wav.read_bytes(), "audio/wav")})).json()
            path = f"/api/captures/{capture['id']}/regions"
            region = (await client.post(path, json={"sample_start": 0, "sample_count": 1200, "mode_id": "bpsk31", "origin": "detector_proposal"})).json()
            feedback_path = f"{path}/{region['id']}/feedback"
            correct = await client.post(feedback_path, json={"expected_revision": 1, "expected_geometry_revision": 1, "action": "correct"})
            assert correct.status_code == 201, correct.text
            assert correct.json()["action"] == "correct"
            assert correct.json()["resulting_revision"] == 2
            wrong = await client.post(feedback_path, json={"expected_revision": 2, "expected_geometry_revision": 1, "action": "incorrect_identity", "corrected_mode_id": "rtty", "note": "Observed two-tone FSK"})
            assert wrong.status_code == 201, wrong.text
            assert wrong.json()["correction"]["family"] == "FSK"
            assert wrong.json()["correction"]["variant"] == "RTTY"
            assert wrong.json()["resulting_revision"] == 3
            not_signal = await client.post(feedback_path, json={"expected_revision": 3, "expected_geometry_revision": 1, "action": "not_signal"})
            assert not_signal.status_code == 201 and not_signal.json()["resulting_revision"] == 4
            stale = await client.post(feedback_path, json={"expected_revision": 3, "expected_geometry_revision": 1, "action": "correct"})
            assert stale.status_code == 409
            history = (await client.get(feedback_path)).json()
            assert [item["action"] for item in history] == ["correct", "incorrect_identity", "not_signal"]
            assert history[1]["correction"]["mode_id"] == "rtty"
            current = (await client.get(path)).json()[0]
            assert current["review_state"] == "rejected" and current["variant"] == "RTTY" and current["revision"] == 4

    asyncio.run(run())
