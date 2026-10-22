import io
import wave

import pytest
from fastapi.testclient import TestClient

from sigid.hub import create_hub_app


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(b"\0\0" * 800)
    return output.getvalue()


def test_hub_requires_upload_auth_and_accepts_audio_bundle(tmp_path):
    app = create_hub_app(tmp_path, upload_token="collector", reviewer_tokens={"alice", "bob"})
    with TestClient(app) as client:
        denied = client.post("/api/submissions", files={"audio": ("signal.wav", wav_bytes(), "audio/wav")}, data={"manifest": "{}"})
        assert denied.status_code == 401
        uploaded = client.post(
            "/api/submissions",
            headers={"Authorization": "Bearer collector"},
            files={"audio": ("signal.wav", wav_bytes(), "audio/wav")},
            data={"manifest": '{"schema_version":1,"sample_rate":8000,"channels":1,"sample_count":800,"annotations":[{"sample_start":100,"sample_count":400,"frequency_low_hz":500,"frequency_high_hz":1200,"family":"FSK"}]}'},
        )
        assert uploaded.status_code == 201, uploaded.text
        assert uploaded.json()["status"] == "awaiting_review"
        assert len(list((tmp_path / "assets").glob("*.wav"))) == 1


def test_hub_requires_two_distinct_non_submitter_reviews_and_rejects_stale_revision(tmp_path):
    app = create_hub_app(tmp_path, upload_token="collector", reviewer_tokens={"collector": "collector", "alice": "alice", "bob": "bob"})
    with TestClient(app) as client:
        created = client.post(
            "/api/submissions", headers={"Authorization": "Bearer collector"},
            files={"audio": ("signal.wav", wav_bytes(), "audio/wav")},
            data={"manifest": '{"schema_version":1,"sample_rate":8000,"channels":1,"sample_count":800,"annotations":[{"sample_start":100,"sample_count":400}]}'},
        ).json()
        sid, rid = created["id"], created["annotations"][0]["id"]
        self_review = client.post(f"/api/submissions/{sid}/annotations/{rid}/reviews", headers={"Authorization": "Bearer collector"}, json={"revision":1,"decision":"looks_right"})
        assert self_review.status_code == 403
        assert client.get(f"/api/submissions/{sid}/audio").status_code == 401
        assert client.get(f"/api/submissions/{sid}/audio", headers={"Authorization": "Bearer alice"}).status_code == 200
        first = client.post(f"/api/submissions/{sid}/annotations/{rid}/reviews", headers={"Authorization": "Bearer alice"}, json={"revision":1,"decision":"looks_right"})
        assert first.status_code == 201, first.text
        assert first.json()["consensus"] == "awaiting_second_review"
        duplicate_reviewer = client.post(f"/api/submissions/{sid}/annotations/{rid}/reviews", headers={"Authorization": "Bearer alice"}, json={"revision":1,"decision":"looks_right"})
        assert duplicate_reviewer.status_code == 409
        second = client.post(f"/api/submissions/{sid}/annotations/{rid}/reviews", headers={"Authorization": "Bearer bob"}, json={"revision":1,"decision":"looks_right"})
        assert second.status_code == 201
        assert second.json()["consensus"] == "extent_verified"
        stale = client.post(f"/api/submissions/{sid}/annotations/{rid}/reviews", headers={"Authorization": "Bearer bob"}, json={"revision":2,"decision":"looks_right"})
        assert stale.status_code == 409


def test_hub_validates_annotation_bounds_and_review_tokens(tmp_path):
    app = create_hub_app(tmp_path, upload_token="collector", reviewer_tokens={"alice"})
    with TestClient(app) as client:
        bad = client.post(
            "/api/submissions", headers={"Authorization": "Bearer collector"},
            files={"audio": ("signal.wav", wav_bytes(), "audio/wav")},
            data={"manifest": '{"schema_version":1,"sample_rate":8000,"channels":1,"sample_count":800,"annotations":[{"sample_start":700,"sample_count":200}]}'},
        )
        assert bad.status_code == 422
        denied = client.get("/api/review/queue")
        assert denied.status_code == 401
