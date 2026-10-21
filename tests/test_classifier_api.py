from __future__ import annotations

import asyncio
import hashlib
import json

import httpx
import numpy as np

from sigid.api import create_app
from sigid.classifier import FEATURE_VERSION
from sigid.corpus import MODES
from sigid.synthetic import generate_bpsk_wav


def make_model(path):
    feature_count = 260
    np.savez_compressed(
        path,
        mean=np.zeros(feature_count, dtype=np.float32),
        scale=np.ones(feature_count, dtype=np.float32),
        weights=np.zeros((feature_count, len(MODES)), dtype=np.float32),
        bias=np.zeros(len(MODES), dtype=np.float32),
    )
    metadata = {
        "schema_version": 1,
        "model_type": "multinomial logistic regression",
        "feature_version": FEATURE_VERSION,
        "feature_count": feature_count,
        "classes": list(MODES),
        "abstention_threshold": 0.0,
        "family_abstention_threshold": 0.0,
        "unknown_rejection_validated": False,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))


def test_classifier_api_returns_proposals_without_saving_labels(tmp_path):
    model_path = tmp_path / "method.npz"
    make_model(model_path)
    wav_path = tmp_path / "signal.wav"
    generate_bpsk_wav(wav_path, "00110101" * 64)
    app = create_app(tmp_path / "data", classifier_model_path=model_path)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            status = await client.get("/api/model/status")
            assert status.status_code == 200
            assert status.json()["available"] is True
            uploaded = await client.post(
                "/api/captures", files={"file": ("signal.wav", wav_path.read_bytes(), "audio/wav")}
            )
            capture_id = uploaded.json()["id"]
            result = await client.post(
                f"/api/captures/{capture_id}/classify",
                json={"sample_start": 100, "sample_count": 6000},
            )
            assert result.status_code == 200, result.text
            proposal = result.json()
            assert proposal["method"] == "binary frequency shift keying"
            assert proposal["method_id"] == "2fsk"
            assert proposal["family"] == "FSK"
            assert proposal["protocol"] is None
            assert proposal["capture_sha256"] == uploaded.json()["sha256"]
            assert (await client.get(f"/api/captures/{capture_id}/regions")).json() == []

            short = await client.post(
                f"/api/captures/{capture_id}/classify", json={"sample_start": 0, "sample_count": 500}
            )
            assert short.status_code == 422

    asyncio.run(exercise())


def test_classifier_api_fails_open_when_model_is_not_configured(tmp_path):
    app = create_app(tmp_path / "data")

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/api/model/status")).json()["available"] is False
            unavailable = await client.post(
                "/api/captures/unknown/classify", json={"sample_count": 2000}
            )
            assert unavailable.status_code == 503

    asyncio.run(exercise())
