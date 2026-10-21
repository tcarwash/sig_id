from __future__ import annotations

import asyncio
import hashlib
import json
import wave

import httpx
import numpy as np

from sigid.api import create_app
from sigid.classifier import FEATURE_COUNT_BY_VERSION, FEATURE_VERSION, LEGACY_FEATURE_VERSION, load_model
from sigid.corpus import MODES
from sigid.synthetic import generate_bpsk_wav, generate_ppm_wav


class FakeReceiveBackend:
    def __init__(self):
        self.mode = "BPSK31"

    def mode_names(self):
        return ["BPSK31", "RTTY", "THOR16"]

    def select_mode(self, name):
        self.mode = name
        return name

    def receive_text_length(self):
        return 0

    def receive_text(self, *, start=0, length=65536):
        return b""


def make_model(path, *, novelty_threshold=100.0):
    feature_count = FEATURE_COUNT_BY_VERSION[FEATURE_VERSION]
    np.savez_compressed(
        path,
        mean=np.zeros(feature_count, dtype=np.float32),
        scale=np.ones(feature_count, dtype=np.float32),
        weights=np.zeros((feature_count, len(MODES)), dtype=np.float32),
        bias=np.zeros(len(MODES), dtype=np.float32),
        support_centers=np.zeros((len(MODES), feature_count), dtype=np.float32),
    )
    metadata = {
        "schema_version": 1,
        "model_type": "multinomial logistic regression",
        "feature_version": FEATURE_VERSION,
        "feature_count": feature_count,
        "classes": list(MODES),
        "abstention_threshold": 0.0,
        "family_abstention_threshold": 0.0,
        "novelty_distance_threshold": novelty_threshold,
        "unknown_rejection_validated": False,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))


def test_loader_keeps_legacy_spectral_models_usable(tmp_path):
    path = tmp_path / "legacy.npz"
    count = FEATURE_COUNT_BY_VERSION[LEGACY_FEATURE_VERSION]
    np.savez_compressed(
        path,
        mean=np.zeros(count, dtype=np.float32),
        scale=np.ones(count, dtype=np.float32),
        weights=np.zeros((count, len(MODES)), dtype=np.float32),
        bias=np.zeros(len(MODES), dtype=np.float32),
    )
    path.with_suffix(".json").write_text(json.dumps({
        "schema_version": 1,
        "feature_version": LEGACY_FEATURE_VERSION,
        "feature_count": count,
        "classes": list(MODES),
        "abstention_threshold": 0.5,
        "family_abstention_threshold": 0.5,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }))
    metadata, parameters = load_model(path)
    assert metadata["feature_version"] == LEGACY_FEATURE_VERSION
    assert parameters["weights"].shape == (count, len(MODES))
    assert "support_centers" not in parameters


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
            assert status.json()["novelty_distance_threshold"] == 100.0
            assert status.json()["novelty_rejection_validated"] is False
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
            assert proposal["novelty_rejected"] is False
            assert proposal["novelty_distance"] is not None
            assert proposal["protocol"] is None
            assert proposal["capture_sha256"] == uploaded.json()["sha256"]
            assert (await client.get(f"/api/captures/{capture_id}/regions")).json() == []

            short = await client.post(
                f"/api/captures/{capture_id}/classify", json={"sample_start": 0, "sample_count": 500}
            )
            assert short.status_code == 422

    asyncio.run(exercise())


def test_classifier_api_returns_novelty_abstention_for_out_of_support_audio(tmp_path):
    model_path = tmp_path / "method.npz"
    make_model(model_path, novelty_threshold=0.01)
    wav_path = tmp_path / "ppm.wav"
    generate_ppm_wav(wav_path, "01011010" * 16)
    app = create_app(tmp_path / "data", classifier_model_path=model_path)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            uploaded = await client.post(
                "/api/captures", files={"file": ("ppm.wav", wav_path.read_bytes(), "audio/wav")}
            )
            response = await client.post(
                f"/api/captures/{uploaded.json()['id']}/classify",
                json={"sample_start": 0, "sample_count": 3840},
            )
            assert response.status_code == 200, response.text
            proposal = response.json()
            assert proposal["top_candidate"] == "2fsk"
            assert proposal["novelty_rejected"] is True
            assert proposal["novelty_distance"] > proposal["novelty_threshold"]
            assert proposal["method"] is None
            assert proposal["abstained"] is True

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


def test_live_candidate_gets_bounded_model_proposal_and_model_ranking(tmp_path, monkeypatch):
    model_path = tmp_path / "method.npz"
    make_model(model_path)
    app = create_app(
        tmp_path / "data", classifier_model_path=model_path, fldigi_client=FakeReceiveBackend()
    )
    sample_rate = 12_000
    signal_path = tmp_path / "live.wav"
    generate_bpsk_wav(signal_path, "00110101" * 32, sample_rate=sample_rate)
    with wave.open(str(signal_path), "rb") as source:
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    snapshot = {
        "pcm": samples.tobytes(), "sample_rate": sample_rate, "channels": 1,
        "sample_offset": 0, "sample_count": len(samples),
        "window_sample_count": len(samples), "session_id": "live-model-test", "active": True,
    }
    app.state.live_capture.analysis_audio = lambda: snapshot
    monkeypatch.setattr(app.state.live_capture, "status", lambda: {"active": True})
    monkeypatch.setattr(
        "sigid.api.detect_tracks",
        lambda spectrum, threshold_db: [
            {
                "id": "short", "sample_start": 0, "sample_count": 500,
                "center_hz": 2400.0, "bandwidth_hz": 200.0, "peak_snr_db": 40.0,
                "tone_count_candidate": 2, "modulation_hypothesis": "FSK-like",
                "frequency_low_hz": 2300.0, "frequency_high_hz": 2500.0,
                "time_start_seconds": 0.0, "time_end_seconds": 500 / sample_rate,
            },
            {
                "id": "scored", "sample_start": 0, "sample_count": len(samples),
                "center_hz": 1500.0, "bandwidth_hz": 300.0, "peak_snr_db": 20.0,
                "tone_count_candidate": 2, "modulation_hypothesis": "FSK-like",
                "frequency_low_hz": 1300.0, "frequency_high_hz": 1700.0,
                "time_start_seconds": 0.0, "time_end_seconds": len(samples) / sample_rate,
            },
        ],
    )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/api/live/analysis", params={"fft_size": 256, "hop_size": 64, "auto_trial": True}
            )
            assert response.status_code == 200, response.text
            tracks = response.json()["tracks"]
            assert tracks[0]["id"] == "scored"
            track = tracks[0]
            proposal = track["model_proposal"]
            assert proposal["available"] is True
            assert proposal["abstained"] is False
            assert proposal["method_id"] == "2fsk"
            assert proposal["protocol"] is None
            assert proposal["sample_start"] == len(samples) - 2 * sample_rate
            assert proposal["sample_count"] == 2 * sample_rate
            assert proposal["model_sha256"] == app.state.classifier["metadata"]["model_sha256"]
            assert tracks[1]["model_proposal"]["available"] is False
            assert "less than 1024" in tracks[1]["model_proposal"]["reason"]

            assert response.json()["automatic_trial"] is None
            app.state.classifier["metadata"]["abstention_threshold"] = 1.0
            snapshot["sample_count"] += 1
            abstained = await client.get(
                "/api/live/analysis", params={"fft_size": 256, "hop_size": 64, "auto_trial": True}
            )
            assert abstained.json()["tracks"][0]["model_proposal"]["abstained"] is True
            assert abstained.json()["automatic_trial"] is None
            app.state.classifier["metadata"]["abstention_threshold"] = 0.0
            snapshot["sample_count"] += 1
            response = await client.get(
                "/api/live/analysis", params={"fft_size": 256, "hop_size": 64, "auto_trial": True}
            )
            result = response.json()
            assert result["tracks"][0]["track_state"] == "persistent"
            assert result["automatic_trial"]["status"] == "running"
            assert result["automatic_trial"]["trigger"] == "accepted_model_proposal"
            assert result["automatic_trial"]["current_mode"] == "RTTY"

    asyncio.run(exercise())
