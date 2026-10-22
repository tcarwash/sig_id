from __future__ import annotations

import asyncio
import time
import wave

import httpx

from sigid.api import create_app
from sigid.dataset import CaptureStore
from sigid.synthetic import generate_2fsk_wav


class FakeFldigi:
    backend_name = "fldigi-test"

    def mode_names(self):
        return ["BPSK31", "RTTY"]


class FakeReplay:
    def __init__(self):
        self.request = None

    def decode(self, wav_path, *, mode, carrier_hz=None):
        with wave.open(str(wav_path), "rb") as audio:
            self.request = {
                "mode": mode,
                "carrier_hz": carrier_hz,
                "sample_rate": audio.getframerate(),
                "frames": audio.getnframes(),
                "channels": audio.getnchannels(),
            }
        return {"text": "HELLO FROM REGION", "status": "completed", "note": "Isolated replay finished."}


def test_saved_region_replays_exact_sample_window_and_persists_result(tmp_path):
    wav_path = tmp_path / "source.wav"
    generate_2fsk_wav(wav_path, "00110110" * 40, sample_rate=8000, symbol_rate=100)
    replay = FakeReplay()
    app = create_app(
        tmp_path / "data", fldigi_client=FakeFldigi(), fldigi_replay_backend=replay,
    )
    capture_id = app.state.capture_store.add_wav_capture(wav_path)
    start, count = 400, 2400

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                f"/api/captures/{capture_id}/decode",
                json={"mode": "RTTY", "sample_start": start, "sample_count": count,
                      "track_id": "saved-track-7", "carrier_hz": 1234},
            )
            assert response.status_code == 202, response.text
            run_id = response.json()["run_id"]
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                runs = (await client.get(f"/api/captures/{capture_id}/decode-runs")).json()
                if runs and runs[0]["status"] != "queued" and runs[0]["status"] != "running":
                    break
                await asyncio.sleep(0.01)
            assert runs[0]["id"] == run_id
            assert runs[0]["status"] == "completed"
            assert runs[0]["output_text"] == "HELLO FROM REGION"
            assert runs[0]["output_scope"] == "isolated_region_replay"
            assert runs[0]["association"] == "exact_capture_window"
            assert runs[0]["sample_start"] == start
            assert runs[0]["sample_count"] == count
            assert replay.request == {
                "mode": "RTTY", "carrier_hz": 1234, "sample_rate": 8000,
                "frames": count, "channels": 1,
            }

    asyncio.run(exercise())
    reloaded = CaptureStore(tmp_path / "data" / "dataset.sqlite3")
    [run] = reloaded.decoder_runs(capture_id=capture_id)
    assert run["output_text"] == "HELLO FROM REGION"


def test_region_replay_rejects_invalid_range_and_unavailable_mode(tmp_path):
    wav_path = tmp_path / "source.wav"
    generate_2fsk_wav(wav_path, "10101010", sample_rate=8000, symbol_rate=100)
    app = create_app(tmp_path / "data", fldigi_client=FakeFldigi(), fldigi_replay_backend=FakeReplay())
    capture_id = app.state.capture_store.add_wav_capture(wav_path)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            bad_range = await client.post(
                f"/api/captures/{capture_id}/decode",
                json={"mode": "RTTY", "sample_start": 0, "sample_count": 999999},
            )
            assert bad_range.status_code == 422
            bad_mode = await client.post(
                f"/api/captures/{capture_id}/decode",
                json={"mode": "OLIVIA", "sample_start": 0, "sample_count": 100},
            )
            assert bad_mode.status_code == 422

    asyncio.run(exercise())
