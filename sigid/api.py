"""HTTP API and small browser workbench for WAV replay and manual signal labels."""

from __future__ import annotations

import os
from pathlib import Path
import wave
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .dataset import CaptureStore, SignalLabel
from .detection import detect_tracks
from .fldigi import FldigiClient
from .live_audio import LiveCaptureManager
from .spectrum import waterfall, waterfall_samples
from .synthetic_dataset import MAX_MANIFEST_BYTES, import_synthetic_wav


MAX_UPLOAD_BYTES = 100 * 1024 * 1024
WEB_DIR = Path(__file__).resolve().parent / "web"


class RegionCreate(BaseModel):
    sample_start: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    frequency_low_hz: float | None = None
    frequency_high_hz: float | None = None
    family: str | None = None
    method: str | None = None
    protocol: str | None = None
    variant: str | None = None
    notes: str = ""


class LiveCaptureStart(BaseModel):
    device_id: int | None = None
    sample_rate: int = Field(default=48_000, ge=4000, le=192000)
    channels: int = Field(default=1, ge=1, le=2)
    max_duration_seconds: int = Field(default=600, ge=1, le=1800)


class FldigiModeSelect(BaseModel):
    name: str = Field(min_length=1, max_length=80)


def create_app(
    data_dir: str | Path | None = None,
    audio_backend: Any | None = None,
    fldigi_client: Any | None = None,
) -> FastAPI:
    root = Path(data_dir or os.environ.get("SIG_ID_DATA_DIR", "~/.local/share/sig-id")).expanduser()
    capture_dir = root / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    store = CaptureStore(root / "dataset.sqlite3")
    live_capture = LiveCaptureManager(store, capture_dir, audio_backend=audio_backend)
    fldigi = fldigi_client or FldigiClient(
        url=os.environ.get("SIG_ID_FLDIGI_URL", "http://127.0.0.1:7362/RPC2"),
        timeout=float(os.environ.get("SIG_ID_FLDIGI_TIMEOUT", "0.75")),
    )

    app = FastAPI(title="Signal ID Workbench", version="0.1.0")
    app.state.data_dir = root
    app.state.capture_dir = capture_dir
    app.state.capture_store = store
    app.state.live_capture = live_capture
    app.state.fldigi = fldigi

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "release": "0.1"}

    @app.get("/api/captures")
    def list_captures() -> list[dict[str, Any]]:
        return store.list_captures()

    @app.get("/api/audio/devices")
    def input_devices() -> list[dict[str, Any]]:
        try:
            return live_capture.devices()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/live")
    def live_status() -> dict[str, Any]:
        return live_capture.status()

    @app.get("/api/live/analysis")
    def live_analysis(
        fft_size: int = Query(default=256, ge=32, le=8192),
        hop_size: int = Query(default=64, ge=1, le=8192),
        max_rows: int = Query(default=640, ge=1, le=8192),
        threshold_db: float = Query(default=18.0, ge=1.0, le=60.0),
    ) -> dict[str, Any]:
        snapshot = live_capture.analysis_audio()
        if snapshot is None or not snapshot["active"]:
            raise HTTPException(status_code=409, detail="live capture is not active")
        if snapshot["window_sample_count"] < fft_size:
            return {
                "active": True,
                "warming_up": True,
                "sample_count": snapshot["sample_count"],
                "window_sample_count": snapshot["window_sample_count"],
                "tracks": [],
            }
        try:
            channels = snapshot["channels"]
            pcm = np.frombuffer(snapshot["pcm"], dtype="<i2").reshape(-1, channels)
            mono = pcm.astype(np.float64).mean(axis=1) / 32768.0
            spectrum = waterfall_samples(
                mono,
                snapshot["sample_rate"],
                fft_size=fft_size,
                hop_size=hop_size,
                max_rows=max_rows,
                sample_offset=snapshot["sample_offset"],
                total_sample_count=snapshot["sample_count"],
            )
            return {
                "active": True,
                "warming_up": False,
                "session_id": snapshot["session_id"],
                "spectrum": spectrum,
                "tracks": detect_tracks(spectrum, threshold_db=threshold_db),
                "analysis": {"threshold_db": threshold_db, "method": "row-median threshold with persistent peak clusters"},
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/live/start")
    def start_live_capture(data: LiveCaptureStart) -> dict[str, Any]:
        try:
            return live_capture.start(
                device_id=data.device_id,
                sample_rate=data.sample_rate,
                channels=data.channels,
                max_duration_seconds=data.max_duration_seconds,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"could not start audio input: {exc}") from exc

    @app.post("/api/live/stop")
    def stop_live_capture() -> dict[str, Any]:
        try:
            return live_capture.stop()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/fldigi/status")
    def fldigi_status() -> dict[str, Any]:
        try:
            version = fldigi.version()
            return {
                "connected": True,
                "name": version.name,
                "version": version.version,
                "current_mode": fldigi.current_modem(),
                "mode_count": len(fldigi.modem_names()),
                "receive_text_length": fldigi.receive_text_length(),
                "audio_routed": None,
                "audio_route_managed": False,
                "note": "Connect the radio passband to Fldigi's audio input to decode.",
            }
        except Exception as exc:
            return {"connected": False, "error": str(exc)}

    @app.get("/api/fldigi/modes")
    def fldigi_modes() -> list[str]:
        try:
            return fldigi.modem_names()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Fldigi is unavailable: {exc}") from exc

    @app.post("/api/fldigi/mode")
    def select_fldigi_mode(data: FldigiModeSelect) -> dict[str, Any]:
        try:
            selected = fldigi.select_modem(data.name)
            return {"current_mode": selected, "output_offset": fldigi.receive_text_length()}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Fldigi mode selection failed: {exc}") from exc

    @app.get("/api/fldigi/output")
    def fldigi_output(start: int = Query(default=0, ge=0), limit: int = Query(default=8192, ge=1, le=65536)) -> dict[str, Any]:
        try:
            end = fldigi.receive_text_length()
            offset = min(start, end)
            length = min(limit, max(0, end - offset))
            raw = fldigi.receive_text(start=offset, length=length)
            return {
                "current_mode": fldigi.current_modem(),
                "start": offset,
                "end": offset + length,
                "text": raw.decode("utf-8", errors="replace"),
                "truncated": end - offset > length,
            }
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Fldigi receive output unavailable: {exc}") from exc

    @app.post("/api/captures")
    def upload_capture(file: UploadFile = File(...)) -> dict[str, Any]:
        if file.filename is None or Path(file.filename).suffix.lower() != ".wav":
            raise HTTPException(status_code=415, detail="upload a PCM WAV file")
        capture_id = str(uuid4())
        destination = capture_dir / f"{capture_id}.wav"
        byte_count = 0
        try:
            with destination.open("wb") as output:
                while chunk := file.file.read(1024 * 1024):
                    byte_count += len(chunk)
                    if byte_count > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="WAV exceeds 100 MiB upload limit")
                    output.write(chunk)
            capture_id = store.add_wav_capture(
                destination,
                source_kind="wav",
                source_metadata={"uploaded_filename": Path(file.filename).name},
            )
        except HTTPException:
            destination.unlink(missing_ok=True)
            raise
        except (OSError, EOFError, ValueError, wave.Error) as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=415, detail=f"invalid or unsupported WAV: {exc}") from exc
        finally:
            file.file.close()
        return store.capture(capture_id)

    @app.post("/api/synthetic/import")
    def import_synthetic(
        wav_file: UploadFile = File(...),
        manifest_file: UploadFile = File(...),
    ) -> dict[str, Any]:
        if wav_file.filename is None or Path(wav_file.filename).suffix.lower() != ".wav":
            raise HTTPException(status_code=415, detail="synthetic audio must be a PCM WAV file")
        if manifest_file.filename is None or Path(manifest_file.filename).suffix.lower() != ".json":
            raise HTTPException(status_code=415, detail="include the generated .wav.json manifest")
        incoming_wav = capture_dir / f".incoming-{uuid4()}.wav"
        incoming_manifest = incoming_wav.with_suffix(incoming_wav.suffix + ".json")

        def save_upload(upload: UploadFile, destination: Path, maximum: int) -> None:
            size = 0
            with destination.open("wb") as output:
                while chunk := upload.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > maximum:
                        raise HTTPException(status_code=413, detail="synthetic upload exceeds its size limit")
                    output.write(chunk)
            if size == 0:
                raise HTTPException(status_code=422, detail="synthetic upload file is empty")

        try:
            save_upload(wav_file, incoming_wav, MAX_UPLOAD_BYTES)
            save_upload(manifest_file, incoming_manifest, MAX_MANIFEST_BYTES)
            return import_synthetic_wav(
                incoming_wav,
                incoming_manifest,
                store=store,
                capture_dir=capture_dir,
            )
        except HTTPException:
            raise
        except (OSError, EOFError, ValueError, wave.Error) as exc:
            raise HTTPException(status_code=422, detail=f"invalid synthetic WAV/manifest: {exc}") from exc
        finally:
            incoming_wav.unlink(missing_ok=True)
            incoming_manifest.unlink(missing_ok=True)
            wav_file.file.close()
            manifest_file.file.close()

    @app.get("/api/captures/{capture_id}")
    def get_capture(capture_id: str) -> dict[str, Any]:
        try:
            return store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc

    @app.get("/api/captures/{capture_id}/audio")
    def audio(capture_id: str) -> FileResponse:
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        path = Path(capture["audio_path"])
        return FileResponse(path, media_type="audio/wav", filename=f"{capture_id}.wav")

    @app.get("/api/captures/{capture_id}/manifest")
    def synthetic_manifest(capture_id: str) -> FileResponse:
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        manifest_name = capture["source_metadata"].get("manifest_file")
        if capture["source_kind"] != "synthetic" or not manifest_name:
            raise HTTPException(status_code=404, detail="capture has no synthetic manifest")
        path = (capture_dir / manifest_name).resolve()
        if not path.is_relative_to(capture_dir.resolve()) or not path.is_file():
            raise HTTPException(status_code=404, detail="synthetic manifest not found")
        return FileResponse(path, media_type="application/json", filename=f"{capture_id}.synthetic.json")

    @app.get("/api/captures/{capture_id}/waterfall")
    def get_waterfall(
        capture_id: str,
        fft_size: int = 1024,
        hop_size: int = 256,
        max_rows: int = 512,
    ) -> dict[str, Any]:
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        try:
            return waterfall(
                capture["audio_path"],
                fft_size=fft_size,
                hop_size=hop_size,
                max_rows=max_rows,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/captures/{capture_id}/tracks")
    def get_signal_tracks(
        capture_id: str,
        fft_size: int = Query(default=128, ge=32, le=8192),
        hop_size: int = Query(default=32, ge=1, le=8192),
        max_rows: int = Query(default=2048, ge=1, le=8192),
        threshold_db: float = Query(default=18.0, ge=1.0, le=60.0),
    ) -> dict[str, Any]:
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        try:
            spectrum = waterfall(
                capture["audio_path"],
                fft_size=fft_size,
                hop_size=hop_size,
                max_rows=max_rows,
            )
            return {
                "tracks": detect_tracks(spectrum, threshold_db=threshold_db),
                "analysis": {
                    "fft_size": fft_size,
                    "hop_size": hop_size,
                    "rows": len(spectrum["db"]),
                    "threshold_db": threshold_db,
                    "method": "row-median threshold with persistent peak clusters",
                },
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/captures/{capture_id}/regions")
    def get_regions(capture_id: str) -> list[dict[str, Any]]:
        try:
            return store.regions(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc

    @app.post("/api/captures/{capture_id}/regions", status_code=201)
    def create_region(capture_id: str, data: RegionCreate) -> dict[str, Any]:
        try:
            region_id = store.add_region(
                capture_id,
                sample_start=data.sample_start,
                sample_count=data.sample_count,
                frequency_low_hz=data.frequency_low_hz,
                frequency_high_hz=data.frequency_high_hz,
                label=SignalLabel(
                    family=data.family,
                    method=data.method,
                    protocol=data.protocol,
                    variant=data.variant,
                    source="human" if any((data.family, data.method, data.protocol, data.variant)) else "unknown",
                    review_state="accepted" if any((data.family, data.method, data.protocol, data.variant)) else "pending",
                    notes=data.notes,
                ),
            )
            return next(region for region in store.regions(capture_id) if region["id"] == region_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/exports/regions.jsonl")
    def export_regions() -> FileResponse:
        destination = root / "exports" / "regions.jsonl"
        store.export_jsonl(destination)
        return FileResponse(
            destination,
            media_type="application/x-ndjson",
            filename="signal-regions.jsonl",
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/app.js", include_in_schema=False)
    def javascript() -> FileResponse:
        return FileResponse(WEB_DIR / "app.js", media_type="text/javascript")

    @app.get("/style.css", include_in_schema=False)
    def stylesheet() -> FileResponse:
        return FileResponse(WEB_DIR / "style.css", media_type="text/css")

    return app
