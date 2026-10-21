"""HTTP API and small browser workbench for WAV replay and manual signal labels."""

from __future__ import annotations

import os
from pathlib import Path
import wave
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .dataset import CaptureStore, SignalLabel
from .spectrum import waterfall


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


def create_app(data_dir: str | Path | None = None) -> FastAPI:
    root = Path(data_dir or os.environ.get("SIG_ID_DATA_DIR", "~/.local/share/sig-id")).expanduser()
    capture_dir = root / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    store = CaptureStore(root / "dataset.sqlite3")

    app = FastAPI(title="Signal ID Workbench", version="0.1.0")
    app.state.data_dir = root
    app.state.capture_dir = capture_dir
    app.state.capture_store = store

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "release": "0.1"}

    @app.get("/api/captures")
    def list_captures() -> list[dict[str, Any]]:
        return store.list_captures()

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
