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
from .classifier import extract_features, load_model, propose
from .demod import demodulate_2fsk, demodulate_bpsk, demodulate_mfsk, demodulate_ook, demodulate_qpsk
from .detection import detect_tracks
from .fldigi import FldigiClient
from .live_audio import LiveCaptureManager
from .receive_products import ReceiveProductStore
from .spectrum import waterfall, waterfall_samples
from .synthetic_dataset import MAX_MANIFEST_BYTES, import_synthetic_wav
from .tracking import LiveTrackRegistry
from .trials import FldigiTrialRunner
from .wav import WavSource


MAX_UPLOAD_BYTES = 100 * 1024 * 1024
WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_MFSK_RATE_SEARCH_WORK = 200_000_000


def default_track_fft_size(sample_rate: int) -> int:
    """Choose a track-analysis window near 42 ms, capped by the API FFT limit."""
    target = max(32, int(np.ceil(sample_rate / 24)))
    return min(8192, 1 << int(np.ceil(np.log2(target))))


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


class FldigiTrialStart(BaseModel):
    track_id: str = Field(min_length=1, max_length=128)
    modulation_hypothesis: str = Field(default="unknown", max_length=80)


class FskDemodRequest(BaseModel):
    sample_start: int = Field(default=0, ge=0)
    sample_count: int = Field(default=120_000, ge=1, le=1_000_000)
    symbol_rate: float = Field(default=100, gt=0, le=20_000)
    tone_0_hz: float = Field(default=1000, gt=0, le=96_000)
    tone_1_hz: float = Field(default=1200, gt=0, le=96_000)
    timing_offset_samples: float = Field(default=0, ge=0)
    auto_timing: bool = False
    integration_fraction: float = Field(default=0.8, ge=0.2, le=1.0)


class OokDemodRequest(BaseModel):
    sample_start: int = Field(default=0, ge=0)
    sample_count: int = Field(default=120_000, ge=1, le=1_000_000)
    symbol_rate: float = Field(default=100, gt=0, le=20_000)
    carrier_hz: float = Field(default=1500, gt=0, le=96_000)
    timing_offset_samples: float = Field(default=0, ge=0)
    auto_timing: bool = False
    integration_fraction: float = Field(default=0.8, ge=0.2, le=1.0)


class MfskDemodRequest(BaseModel):
    sample_start: int = Field(default=0, ge=0)
    sample_count: int = Field(default=120_000, ge=1, le=1_000_000)
    symbol_rate: float = Field(default=50, gt=0, le=20_000)
    tone_count: int = Field(default=4, ge=2, le=64)
    base_tone_hz: float = Field(default=1000, gt=0, le=96_000)
    tone_spacing_hz: float = Field(default=200, gt=0, le=96_000)
    timing_offset_samples: float = Field(default=0, ge=0)
    auto_timing: bool = False
    auto_rate_search: bool = False
    rate_search_min: float = Field(default=20, gt=0, le=20_000)
    rate_search_max: float = Field(default=500, gt=0, le=20_000)
    rate_search_steps: int = Field(default=9, ge=3, le=24)
    integration_fraction: float = Field(default=0.8, ge=0.2, le=1.0)


class BpskDemodRequest(BaseModel):
    sample_start: int = Field(default=0, ge=0)
    sample_count: int = Field(default=120_000, ge=1, le=1_000_000)
    symbol_rate: float = Field(default=100, gt=0, le=20_000)
    carrier_hz: float = Field(default=1500, gt=0, le=96_000)
    timing_offset_samples: float = Field(default=0, ge=0)
    integration_fraction: float = Field(default=0.8, ge=0.2, le=1.0)


class QpskDemodRequest(BaseModel):
    sample_start: int = Field(default=0, ge=0)
    sample_count: int = Field(default=120_000, ge=1, le=1_000_000)
    symbol_rate: float = Field(default=100, gt=0, le=20_000)
    carrier_hz: float = Field(default=1500, gt=0, le=96_000)
    timing_offset_samples: float = Field(default=0, ge=0)
    integration_fraction: float = Field(default=0.8, ge=0.2, le=1.0)


class ClassifyRequest(BaseModel):
    sample_start: int = Field(default=0, ge=0)
    sample_count: int = Field(default=120_000, ge=1024, le=1_000_000)


def create_app(
    data_dir: str | Path | None = None,
    audio_backend: Any | None = None,
    fldigi_client: Any | None = None,
    classifier_model_path: str | Path | None = None,
) -> FastAPI:
    root = Path(data_dir or os.environ.get("SIG_ID_DATA_DIR", "~/.local/share/sig-id")).expanduser()
    capture_dir = root / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    store = CaptureStore(root / "dataset.sqlite3")
    live_capture = LiveCaptureManager(store, capture_dir, audio_backend=audio_backend)
    receive_products = ReceiveProductStore(root / "receive-products")
    fldigi = fldigi_client or FldigiClient(
        url=os.environ.get("SIG_ID_FLDIGI_URL", "http://127.0.0.1:7362/RPC2"),
        timeout=float(os.environ.get("SIG_ID_FLDIGI_TIMEOUT", "0.75")),
    )
    configured_model = classifier_model_path or os.environ.get("SIG_ID_MODEL_PATH")
    classifier = None
    classifier_error = None
    if configured_model:
        try:
            model_metadata, model_parameters = load_model(configured_model)
            classifier = {"metadata": model_metadata, "parameters": model_parameters}
        except Exception as exc:
            classifier_error = str(exc)

    app = FastAPI(title="Signal ID Workbench", version="0.1.0")
    app.state.data_dir = root
    app.state.capture_dir = capture_dir
    app.state.capture_store = store
    app.state.live_capture = live_capture
    app.state.receive_products = receive_products
    app.state.fldigi = fldigi
    app.state.classifier = classifier
    app.state.classifier_error = classifier_error
    app.state.live_track_registry = None
    app.state.fldigi_trials = FldigiTrialRunner(fldigi)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "release": "0.1"}

    @app.get("/api/model/status")
    def model_status() -> dict[str, Any]:
        if app.state.classifier is None:
            return {"available": False, "error": app.state.classifier_error}
        metadata = app.state.classifier["metadata"]
        return {
            "available": True,
            "model_type": metadata["model_type"],
            "feature_version": metadata["feature_version"],
            "classes": metadata["classes"],
            "threshold": metadata["abstention_threshold"],
            "family_threshold": metadata["family_abstention_threshold"],
            "unknown_rejection_validated": metadata["unknown_rejection_validated"],
            "unknown_rejection_scope": metadata.get("unknown_rejection_scope", "unvalidated"),
            "novelty_distance_threshold": metadata.get("novelty_distance_threshold"),
            "novelty_rejection_validated": metadata.get("novelty_rejection_validated", False),
            "noise_class_trained": metadata.get("noise_class_trained", False),
            "noise_class_validated": metadata.get("noise_class_validated", False),
            "model_sha256": metadata["model_sha256"],
        }

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
        fft_size: int = Query(default=4096, ge=32, le=8192),
        hop_size: int = Query(default=512, ge=1, le=8192),
        max_rows: int = Query(default=640, ge=1, le=8192),
        threshold_db: float = Query(default=18.0, ge=1.0, le=60.0),
        auto_trial: bool = Query(default=False),
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
            detections = detect_tracks(spectrum, threshold_db=threshold_db)
            registry = app.state.live_track_registry
            if registry is None or registry.session_id != snapshot["session_id"]:
                registry = LiveTrackRegistry(snapshot["session_id"])
                app.state.live_track_registry = registry
            tracks = registry.update(
                detections,
                sample_position=int(snapshot["sample_count"]),
                sample_rate=int(snapshot["sample_rate"]),
            )
            model = app.state.classifier
            if model is not None and snapshot["sample_rate"] >= 8000:
                # Score only the portion of each candidate that is still in the
                # bounded ring, with a strict two-second work cap per track.
                ring_start = int(snapshot["sample_offset"])
                ring_end = ring_start + len(mono)
                scoring_tracks = set(
                    id(track) for track in sorted(
                        tracks, key=lambda item: float(item.get("peak_snr_db", 0.0)), reverse=True
                    )[:4]
                )
                for track in tracks:
                    if id(track) not in scoring_tracks:
                        track["model_proposal"] = {
                            "available": False,
                            "reason": "per-update model scoring limit reached",
                        }
                        continue
                    segment_start = max(ring_start, int(track["sample_start"]))
                    segment_end = min(ring_end, int(track["sample_start"]) + int(track["sample_count"]))
                    segment_start = max(segment_start, segment_end - 2 * int(snapshot["sample_rate"]))
                    if segment_end - segment_start < 1024:
                        track["model_proposal"] = {
                            "available": False,
                            "reason": "candidate has less than 1024 in-ring samples",
                        }
                        continue
                    left = segment_start - ring_start
                    right = segment_end - ring_start
                    try:
                        features = extract_features(
                            mono[left:right],
                            int(snapshot["sample_rate"]),
                            feature_version=model["metadata"]["feature_version"],
                        )
                        proposal = propose(
                            features,
                            model["parameters"],
                            model["metadata"]["classes"],
                            threshold=float(model["metadata"]["abstention_threshold"]),
                            family_threshold=float(model["metadata"]["family_abstention_threshold"]),
                            novelty_threshold=model["metadata"].get("novelty_distance_threshold"),
                        )
                        track["model_proposal"] = {
                            "available": True,
                            **proposal,
                            "model_sha256": model["metadata"]["model_sha256"],
                            "sample_start": segment_start,
                            "sample_count": segment_end - segment_start,
                        }
                    except ValueError as exc:
                        track["model_proposal"] = {"available": False, "reason": str(exc)}
                tracks.sort(
                    key=lambda track: (
                        bool(track.get("model_proposal", {}).get("method")),
                        float(track.get("model_proposal", {}).get("confidence", 0.0)),
                        float(track.get("peak_snr_db", 0.0)),
                    ),
                    reverse=True,
                )
            automatic_trial = None
            if auto_trial and model is not None:
                for track in tracks:
                    proposal = track.get("model_proposal", {})
                    if (
                        track.get("track_state") != "persistent"
                        or not proposal.get("available")
                        or proposal.get("abstained")
                        or not proposal.get("method")
                        or not proposal.get("family")
                    ):
                        continue
                    try:
                        automatic_trial = app.state.fldigi_trials.start(
                            track_id=track["track_id"],
                            modulation_hypothesis={
                                "FSK": "FSK-like",
                                "MFSK": "MFSK-like",
                                "PSK": "single-tone / unknown",
                            }.get(proposal["family"], proposal["family"]),
                            live=True,
                        )
                        automatic_trial["trigger"] = "accepted_model_proposal"
                        track["decoder_trial"] = {
                            "status": automatic_trial["status"],
                            "mode": automatic_trial["current_mode"],
                        }
                    except ValueError as exc:
                        reason = str(exc)
                        if "already had" in reason:
                            continue
                        if "already running" in reason:
                            automatic_trial = None
                            break
                        automatic_trial = {"status": "not_started", "reason": reason}
                    except Exception as exc:
                        automatic_trial = {"status": "failed", "reason": str(exc)}
                    break
            return {
                "active": True,
                "warming_up": False,
                "session_id": snapshot["session_id"],
                "spectrum": spectrum,
                "tracks": tracks,
                "automatic_trial": automatic_trial,
                "analysis": {"threshold_db": threshold_db, "method": "row-median threshold with persistent peak clusters"},
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/live/start")
    def start_live_capture(data: LiveCaptureStart) -> dict[str, Any]:
        try:
            started = live_capture.start(
                device_id=data.device_id,
                sample_rate=data.sample_rate,
                channels=data.channels,
                max_duration_seconds=data.max_duration_seconds,
            )
            app.state.live_track_registry = LiveTrackRegistry(started["session_id"])
            return started
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"could not start audio input: {exc}") from exc

    @app.post("/api/live/stop")
    def stop_live_capture() -> dict[str, Any]:
        try:
            result = live_capture.stop()
            app.state.fldigi_trials.cancel()
            return result
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
                "wefax_output_available": callable(getattr(fldigi, "receive_wefax_file", None)),
                "note": "Connect the radio passband to Fldigi's audio input to decode.",
            }
        except Exception as exc:
            return {"connected": False, "error": str(exc)}

    @app.get("/api/fldigi/modes")
    def fldigi_modes() -> list[str]:
        try:
            return fldigi.mode_names()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Fldigi is unavailable: {exc}") from exc

    @app.post("/api/fldigi/mode")
    def select_fldigi_mode(data: FldigiModeSelect) -> dict[str, Any]:
        try:
            app.state.fldigi_trials.cancel()
            selected = fldigi.select_modem(data.name)
            return {"current_mode": selected, "output_offset": fldigi.receive_text_length()}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Fldigi mode selection failed: {exc}") from exc

    @app.post("/api/fldigi/trials/start")
    def start_fldigi_trial(data: FldigiTrialStart) -> dict[str, Any]:
        try:
            live = bool(live_capture.status().get("active"))
            family = data.modulation_hypothesis.upper()
            trial_hint = {
                "FSK": "FSK-like",
                "MFSK": "MFSK-like",
                "PSK": "single-tone / unknown",
            }.get(family, data.modulation_hypothesis)
            return app.state.fldigi_trials.start(
                track_id=data.track_id,
                modulation_hypothesis=trial_hint,
                live=live,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Fldigi trial could not start: {exc}") from exc

    @app.get("/api/fldigi/trials/current")
    def current_fldigi_trial() -> dict[str, Any]:
        try:
            if not bool(live_capture.status().get("active")):
                return app.state.fldigi_trials.cancel()
            return app.state.fldigi_trials.advance()
        except Exception as exc:
            app.state.fldigi_trials.cancel()
            raise HTTPException(status_code=503, detail=f"Fldigi trial failed: {exc}") from exc

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

    @app.get("/api/fldigi/products")
    def list_fldigi_products() -> list[dict[str, Any]]:
        return receive_products.list()

    @app.post("/api/fldigi/products/wefax/poll")
    def poll_fldigi_wefax(timeout_seconds: int = Query(default=0, ge=0, le=1)) -> dict[str, Any]:
        receive_wefax_file = getattr(fldigi, "receive_wefax_file", None)
        if receive_wefax_file is None:
            raise HTTPException(status_code=501, detail="the configured decoder backend does not expose WEFAX products")
        try:
            path = receive_wefax_file(timeout_seconds=timeout_seconds)
            product = receive_products.import_wefax(path) if path else None
            return {"received": product is not None, "product": product, "products": receive_products.list()}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Fldigi WEFAX output unavailable: {exc}") from exc

    @app.get("/api/fldigi/products/{product_id}/image")
    def fldigi_product_image(product_id: str) -> FileResponse:
        try:
            path, media_type = receive_products.image_path(product_id)
        except (KeyError, OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="received image not found") from exc
        return FileResponse(path, media_type=media_type, filename=path.name)

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
        fft_size: int = 4096,
        hop_size: int = 512,
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
        fft_size: int | None = Query(default=None, ge=32, le=8192),
        hop_size: int | None = Query(default=None, ge=1, le=8192),
        max_rows: int = Query(default=2048, ge=1, le=8192),
        threshold_db: float = Query(default=18.0, ge=1.0, le=60.0),
    ) -> dict[str, Any]:
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        try:
            fft_size = fft_size or default_track_fft_size(int(capture["sample_rate"]))
            hop_size = hop_size or max(1, fft_size // 8)
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

    @app.post("/api/captures/{capture_id}/classify")
    def classify_capture_region(capture_id: str, settings: ClassifyRequest) -> dict[str, Any]:
        if app.state.classifier is None:
            raise HTTPException(status_code=503, detail="no valid modulation model is configured")
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        start = settings.sample_start
        if start >= int(capture["sample_count"]):
            raise HTTPException(status_code=422, detail="sample_start is outside the capture")
        end = min(int(capture["sample_count"]), start + settings.sample_count)
        try:
            values: list[float] = []
            for block in WavSource(capture["audio_path"]).blocks():
                block_end = block.first_sample + block.sample_count
                if block_end <= start:
                    continue
                if block.first_sample >= end:
                    break
                left = max(0, start - block.first_sample)
                right = min(block.sample_count, end - block.first_sample)
                values.extend(block.samples[left:right])
            model = app.state.classifier
            features = extract_features(
                values,
                int(capture["sample_rate"]),
                feature_version=model["metadata"]["feature_version"],
            )
            result = propose(
                features,
                model["parameters"],
                model["metadata"]["classes"],
                threshold=float(model["metadata"]["abstention_threshold"]),
                family_threshold=float(model["metadata"]["family_abstention_threshold"]),
                novelty_threshold=model["metadata"].get("novelty_distance_threshold"),
            )
            return {
                **result,
                "model_sha256": model["metadata"]["model_sha256"],
                "capture_id": capture_id,
                "sample_start": start,
                "sample_count": len(values),
                "capture_sha256": capture["sha256"],
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/captures/{capture_id}/demod/2fsk")
    def demodulate_capture_2fsk(capture_id: str, settings: FskDemodRequest) -> dict[str, Any]:
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        start = settings.sample_start
        end = min(int(capture["sample_count"]), start + settings.sample_count)
        if start >= int(capture["sample_count"]):
            raise HTTPException(status_code=422, detail="sample_start is outside the capture")
        try:
            values: list[float] = []
            for block in WavSource(capture["audio_path"]).blocks():
                block_end = block.first_sample + block.sample_count
                if block_end <= start:
                    continue
                if block.first_sample >= end:
                    break
                left = max(0, start - block.first_sample)
                right = min(block.sample_count, end - block.first_sample)
                values.extend(block.samples[left:right])
            result = demodulate_2fsk(
                values,
                int(capture["sample_rate"]),
                symbol_rate=settings.symbol_rate,
                tone_0_hz=settings.tone_0_hz,
                tone_1_hz=settings.tone_1_hz,
                timing_offset_samples=settings.timing_offset_samples,
                auto_timing=settings.auto_timing,
                sample_offset=start,
                integration_fraction=settings.integration_fraction,
            )
            result["capture_id"] = capture_id
            result["sample_start"] = start
            result["sample_count"] = len(values)
            result["protocol"] = None
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/captures/{capture_id}/demod/ook")
    def demodulate_capture_ook(capture_id: str, settings: OokDemodRequest) -> dict[str, Any]:
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        start = settings.sample_start
        end = min(int(capture["sample_count"]), start + settings.sample_count)
        if start >= int(capture["sample_count"]):
            raise HTTPException(status_code=422, detail="sample_start is outside the capture")
        try:
            values: list[float] = []
            for block in WavSource(capture["audio_path"]).blocks():
                block_end = block.first_sample + block.sample_count
                if block_end <= start:
                    continue
                if block.first_sample >= end:
                    break
                left = max(0, start - block.first_sample)
                right = min(block.sample_count, end - block.first_sample)
                values.extend(block.samples[left:right])
            result = demodulate_ook(
                values,
                int(capture["sample_rate"]),
                symbol_rate=settings.symbol_rate,
                carrier_hz=settings.carrier_hz,
                timing_offset_samples=settings.timing_offset_samples,
                auto_timing=settings.auto_timing,
                sample_offset=start,
                integration_fraction=settings.integration_fraction,
            )
            result["capture_id"] = capture_id
            result["capture_sha256"] = capture["sha256"]
            result["sample_start"] = start
            result["sample_count"] = len(values)
            result["protocol"] = None
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/captures/{capture_id}/demod/mfsk")
    def demodulate_capture_mfsk(capture_id: str, settings: MfskDemodRequest) -> dict[str, Any]:
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        start = settings.sample_start
        end = min(int(capture["sample_count"]), start + settings.sample_count)
        if start >= int(capture["sample_count"]):
            raise HTTPException(status_code=422, detail="sample_start is outside the capture")
        estimated_work = (end - start) * settings.tone_count * settings.rate_search_steps * 32
        if settings.auto_rate_search and estimated_work > MAX_MFSK_RATE_SEARCH_WORK:
            raise HTTPException(
                status_code=422,
                detail="M-FSK rate search is too large; select a shorter region, fewer tones, or fewer rate hypotheses",
            )
        try:
            values: list[float] = []
            for block in WavSource(capture["audio_path"]).blocks():
                block_end = block.first_sample + block.sample_count
                if block_end <= start:
                    continue
                if block.first_sample >= end:
                    break
                left = max(0, start - block.first_sample)
                right = min(block.sample_count, end - block.first_sample)
                values.extend(block.samples[left:right])
            frequencies = [settings.base_tone_hz + index * settings.tone_spacing_hz for index in range(settings.tone_count)]
            result = demodulate_mfsk(
                values,
                int(capture["sample_rate"]),
                symbol_rate=settings.symbol_rate,
                tone_frequencies_hz=frequencies,
                timing_offset_samples=settings.timing_offset_samples,
                auto_timing=settings.auto_timing,
                auto_rate_search=settings.auto_rate_search,
                rate_search_min=settings.rate_search_min,
                rate_search_max=settings.rate_search_max,
                rate_search_steps=settings.rate_search_steps,
                sample_offset=start,
                integration_fraction=settings.integration_fraction,
            )
            result["capture_id"] = capture_id
            result["sample_start"] = start
            result["sample_count"] = len(values)
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/captures/{capture_id}/demod/bpsk")
    def demodulate_capture_bpsk(capture_id: str, settings: BpskDemodRequest) -> dict[str, Any]:
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        start = settings.sample_start
        end = min(int(capture["sample_count"]), start + settings.sample_count)
        if start >= int(capture["sample_count"]):
            raise HTTPException(status_code=422, detail="sample_start is outside the capture")
        try:
            values: list[float] = []
            for block in WavSource(capture["audio_path"]).blocks():
                block_end = block.first_sample + block.sample_count
                if block_end <= start:
                    continue
                if block.first_sample >= end:
                    break
                left = max(0, start - block.first_sample)
                right = min(block.sample_count, end - block.first_sample)
                values.extend(block.samples[left:right])
            result = demodulate_bpsk(
                values,
                int(capture["sample_rate"]),
                symbol_rate=settings.symbol_rate,
                carrier_hz=settings.carrier_hz,
                timing_offset_samples=settings.timing_offset_samples,
                sample_offset=start,
                integration_fraction=settings.integration_fraction,
            )
            result["capture_id"] = capture_id
            result["sample_start"] = start
            result["sample_count"] = len(values)
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/captures/{capture_id}/demod/qpsk")
    def demodulate_capture_qpsk(capture_id: str, settings: QpskDemodRequest) -> dict[str, Any]:
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        start = settings.sample_start
        end = min(int(capture["sample_count"]), start + settings.sample_count)
        if start >= int(capture["sample_count"]):
            raise HTTPException(status_code=422, detail="sample_start is outside the capture")
        try:
            values: list[float] = []
            for block in WavSource(capture["audio_path"]).blocks():
                block_end = block.first_sample + block.sample_count
                if block_end <= start:
                    continue
                if block.first_sample >= end:
                    break
                left = max(0, start - block.first_sample)
                right = min(block.sample_count, end - block.first_sample)
                values.extend(block.samples[left:right])
            result = demodulate_qpsk(
                values,
                int(capture["sample_rate"]),
                symbol_rate=settings.symbol_rate,
                carrier_hz=settings.carrier_hz,
                timing_offset_samples=settings.timing_offset_samples,
                sample_offset=start,
                integration_fraction=settings.integration_fraction,
            )
            result["capture_id"] = capture_id
            result["sample_start"] = start
            result["sample_count"] = len(values)
            return result
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

    @app.get("/waterfall-window.mjs", include_in_schema=False)
    def waterfall_window_module() -> FileResponse:
        return FileResponse(WEB_DIR / "waterfall-window.mjs", media_type="text/javascript")

    @app.get("/demod-export.mjs", include_in_schema=False)
    def demod_export_module() -> FileResponse:
        return FileResponse(WEB_DIR / "demod-export.mjs", media_type="text/javascript")

    @app.get("/signal-overlays.mjs", include_in_schema=False)
    def signal_overlays_module() -> FileResponse:
        return FileResponse(WEB_DIR / "signal-overlays.mjs", media_type="text/javascript")

    @app.get("/style.css", include_in_schema=False)
    def stylesheet() -> FileResponse:
        return FileResponse(WEB_DIR / "style.css", media_type="text/css")

    return app
