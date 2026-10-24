"""HTTP API and small browser workbench for WAV replay and manual signal labels."""

from __future__ import annotations

import os
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import tempfile
import time
import wave
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, model_validator

from .dataset import CaptureStore, SignalLabel, LabelRevisionConflict
from .classifier import FEATURE_VERSION, extract_features, load_model, propose
from .cv_detector import (
    FFT_SIZE,
    MAX_FREQUENCY_HZ,
    OVERLAP_DEDUP_IOU,
    PREPROCESSING_VERSION,
    WINDOW_SECONDS,
    WINDOW_STRIDE_SECONDS,
    YoloSignalDetector,
    proposal_id as cv_proposal_id,
)
from .cv_embeddings import load_embedding_index, read_audio_range, region_image
from .laya import LayaError, classify as classify_with_laya, configured as laya_configured, load_reference_examples
from .demod import demodulate_2fsk, demodulate_bpsk, demodulate_mfsk, demodulate_ook, demodulate_qpsk
from .detection import detect_tracks
from .fldigi import FldigiClient
from .fldigi_replay import IsolatedFldigiReplay
from .live_audio import LiveCaptureManager
from .live_spectrum import spectrum_packet
from .modes import MODE_REGISTRY_VERSION, find_modes, resolve_mode_label
from .receive_products import ReceiveProductStore
from .spectrum import waterfall, waterfall_samples
from .synthetic_dataset import MAX_MANIFEST_BYTES, import_synthetic_wav
from .tracking import LiveTrackRegistry
from .trials import FldigiTrialRunner, MODULATION_TRIAL_ORDER, VALIDATED_TRIAL_MODES
from .wav import WavSource


MAX_UPLOAD_BYTES = 100 * 1024 * 1024
WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_MFSK_RATE_SEARCH_WORK = 200_000_000


def default_track_fft_size(sample_rate: int) -> int:
    """Choose a track-analysis window near 42 ms, capped by the API FFT limit."""
    target = max(32, int(np.ceil(sample_rate / 24)))
    return min(8192, 1 << int(np.ceil(np.log2(target))))


class DetectorSourceWindow(BaseModel):
    sample_start: int = Field(ge=0)
    sample_count: int = Field(gt=0)


class DetectorProposalEvidence(BaseModel):
    detector: Literal["yolo"]
    proposal_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}:[0-9a-f]{32}$")
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_version: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0.0, le=1.0)
    box_xyxy_normalized: tuple[float, float, float, float]
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    preprocessing_version: str = Field(min_length=1, max_length=80)
    window_sample_start: int = Field(ge=0)
    window_sample_count: int = Field(gt=0)
    sample_rate: int = Field(gt=0)
    supporting_windows: list[DetectorSourceWindow] = Field(default_factory=list, max_length=4096)

    @model_validator(mode="after")
    def validate_normalized_box(self) -> "DetectorProposalEvidence":
        x1, y1, x2, y2 = self.box_xyxy_normalized
        if (
            self.model_version != f"sha256:{self.model_sha256}"
            or (self.proposal_id is not None and not self.proposal_id.startswith(f"{self.model_sha256}:"))
            or not all(0.0 <= value <= 1.0 for value in self.box_xyxy_normalized)
            or x2 <= x1 or y2 <= y1
        ):
            raise ValueError("detector evidence must contain an ordered normalized box and matching model version")
        return self


class CVProposalDecisionCreate(BaseModel):
    proposal_id: str = Field(min_length=1, max_length=110)
    decision: Literal["dismissed", "restored"]
    sample_start: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    frequency_low_hz: float = Field(ge=0, le=3000)
    frequency_high_hz: float = Field(gt=0, le=3000)
    confidence: float = Field(ge=0, le=1)
    detector_evidence: DetectorProposalEvidence
    note: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def validate_bounds(self) -> "CVProposalDecisionCreate":
        if self.frequency_low_hz >= self.frequency_high_hz:
            raise ValueError("CV proposal frequency bounds must be ordered")
        if self.detector_evidence.proposal_id not in (None, self.proposal_id):
            raise ValueError("CV proposal evidence id does not match its decision")
        return self


class ClassifierProposalRequest(BaseModel):
    sample_start: int = Field(ge=0)
    sample_count: int = Field(ge=1024, le=1_000_000)
    selection_sample_count: int = Field(gt=0)
    frequency_low_hz: float | None = Field(default=None, ge=0, le=96000)
    frequency_high_hz: float | None = Field(default=None, gt=0, le=96000)

    @model_validator(mode="after")
    def validate_frequency_bounds(self) -> "ClassifierProposalRequest":
        if (self.frequency_low_hz is None) != (self.frequency_high_hz is None):
            raise ValueError("classifier frequency bounds must be supplied together")
        if self.frequency_low_hz is not None and self.frequency_low_hz >= self.frequency_high_hz:
            raise ValueError("classifier frequency bounds must be ordered")
        return self


class CVSimilarRegionsRequest(BaseModel):
    sample_start: int = Field(ge=0)
    sample_count: int = Field(gt=0, le=10_000_000)
    frequency_low_hz: float = Field(ge=0, le=3000)
    frequency_high_hz: float = Field(gt=0, le=3000)
    exclude_region_id: str | None = Field(default=None, max_length=80)
    limit: int = Field(default=8, ge=1, le=20)


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
    mode_id: str | None = Field(default=None, max_length=80)
    origin: Literal["manual", "detector_proposal", "user_added_missed_detection"] = "manual"
    detector_evidence: DetectorProposalEvidence | None = None
    classifier_proposal: ClassifierProposalRequest | None = None
    signal_presence_confirmed: bool = False


class LabelRevisionCreate(BaseModel):
    expected_revision: int = Field(ge=1)
    family: str | None = Field(default=None, max_length=200)
    method: str | None = Field(default=None, max_length=200)
    protocol: str | None = Field(default=None, max_length=200)
    variant: str | None = Field(default=None, max_length=200)
    mode_id: str | None = Field(default=None, max_length=80)
    notes: str = Field(default="", max_length=20000)
    classifier_proposal: ClassifierProposalRequest | None = None


class AnnotationFeedbackCreate(BaseModel):
    expected_revision: int = Field(ge=1)
    expected_geometry_revision: int = Field(ge=1)
    action: Literal["correct", "incorrect_identity", "not_signal"]
    corrected_mode_id: str | None = Field(default=None, max_length=80)
    corrected_family: str | None = Field(default=None, max_length=200)
    corrected_method: str | None = Field(default=None, max_length=200)
    corrected_protocol: str | None = Field(default=None, max_length=200)
    corrected_variant: str | None = Field(default=None, max_length=200)
    note: str = Field(default="", max_length=2000)


class GeometryRevisionCreate(BaseModel):
    expected_revision: int = Field(ge=1)
    sample_start: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    frequency_low_hz: float | None = Field(default=None, ge=0)
    frequency_high_hz: float | None = Field(default=None, ge=0)


class RegionBounds(BaseModel):
    sample_start: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    frequency_low_hz: float | None = Field(default=None, ge=0)
    frequency_high_hz: float | None = Field(default=None, ge=0)


class RegionSplitCreate(BaseModel):
    expected_geometry_revision: int = Field(ge=1)
    children: list[RegionBounds] = Field(min_length=2, max_length=8)


class RegionMergeCreate(BaseModel):
    source_region_ids: list[str] = Field(min_length=2, max_length=8)
    expected_geometry_revisions: dict[str, int]


class CoverageMask(BaseModel):
    sample_start: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    frequency_low_hz: float | None = Field(default=None, ge=0)
    frequency_high_hz: float | None = Field(default=None, ge=0)
    reason: str = Field(default="", max_length=200)


class SceneCoverageFields(BaseModel):
    sample_start: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    frequency_low_hz: float | None = Field(default=None, ge=0)
    frequency_high_hz: float | None = Field(default=None, ge=0)
    task: Literal["detection", "modulation", "protocol", "decoder"]
    taxonomy_scope: list[str] = Field(default_factory=list, max_length=32)
    state: Literal["complete", "partial"]
    ignore_regions: list[CoverageMask] = Field(default_factory=list, max_length=64)
    uncertain_regions: list[CoverageMask] = Field(default_factory=list, max_length=64)
    notes: str = Field(default="", max_length=2000)


class SceneCoverageCreate(SceneCoverageFields):
    pass


class SceneCoverageRevision(SceneCoverageFields):
    expected_revision: int = Field(ge=1)


class ModeEvidenceRegion(BaseModel):
    capture_id: str = Field(min_length=1, max_length=80)
    region_id: str = Field(min_length=1, max_length=80)
    evidence_type: Literal["on_air", "controlled_fixture", "independent_decoder"] = "on_air"
    observations: str = Field(min_length=40, max_length=2000)


class ModeProposalCreate(BaseModel):
    id: str = Field(pattern="^[a-z0-9]+(?:-[a-z0-9]+)*$", min_length=2, max_length=80)
    display_name: str = Field(min_length=1, max_length=120)
    family: str = Field(min_length=1, max_length=120)
    method: str = Field(min_length=1, max_length=120)
    protocol: str = Field(min_length=1, max_length=120)
    variant: str = Field(min_length=1, max_length=120)
    aliases: list[str] = Field(min_length=1, max_length=10)
    rationale: str = Field(min_length=80, max_length=5000)
    distinguishing_evidence: str = Field(min_length=80, max_length=5000)
    evidence_regions: list[ModeEvidenceRegion] = Field(min_length=2, max_length=8)


class ModeProposalDecision(BaseModel):
    decision: Literal["accepted", "rejected"]
    note: str = Field(min_length=40, max_length=2000)


class BitAnnotationCreate(BaseModel):
    mode: str = Field(min_length=1, max_length=40)
    sample_start: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    settings: dict[str, Any]
    hypothesis: str = Field(min_length=1, max_length=80)
    bits: str = Field(min_length=1, max_length=1_000_000, pattern="^[01]+$")
    bit_start: int = Field(ge=0)
    bit_count: int = Field(gt=0)
    label: str = Field(min_length=1, max_length=200)
    color: str = Field(pattern="^#[0-9A-Fa-f]{6}$")


class ProtocolStencilCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    field: str = Field(min_length=1, max_length=100)
    pattern: str = Field(min_length=4, max_length=512, pattern="^[01?]+$")
    protocols: list[str] = Field(default_factory=list, max_length=20)
    source_capture_id: str | None = None


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
    sample_start: int | None = Field(default=None, ge=0)
    sample_count: int | None = Field(default=None, gt=0, le=192_000 * 600)


class FldigiReplayRequest(BaseModel):
    mode: str = Field(min_length=1, max_length=80)
    sample_start: int = Field(ge=0)
    sample_count: int = Field(gt=0, le=192_000 * 600)
    track_id: str | None = Field(default=None, max_length=128)
    carrier_hz: float | None = Field(default=None, ge=0, le=20000)
    region_id: str | None = Field(default=None, max_length=80)
    detector_evidence: DetectorProposalEvidence | None = None


class FldigiTrackTrialRequest(BaseModel):
    track_id: str = Field(min_length=1, max_length=128)
    modulation_hypothesis: str = Field(default="unknown", max_length=80)
    sample_start: int = Field(ge=0)
    sample_count: int = Field(gt=0, le=192_000 * 10)
    carrier_hz: float | None = Field(default=None, ge=0, le=20000)


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
    frequency_low_hz: float | None = Field(default=None, ge=0, le=96000)
    frequency_high_hz: float | None = Field(default=None, gt=0, le=96000)

    @model_validator(mode="after")
    def validate_frequency_bounds(self) -> "ClassifyRequest":
        if (self.frequency_low_hz is None) != (self.frequency_high_hz is None):
            raise ValueError("classifier frequency bounds must be supplied together")
        if self.frequency_low_hz is not None and self.frequency_low_hz >= self.frequency_high_hz:
            raise ValueError("classifier frequency bounds must be ordered")
        return self


class LayaClassifyRequest(BaseModel):
    sample_start: int = Field(ge=0)
    sample_count: int = Field(ge=1024, le=1_000_000)
    frequency_low_hz: float | None = Field(default=None, ge=0, le=96000)
    frequency_high_hz: float | None = Field(default=None, ge=0, le=96000)


def create_app(
    data_dir: str | Path | None = None,
    audio_backend: Any | None = None,
    fldigi_client: Any | None = None,
    fldigi_replay_backend: Any | None = None,
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
    fldigi_replay = fldigi_replay_backend or IsolatedFldigiReplay()
    decode_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sig-id-fldigi-replay")
    decode_job_lock = Lock()
    configured_model = classifier_model_path or os.environ.get("SIG_ID_MODEL_PATH")
    classifier = None
    classifier_error = None
    if configured_model:
        try:
            model_metadata, model_parameters = load_model(configured_model)
            classifier = {"metadata": model_metadata, "parameters": model_parameters}
        except Exception as exc:
            classifier_error = str(exc)

    configured_cv_model = os.environ.get("SIG_ID_CV_MODEL_PATH")
    cv_detector = None
    cv_detector_error = None
    if configured_cv_model:
        try:
            cv_detector = YoloSignalDetector(configured_cv_model)
        except Exception as exc:
            cv_detector_error = str(exc)

    configured_cv_index = os.environ.get("SIG_ID_CV_EMBEDDING_INDEX")
    cv_embedding_vectors = None
    cv_embedding_entries = None
    cv_embedding_error = None
    if configured_cv_index:
        if cv_detector is None:
            cv_embedding_error = "configure a valid SIG_ID_CV_MODEL_PATH before loading its embedding index"
        else:
            try:
                cv_embedding_vectors, cv_embedding_entries = load_embedding_index(
                    configured_cv_index, model_sha256=cv_detector.model_sha256,
                )
            except Exception as exc:
                cv_embedding_error = str(exc)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            yield
        finally:
            application.state.fldigi_trials.cancel()
            close_replay = getattr(fldigi_replay, "close", None)
            if callable(close_replay):
                close_replay()
            decode_executor.shutdown(wait=True, cancel_futures=True)
            try:
                stopped = await asyncio.to_thread(live_capture.shutdown)
                if stopped and stopped.get("capture"):
                    store.associate_decoder_runs(stopped["session_id"], stopped["capture"]["id"])
            except Exception:
                logging.getLogger(__name__).exception("Could not finalize live capture during shutdown; inspect retained audio files")

    app = FastAPI(title="Signal ID Workbench", version="0.1.0", lifespan=lifespan)
    app.state.data_dir = root
    app.state.capture_dir = capture_dir
    app.state.capture_store = store
    app.state.live_capture = live_capture
    app.state.receive_products = receive_products
    app.state.fldigi = fldigi
    app.state.fldigi_replay = fldigi_replay
    app.state.classifier = classifier
    app.state.classifier_error = classifier_error
    app.state.cv_detector = cv_detector
    app.state.cv_detector_error = cv_detector_error
    app.state.cv_model_path = configured_cv_model
    app.state.cv_embedding_vectors = cv_embedding_vectors
    app.state.cv_embedding_entries = cv_embedding_entries
    app.state.cv_embedding_index_path = configured_cv_index
    app.state.cv_embedding_error = cv_embedding_error
    app.state.laya_references = load_reference_examples()
    app.state.live_track_registry = None
    app.state.fldigi_trials = FldigiTrialRunner(fldigi, run_store=store)

    def run_region_replay(
        run_id: str, settings: FldigiReplayRequest, sample_rate: int,
        sample_reader: Any,
    ) -> None:
        temporary_audio: Path | None = None
        try:
            samples = sample_reader()
            if len(samples) != settings.sample_count:
                raise ValueError("selected sample range could not be read in full")
            temporary = tempfile.NamedTemporaryFile(prefix="sig-id-region-", suffix=".wav", delete=False)
            temporary_audio = Path(temporary.name)
            temporary.close()
            pcm = np.clip(np.rint(np.asarray(samples, dtype=np.float64) * 32767.0), -32768, 32767).astype("<i2")
            with wave.open(str(temporary_audio), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(sample_rate)
                output.writeframes(pcm.tobytes())
            app.state.capture_store.update_decoder_run(run_id, {
                "attempted_modes": [settings.mode], "current_mode": settings.mode,
                "output_offset": 0, "output_end": None, "text": "",
                "status": "running", "note": "Preparing the selected sample window for isolated receive replay.",
            })
            result = fldigi_replay.decode(
                temporary_audio, mode=settings.mode, carrier_hz=settings.carrier_hz,
            )
            text = str(result.get("text", ""))
            app.state.capture_store.update_decoder_run(run_id, {
                "attempted_modes": [settings.mode], "current_mode": settings.mode,
                "output_offset": 0, "output_end": len(text.encode("utf-8")), "text": text,
                "status": result.get("status", "completed"),
                "note": f"{result.get('note', '')} Carrier setting: {settings.carrier_hz if settings.carrier_hz is not None else 'unset'} Hz.",
            })
        except Exception as exc:
            logging.getLogger(__name__).exception("Isolated Fldigi region replay failed")
            try:
                app.state.capture_store.update_decoder_run(run_id, {
                    "attempted_modes": [settings.mode], "current_mode": settings.mode,
                    "output_offset": 0, "output_end": None, "text": "",
                    "status": "failed", "note": str(exc)[:2000],
                })
            except Exception:
                logging.getLogger(__name__).exception("Could not persist Fldigi replay failure")
        finally:
            if temporary_audio is not None:
                temporary_audio.unlink(missing_ok=True)
            decode_job_lock.release()

    def run_saved_replay(
        run_id: str, capture: dict[str, Any], settings: FldigiReplayRequest,
    ) -> None:
        def read_samples() -> list[float]:
            selected_start = settings.sample_start
            selected_end = selected_start + settings.sample_count
            samples: list[float] = []
            for block in WavSource(capture["audio_path"]).blocks():
                block_end = block.first_sample + block.sample_count
                if block_end <= selected_start:
                    continue
                if block.first_sample >= selected_end:
                    break
                left = max(0, selected_start - block.first_sample)
                right = min(block.sample_count, selected_end - block.first_sample)
                samples.extend(block.samples[left:right])
            return samples

        run_region_replay(run_id, settings, int(capture["sample_rate"]), read_samples)

    trial_attempted_tracks: dict[str, float] = {}
    trial_start_times: deque[float] = deque()
    trial_admission_lock = Lock()

    def run_live_mode_sweep(
        run_id: str, samples: np.ndarray, sample_rate: int,
        settings: FldigiTrackTrialRequest, modes: list[str],
    ) -> None:
        temporary_audio: Path | None = None
        attempted: list[str] = []
        try:
            temporary = tempfile.NamedTemporaryFile(prefix="sig-id-track-trial-", suffix=".wav", delete=False)
            temporary_audio = Path(temporary.name)
            temporary.close()
            pcm = np.clip(np.rint(samples * 32767.0), -32768, 32767).astype("<i2")
            with wave.open(str(temporary_audio), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(sample_rate)
                output.writeframes(pcm.tobytes())

            for mode in modes:
                attempted.append(mode)
                app.state.capture_store.update_decoder_run(run_id, {
                    "attempted_modes": attempted, "current_mode": mode,
                    "output_offset": 0, "output_end": None, "text": "",
                    "status": "running",
                    "note": f"Replaying the exact live track window in isolated {mode} receive mode.",
                })
                result = fldigi_replay.decode(
                    temporary_audio, mode=mode, carrier_hz=settings.carrier_hz,
                )
                text = str(result.get("text", ""))
                if text:
                    app.state.capture_store.update_decoder_run(run_id, {
                        "attempted_modes": attempted, "current_mode": mode,
                        "output_offset": 0, "output_end": len(text.encode("utf-8")), "text": text,
                        "status": "tentative_output",
                        "note": f"{mode} produced text from the selected track window. Review it; this is not a protocol identification.",
                    })
                    return
            app.state.capture_store.update_decoder_run(run_id, {
                "attempted_modes": attempted, "current_mode": attempted[-1] if attempted else None,
                "output_offset": 0, "output_end": 0, "text": "",
                "status": "exhausted",
                "note": "No text was recovered by the bounded isolated tested-mode sweep.",
            })
        except Exception as exc:
            logging.getLogger(__name__).exception("Isolated live track mode sweep failed")
            try:
                app.state.capture_store.update_decoder_run(run_id, {
                    "attempted_modes": attempted, "current_mode": attempted[-1] if attempted else None,
                    "output_offset": 0, "output_end": None, "text": "",
                    "status": "failed", "note": str(exc)[:2000],
                })
            except Exception:
                logging.getLogger(__name__).exception("Could not persist isolated mode-sweep failure")
        finally:
            if temporary_audio is not None:
                temporary_audio.unlink(missing_ok=True)
            decode_job_lock.release()

    def queue_live_mode_sweep(
        session_id: str, settings: FldigiTrackTrialRequest,
    ) -> dict[str, Any]:
        status = live_capture.status()
        if not status.get("active") or status.get("session_id") != session_id:
            raise ValueError("live capture session is no longer active")
        if not settings.track_id.startswith(f"{session_id}:"):
            raise ValueError("track_id does not belong to the live capture session")
        snapshot = live_capture.analysis_audio()
        if snapshot is None or snapshot.get("session_id") != session_id:
            raise ValueError("live audio window is unavailable")
        window_start = int(snapshot["sample_offset"])
        window_end = window_start + int(snapshot["window_sample_count"])
        requested_end = settings.sample_start + settings.sample_count
        if settings.sample_start < window_start or requested_end > window_end:
            raise ValueError(f"selected samples are no longer in the live buffer ({window_start}–{window_end})")
        try:
            catalog = set(fldigi.mode_names())
        except Exception as exc:
            raise RuntimeError(f"Fldigi mode catalog unavailable: {exc}") from exc
        proposed = MODULATION_TRIAL_ORDER.get(settings.modulation_hypothesis, VALIDATED_TRIAL_MODES)
        modes = [mode for mode in proposed if mode in catalog]
        if not modes:
            raise ValueError("no loopback-tested trial modes are present in the Fldigi catalog")

        channels = int(snapshot["channels"])
        pcm = np.frombuffer(snapshot["pcm"], dtype="<i2").reshape(-1, channels)
        left, right = settings.sample_start - window_start, requested_end - window_start
        samples = pcm[left:right].astype(np.float64).mean(axis=1) / 32768.0
        now = time.monotonic()
        with trial_admission_lock:
            while trial_start_times and now - trial_start_times[0] >= 60.0:
                trial_start_times.popleft()
            if settings.track_id in trial_attempted_tracks:
                raise ValueError("this live track has already had an isolated mode sweep")
            if len(trial_start_times) >= 6:
                raise ValueError("isolated live mode-sweep rate limit reached")
            if not decode_job_lock.acquire(blocking=False):
                raise ValueError("an isolated Fldigi replay is already running")
            trial_attempted_tracks[settings.track_id] = now
            trial_start_times.append(now)
            if len(trial_attempted_tracks) > 1024:
                oldest = next(iter(trial_attempted_tracks))
                del trial_attempted_tracks[oldest]

        try:
            run_id = store.create_decoder_run(
                session_id=session_id, track_id=settings.track_id,
                sample_start=settings.sample_start, sample_count=settings.sample_count,
                sample_rate=int(snapshot["sample_rate"]),
                modulation_hypothesis=settings.modulation_hypothesis,
                backend="fldigi-isolated-track-sweep", modes=modes, attempted_modes=[],
                current_mode=modes[0], output_offset=0, status="queued",
                note="Waiting to run the bounded isolated tested-mode sweep on this live track window.",
                carrier_hz=settings.carrier_hz,
                output_scope="isolated_live_mode_sweep", association="exact_live_track_window",
            )
            decode_executor.submit(
                run_live_mode_sweep, run_id, samples, int(snapshot["sample_rate"]), settings, modes,
            )
            return {
                "run_id": run_id, "session_id": session_id, "track_id": settings.track_id,
                "status": "running", "modes": modes, "current_mode": modes[0],
                "output_offset": 0, "output_scope": "isolated_live_mode_sweep",
                "note": "Isolated track-window mode sweep queued.",
            }
        except Exception:
            decode_job_lock.release()
            with trial_admission_lock:
                trial_attempted_tracks.pop(settings.track_id, None)
                try:
                    trial_start_times.remove(now)
                except ValueError:
                    pass
            raise

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
            "feature_frequency_scope": metadata.get("feature_frequency_scope", "full_audio_passband"),
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

    @app.get("/api/cv/status")
    def cv_status() -> dict[str, Any]:
        detector = app.state.cv_detector
        test_report = detector.held_out_test if detector else None
        test_metrics = (test_report or {}).get("metrics", {}) if isinstance(test_report, dict) else {}
        iou_metrics = test_metrics.get("iou_0.50", {}) if isinstance(test_metrics, dict) else {}
        held_out_summary = None
        if isinstance(iou_metrics, dict) and all(
            isinstance(iou_metrics.get(key), int) and not isinstance(iou_metrics.get(key), bool)
            for key in ("true_positive", "false_positive", "false_negative")
        ):
            held_out_summary = {
                "scope": test_report.get("scope"),
                "windows": test_report.get("reviewed_test_windows"),
                "true_positive": iou_metrics["true_positive"],
                "false_positive": iou_metrics["false_positive"],
                "false_negative": iou_metrics["false_negative"],
                "false_detections_per_audio_hour": test_report.get("false_detections_per_audio_hour"),
            }
        return {
            "available": detector is not None,
            "configured": bool(app.state.cv_model_path),
            "model_path": app.state.cv_model_path,
            "error": app.state.cv_detector_error,
            "compute_backend": detector.compute_backend if detector else None,
            "model_version": detector.model_version if detector else None,
            "confidence_threshold": detector.confidence_threshold if detector else None,
            "held_out_test": held_out_summary,
            "training_scope": detector.training_scope if detector else None,
            "training_source_kinds": detector.training_source_kinds if detector else None,
            "held_out_test_source_kinds": detector.held_out_test_source_kinds if detector else None,
            "similarity_search_available": app.state.cv_embedding_vectors is not None,
            "similarity_search_error": app.state.cv_embedding_error,
            "preprocessing_version": PREPROCESSING_VERSION,
            "window_seconds": WINDOW_SECONDS,
            "fft_size": FFT_SIZE,
            "max_frequency_hz": MAX_FREQUENCY_HZ,
            "class_agnostic": True,
        }

    @app.get("/api/laya/status")
    def laya_status() -> dict[str, Any]:
        return {
            "available": laya_configured(),
            "reference_examples": len(app.state.laya_references),
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

    @app.get("/api/live/spectrum")
    def live_spectrum(
        after_sample: int = Query(default=-1, ge=-1),
        session_id: str | None = Query(default=None, max_length=128),
        fft_size: int = Query(default=4096, ge=32, le=8192),
        hop_size: int = Query(default=512, ge=1, le=8192),
    ) -> Response:
        snapshot = live_capture.analysis_audio()
        if snapshot is None or not snapshot["active"]:
            raise HTTPException(status_code=409, detail="live capture is not active")
        try:
            packet = spectrum_packet(snapshot, after_sample=after_sample, session_id=session_id,
                                     fft_size=fft_size, hop_size=hop_size)
            return Response(packet, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/live/analysis")
    def live_analysis(
        fft_size: int = Query(default=4096, ge=32, le=8192),
        hop_size: int = Query(default=512, ge=1, le=8192),
        max_rows: int = Query(default=640, ge=1, le=8192),
        threshold_db: float = Query(default=18.0, ge=1.0, le=60.0),
        auto_trial: bool = Query(default=False),
        include_spectrum: bool = Query(default=True),
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
                        hypothesis = {
                            "FSK": "FSK-like",
                            "MFSK": "MFSK-like",
                            "PSK": "single-tone / unknown",
                        }.get(proposal["family"], proposal["family"])
                        automatic_trial = queue_live_mode_sweep(
                            snapshot["session_id"],
                            FldigiTrackTrialRequest(
                            track_id=track["track_id"],
                                modulation_hypothesis=hypothesis,
                                sample_start=int(proposal.get("sample_start", track["sample_start"])),
                                sample_count=int(proposal.get("sample_count", track["sample_count"])),
                                carrier_hz=float(track["center_hz"]),
                            ),
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
                "spectrum": spectrum if include_spectrum else None,
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
            store.associate_decoder_runs(result["session_id"], result["capture"]["id"])
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
            status = live_capture.status()
            live = bool(status.get("active"))
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
                session_id=status.get("session_id"),
                sample_start=data.sample_start,
                sample_count=data.sample_count,
                sample_rate=status.get("sample_rate"),
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

    @app.get("/api/captures/{capture_id}/decode-runs")
    def capture_decode_runs(capture_id: str) -> list[dict[str, Any]]:
        try:
            return store.decoder_runs(capture_id=capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc

    @app.get("/api/live/sessions/{session_id}/decode-runs")
    def session_decode_runs(session_id: str) -> list[dict[str, Any]]:
        return store.decoder_runs(session_id=session_id)

    @app.post("/api/live/sessions/{session_id}/decode", status_code=202)
    def replay_live_region(session_id: str, settings: FldigiReplayRequest) -> dict[str, Any]:
        if settings.region_id is not None:
            raise HTTPException(status_code=422, detail="saved region links are only valid for saved-capture replays")
        status = live_capture.status()
        if not status.get("active") or status.get("session_id") != session_id:
            raise HTTPException(status_code=409, detail="live capture session is no longer active")
        snapshot = live_capture.analysis_audio()
        if snapshot is None or snapshot.get("session_id") != session_id:
            raise HTTPException(status_code=409, detail="live audio window is unavailable")
        window_start = int(snapshot["sample_offset"])
        window_end = window_start + int(snapshot["window_sample_count"])
        requested_end = settings.sample_start + settings.sample_count
        if settings.sample_start < window_start or requested_end > window_end:
            raise HTTPException(
                status_code=422,
                detail=f"selected samples are no longer in the live buffer ({window_start}–{window_end})",
            )
        try:
            modes = fldigi.mode_names()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Fldigi mode catalog unavailable: {exc}") from exc
        if settings.mode not in modes:
            raise HTTPException(status_code=422, detail="selected mode is not available in the Fldigi catalog")
        channels = int(snapshot["channels"])
        pcm = np.frombuffer(snapshot["pcm"], dtype="<i2").reshape(-1, channels)
        left = settings.sample_start - window_start
        right = requested_end - window_start
        samples = pcm[left:right].astype(np.float64).mean(axis=1) / 32768.0
        track_id = settings.track_id or f"{session_id}:manual:{settings.sample_start}:{settings.sample_count}"
        if not track_id.startswith(f"{session_id}:"):
            raise HTTPException(status_code=422, detail="track_id does not belong to the active live session")
        if not decode_job_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="an isolated Fldigi replay is already running")
        try:
            run_id = store.create_decoder_run(
                session_id=session_id, track_id=track_id,
                sample_start=settings.sample_start, sample_count=settings.sample_count,
                sample_rate=int(snapshot["sample_rate"]), modulation_hypothesis="unknown",
                backend="fldigi-isolated-replay", modes=[settings.mode], attempted_modes=[],
                current_mode=settings.mode, output_offset=0, status="queued",
                note="Waiting to replay the selected live track window in an isolated Fldigi instance.",
                carrier_hz=settings.carrier_hz,
                output_scope="isolated_live_window_replay", association="exact_live_track_window",
            )
            decode_executor.submit(
                run_region_replay, run_id, settings, int(snapshot["sample_rate"]),
                lambda: samples.tolist(),
            )
            return {"run_id": run_id, "status": "queued", "mode": settings.mode, "carrier_hz": settings.carrier_hz}
        except Exception as exc:
            decode_job_lock.release()
            raise HTTPException(status_code=503, detail=f"could not queue live track replay: {exc}") from exc

    @app.post("/api/live/sessions/{session_id}/decode-trials", status_code=202)
    def start_isolated_live_trial(session_id: str, settings: FldigiTrackTrialRequest) -> dict[str, Any]:
        try:
            return queue_live_mode_sweep(session_id, settings)
        except ValueError as exc:
            reason = str(exc)
            conflict = any(token in reason for token in ("no longer active", "already had", "already running", "rate limit"))
            raise HTTPException(status_code=409 if conflict else 422, detail=reason) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/captures/{capture_id}/decode", status_code=202)
    def replay_capture_region(capture_id: str, settings: FldigiReplayRequest) -> dict[str, Any]:
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        if settings.sample_start + settings.sample_count > int(capture["sample_count"]):
            raise HTTPException(status_code=422, detail="selected region exceeds the saved capture")
        linked_region = None
        if settings.region_id:
            linked_region = next((item for item in store.regions(capture_id) if item["id"] == settings.region_id), None)
            if linked_region is None or linked_region["lineage_state"] != "active":
                raise HTTPException(status_code=422, detail="decoder region must be an active region in this capture")
            if (settings.sample_start, settings.sample_count) != (linked_region["sample_start"], linked_region["sample_count"]):
                raise HTTPException(status_code=422, detail="decoder sample bounds must match the linked region")
        detector_evidence = settings.detector_evidence
        if detector_evidence is not None:
            window_start = detector_evidence.window_sample_start
            window_end = window_start + detector_evidence.window_sample_count
            if detector_evidence.sample_rate != int(capture["sample_rate"]):
                raise HTTPException(status_code=422, detail="detector evidence sample rate does not match the capture")
            if window_end > int(capture["sample_count"]):
                raise HTTPException(status_code=422, detail="detector evidence window exceeds the capture sample range")
            if settings.sample_start < window_start or settings.sample_start + settings.sample_count > window_end:
                raise HTTPException(status_code=422, detail="decoder range must remain inside its detector proposal window")
        if detector_evidence is not None and linked_region is not None:
            saved_region_evidence = linked_region["evidence"].get("detector_proposal") or {}
            if (
                saved_region_evidence.get("proposal_id")
                and detector_evidence.proposal_id
                and saved_region_evidence["proposal_id"] != detector_evidence.proposal_id
            ):
                raise HTTPException(status_code=422, detail="detector evidence does not match the linked region proposal")
        saved_detector_evidence = None
        if detector_evidence is not None:
            saved_detector_evidence = detector_evidence.model_dump()
        elif linked_region is not None:
            saved_detector_evidence = linked_region["evidence"].get("detector_proposal")
        try:
            modes = fldigi.mode_names()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Fldigi mode catalog unavailable: {exc}") from exc
        if settings.mode not in modes:
            raise HTTPException(status_code=422, detail="selected mode is not available in the Fldigi catalog")
        if not decode_job_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="an isolated Fldigi replay is already running")
        track_id = settings.track_id or f"capture:{capture_id}:samples:{settings.sample_start}:{settings.sample_count}"
        try:
            run_id = store.create_decoder_run(
                session_id=f"capture:{capture_id}", capture_id=capture_id,
                track_id=track_id, sample_start=settings.sample_start,
                sample_count=settings.sample_count, sample_rate=int(capture["sample_rate"]),
                modulation_hypothesis="unknown", backend="fldigi-isolated-replay",
                modes=[settings.mode], attempted_modes=[], current_mode=settings.mode,
                output_offset=0, status="queued",
                note="Waiting to replay the selected saved-capture region.",
                carrier_hz=settings.carrier_hz,
                region_id=settings.region_id,
                classifier_evidence=(linked_region["evidence"].get("classifier_proposal") if linked_region else None),
                detector_evidence=saved_detector_evidence,
                output_scope="isolated_region_replay", association="exact_capture_window",
            )
            decode_executor.submit(run_saved_replay, run_id, capture, settings)
            return {"run_id": run_id, "status": "queued", "mode": settings.mode, "carrier_hz": settings.carrier_hz}
        except Exception as exc:
            decode_job_lock.release()
            raise HTTPException(status_code=503, detail=f"could not queue the selected-region replay: {exc}") from exc

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
    def audio(capture_id: str, request: Request) -> Response:
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        path = Path(capture["audio_path"])
        file_size = path.stat().st_size
        range_header = request.headers.get("range")
        if range_header is None:
            return FileResponse(
                path, media_type="audio/wav", filename=f"{capture_id}.wav",
                headers={"Accept-Ranges": "bytes"},
            )
        unit, separator, requested = range_header.partition("=")
        start_text, dash, end_text = requested.partition("-")
        invalid = unit.strip().lower() != "bytes" or not separator or not dash or "," in requested
        try:
            if invalid or (not start_text and not end_text):
                raise ValueError("malformed range")
            if not start_text:
                suffix_length = int(end_text)
                if suffix_length <= 0:
                    raise ValueError("empty suffix range")
                start = max(0, file_size - suffix_length)
                end = file_size - 1
            else:
                start = int(start_text)
                end = int(end_text) if end_text else file_size - 1
                if start < 0 or end < start:
                    raise ValueError("invalid byte range")
                end = min(end, file_size - 1)
            if start >= file_size:
                raise ValueError("range begins after end of file")
        except ValueError:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}", "Accept-Ranges": "bytes"})
        length = end - start + 1
        with path.open("rb") as stream:
            stream.seek(start)
            content = stream.read(length)
        return Response(
            content=content,
            status_code=206,
            media_type="audio/wav",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(len(content)),
            },
        )

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

    @app.post("/api/captures/{capture_id}/cv-detections")
    def detect_capture_signals(
        capture_id: str,
        confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    ) -> dict[str, Any]:
        detector = app.state.cv_detector
        if detector is None:
            detail = app.state.cv_detector_error or "CV detector is disabled; configure SIG_ID_CV_MODEL_PATH and install the optional cv extra"
            raise HTTPException(status_code=503, detail=detail)
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        try:
            detections = detector.detect_capture(capture["audio_path"], confidence=confidence)
        except Exception as exc:
            logging.getLogger(__name__).exception("Optional CV detection failed for capture %s", capture_id)
            raise HTTPException(status_code=503, detail=f"CV inference failed: {exc}") from exc
        for detection in detections:
            detection["proposal_id"] = cv_proposal_id(detection)
        return {
            "capture_id": capture_id,
            "model_sha256": detector.model_sha256,
            "model_version": detector.model_version,
            "preprocessing_version": PREPROCESSING_VERSION,
            "window_seconds": WINDOW_SECONDS,
            "window_stride_seconds": WINDOW_STRIDE_SECONDS,
            "overlap_deduplication_iou": OVERLAP_DEDUP_IOU,
            "confidence_threshold": detector.confidence_threshold,
            "detections": detections,
            "dismissed_proposals": store.cv_proposal_decisions(
                capture_id, model_sha256=detector.model_sha256,
            ),
        }

    @app.post("/api/captures/{capture_id}/cv-proposal-decisions", status_code=201)
    def record_cv_proposal_decision(capture_id: str, data: CVProposalDecisionCreate) -> dict[str, Any]:
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        evidence = data.detector_evidence.model_dump()
        if data.proposal_id != cv_proposal_id({
            "model_sha256": evidence["model_sha256"],
            "sample_start": data.sample_start,
            "sample_count": data.sample_count,
            "frequency_low_hz": data.frequency_low_hz,
            "frequency_high_hz": data.frequency_high_hz,
        }):
            raise HTTPException(status_code=422, detail="proposal id does not match its model and geometry")
        if evidence["sample_rate"] != capture["sample_rate"]:
            raise HTTPException(status_code=422, detail="proposal sample rate does not match the capture")
        try:
            return store.record_cv_proposal_decision(
                capture_id, proposal_id=data.proposal_id,
                model_sha256=evidence["model_sha256"], decision=data.decision,
                sample_start=data.sample_start, sample_count=data.sample_count,
                frequency_low_hz=data.frequency_low_hz, frequency_high_hz=data.frequency_high_hz,
                confidence=data.confidence, detector_evidence=evidence, note=data.note,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/captures/{capture_id}/cv-similar-regions")
    def find_similar_cv_regions(capture_id: str, settings: CVSimilarRegionsRequest) -> dict[str, Any]:
        detector = app.state.cv_detector
        vectors = app.state.cv_embedding_vectors
        entries = app.state.cv_embedding_entries
        if detector is None or vectors is None or entries is None:
            detail = app.state.cv_embedding_error or "CV similarity index is unavailable; configure SIG_ID_CV_EMBEDDING_INDEX"
            raise HTTPException(status_code=503, detail=detail)
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        end = settings.sample_start + settings.sample_count
        if end > int(capture["sample_count"]):
            raise HTTPException(status_code=422, detail="selected audio range exceeds the capture")
        if settings.frequency_low_hz >= settings.frequency_high_hz:
            raise HTTPException(status_code=422, detail="frequency_low_hz must be below frequency_high_hz")
        try:
            sample_rate, samples = read_audio_range(capture["audio_path"], settings.sample_start, settings.sample_count)
            image = region_image(samples, sample_rate, settings.frequency_low_hz, settings.frequency_high_hz)
            query_vector = detector.embed_image(image)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logging.getLogger(__name__).exception("CV similarity query failed for capture %s", capture_id)
            raise HTTPException(status_code=503, detail=f"CV similarity search failed: {exc}") from exc
        scores = vectors @ query_vector
        ordered = np.argsort(scores)[::-1]
        neighbors = []
        current_regions = store.current_cv_embedding_regions()
        stale_indices = set()
        for index, entry in enumerate(entries):
            current = current_regions.get((entry.get("capture_id"), entry.get("region_id")))
            if current is None or any(
                entry.get(key) != current.get(current_key)
                for key, current_key in (
                    ("capture_sha256", "capture_sha256"),
                    ("label_revision", "label_revision"),
                    ("geometry_revision", "geometry_revision"),
                    ("sample_start", "sample_start"),
                    ("sample_count", "sample_count"),
                    ("frequency_low_hz", "frequency_low_hz"),
                    ("frequency_high_hz", "frequency_high_hz"),
                    ("family", "family"), ("method", "method"),
                    ("protocol", "protocol"), ("variant", "variant"),
                )
            ):
                stale_indices.add(index)
        for index in ordered:
            if int(index) in stale_indices:
                continue
            entry = entries[int(index)]
            if entry["capture_id"] == capture_id and entry["region_id"] == settings.exclude_region_id:
                continue
            neighbors.append({
                **entry,
                "similarity": float(scores[index]),
                "label": " / ".join(value for value in (entry["family"], entry["method"], entry["protocol"], entry["variant"]) if value) or "Unknown signal",
            })
            if len(neighbors) >= settings.limit:
                break
        return {
            "capture_id": capture_id,
            "model_sha256": detector.model_sha256,
            "index_path": app.state.cv_embedding_index_path,
            "count": len(neighbors),
            "stale_index_entries_skipped": len(stale_indices),
            "neighbors": neighbors,
            "interpretation": "Nearest image embeddings are review candidates, not identifications.",
        }

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

    def classifier_proposal_for_range(
        capture_id: str, capture: dict[str, Any], sample_start: int,
        classified_sample_count: int, selection_sample_count: int,
        frequency_low_hz: float | None = None, frequency_high_hz: float | None = None,
    ) -> dict[str, Any]:
        if app.state.classifier is None:
            raise HTTPException(status_code=503, detail="no valid modulation model is configured")
        if sample_start >= int(capture["sample_count"]):
            raise HTTPException(status_code=422, detail="sample_start is outside the capture")
        expected_count = min(1_000_000, selection_sample_count, int(capture["sample_count"]) - sample_start)
        if classified_sample_count != expected_count:
            raise ValueError("classifier proposal range does not match the saved selection")
        if (frequency_low_hz is None) != (frequency_high_hz is None):
            raise ValueError("classifier frequency bounds must be supplied together")
        if frequency_low_hz is not None and (
            not 0 <= frequency_low_hz < frequency_high_hz <= int(capture["sample_rate"]) / 2
        ):
            raise ValueError("classifier frequency bounds are invalid for this capture")
        end = sample_start + classified_sample_count
        values: list[float] = []
        for block in WavSource(capture["audio_path"]).blocks():
            block_end = block.first_sample + block.sample_count
            if block_end <= sample_start:
                continue
            if block.first_sample >= end:
                break
            left = max(0, sample_start - block.first_sample)
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
            "feature_version": model["metadata"]["feature_version"],
            "capture_id": capture_id,
            "sample_start": sample_start,
            "sample_count": len(values),
            "selection_sample_count": selection_sample_count,
            "selection_frequency_range_hz": [frequency_low_hz, frequency_high_hz],
            "feature_frequency_scope": "full_audio_passband; selected frequency bounds are provenance only",
            "capture_sha256": capture["sha256"],
        }

    @app.post("/api/captures/{capture_id}/classify")
    def classify_capture_region(capture_id: str, settings: ClassifyRequest) -> dict[str, Any]:
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        try:
            return classifier_proposal_for_range(
                capture_id, capture, settings.sample_start,
                min(settings.sample_count, int(capture["sample_count"]) - settings.sample_start),
                settings.sample_count,
                settings.frequency_low_hz, settings.frequency_high_hz,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/captures/{capture_id}/classify/laya")
    def classify_capture_region_with_laya(capture_id: str, settings: LayaClassifyRequest) -> dict[str, Any]:
        if not laya_configured():
            raise HTTPException(status_code=503, detail="Laya is not configured; set LAYA_ENDPOINT")
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        start = settings.sample_start
        if start >= int(capture["sample_count"]):
            raise HTTPException(status_code=422, detail="sample_start is outside the capture")
        end = min(int(capture["sample_count"]), start + settings.sample_count)
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
        if (settings.frequency_low_hz is None) != (settings.frequency_high_hz is None):
            raise HTTPException(status_code=422, detail="frequency bounds must be supplied together")
        if settings.frequency_low_hz is not None and settings.frequency_high_hz is not None:
            if settings.frequency_high_hz <= settings.frequency_low_hz or settings.frequency_high_hz > int(capture["sample_rate"]) / 2:
                raise HTTPException(status_code=422, detail="selected frequency bounds are invalid for this capture")
        try:
            features = extract_features(values, int(capture["sample_rate"]))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        state = {
            "task": "radio modulation method identification",
            "scope": "one selected time window; protocol and variant identification are out of scope",
            "sample_rate_hz": int(capture["sample_rate"]),
            "sample_count": len(values),
            "duration_seconds": len(values) / int(capture["sample_rate"]),
            "selected_frequency_range_hz": [settings.frequency_low_hz, settings.frequency_high_hz],
            "feature_scope": "unfiltered selected time crop; same extraction as synthetic corpus training",
            "feature_version": FEATURE_VERSION,
            "feature_layout": {
                "0:128": "mean normalized log spectral power, evenly spaced from 200 to 3800 Hz",
                "128:256": "temporal variation of normalized log spectral power on the same bins",
                "256:260": ["spectral_centroid_normalized", "spectral_spread_normalized", "spectral_entropy_normalized", "temporal_activity_normalized"],
                "260:266": ["envelope_rms_p10", "envelope_rms_p50", "envelope_rms_p90", "envelope_rms_std", "envelope_active_fraction", "envelope_lag1_correlation"],
            },
            "features": [float(value) for value in features],
        }
        try:
            mode, metadata = classify_with_laya(state, reference_examples=app.state.laya_references)
        except LayaError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        methods = {
            "2fsk": ("FSK", "binary frequency shift keying"),
            "4fsk": ("MFSK", "4-ary frequency shift keying"),
            "8fsk": ("MFSK", "8-ary frequency shift keying"),
            "16fsk": ("MFSK", "16-ary frequency shift keying"),
            "bpsk": ("PSK", "binary phase-shift keying"),
            "qpsk": ("PSK", "quadrature phase-shift keying"),
        }
        family, method = methods.get(mode, (None, None))
        return {
            "choice": mode,
            "family": family,
            "method": method,
            "provider": "Laya System One",
            "model": metadata.get("model"),
            "capture_id": capture_id,
            "sample_start": start,
            "sample_count": len(values),
            "feature_version": FEATURE_VERSION,
            "frequency_isolation": "none",
        }

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

    @app.get("/api/captures/{capture_id}/bit-annotations")
    def get_bit_annotations(capture_id: str) -> list[dict[str, Any]]:
        try:
            return store.bit_annotations(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc

    @app.post("/api/captures/{capture_id}/bit-annotations", status_code=201)
    def create_bit_annotation(capture_id: str, data: BitAnnotationCreate) -> dict[str, Any]:
        try:
            capture = store.capture(capture_id)
            if data.sample_start + data.sample_count > int(capture["sample_count"]):
                raise ValueError("demodulation sample range exceeds the parent capture")
            return store.add_bit_annotation(
                capture_id, mode=data.mode, sample_start=data.sample_start,
                sample_count=data.sample_count, settings=data.settings,
                hypothesis=data.hypothesis, bits=data.bits, bit_start=data.bit_start,
                bit_count=data.bit_count, label=data.label, color=data.color,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/protocol-stencils")
    def list_protocol_stencils() -> list[dict[str, Any]]:
        return store.protocol_stencils()

    @app.post("/api/protocol-stencils", status_code=201)
    def create_protocol_stencil(data: ProtocolStencilCreate) -> dict[str, Any]:
        try:
            return store.add_protocol_stencil(
                name=data.name, field=data.field, pattern=data.pattern,
                protocols=data.protocols, source_capture_id=data.source_capture_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="source capture not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/captures/{capture_id}/regions/{region_id}/history")
    def region_history(capture_id: str, region_id: str) -> list[dict[str, Any]]:
        try:
            if not any(region["id"] == region_id for region in store.regions(capture_id)):
                raise KeyError("region not in this capture")
            return store.label_history(region_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture or region not found") from exc

    @app.get("/api/modes")
    def search_modes(q: str = Query(min_length=1, max_length=80), limit: int = Query(default=20, ge=1, le=50)) -> dict[str, Any]:
        return {"registry_version": MODE_REGISTRY_VERSION, "modes": find_modes(q, limit=limit, additional_modes=store.accepted_mode_definitions())}

    @app.get("/api/mode-proposals")
    def list_mode_proposals() -> list[dict[str, Any]]:
        return store.mode_proposals()

    @app.post("/api/mode-proposals", status_code=201)
    def create_mode_proposal(data: ModeProposalCreate) -> dict[str, Any]:
        try:
            return store.add_mode_proposal(data.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/mode-proposals/{mode_id}/decision", status_code=201)
    def decide_mode_proposal(mode_id: str, data: ModeProposalDecision) -> dict[str, Any]:
        try:
            return store.decide_mode_proposal(mode_id, decision=data.decision, note=data.note)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="mode proposal not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/captures/{capture_id}/regions/{region_id}/labels", status_code=201)
    def revise_region_label(capture_id: str, region_id: str, data: LabelRevisionCreate) -> dict[str, Any]:
        try:
            current_region = next((region for region in store.regions(capture_id) if region["id"] == region_id), None)
            if current_region is None:
                raise KeyError("region not in this capture")
            classifier_evidence = None
            if data.classifier_proposal is not None:
                proposal = data.classifier_proposal
                if proposal.sample_start != current_region["sample_start"] or proposal.selection_sample_count != current_region["sample_count"]:
                    raise ValueError("classifier proposal selection must match the saved region geometry")
                if proposal.frequency_low_hz is not None and (
                    proposal.frequency_low_hz != current_region["frequency_low_hz"]
                    or proposal.frequency_high_hz != current_region["frequency_high_hz"]
                ):
                    raise ValueError("classifier proposal frequency bounds must match the saved region geometry")
                classifier_evidence = classifier_proposal_for_range(
                    capture_id, store.capture(capture_id), proposal.sample_start,
                    proposal.sample_count, proposal.selection_sample_count,
                    proposal.frequency_low_hz, proposal.frequency_high_hz,
                )
            resolved = resolve_mode_label(data.mode_id, {
                "family": data.family, "method": data.method,
                "protocol": data.protocol, "variant": data.variant,
            }, additional_modes=store.accepted_mode_definitions())
            has_label = any((resolved["family"], resolved["method"], resolved["protocol"], resolved["variant"]))
            label_evidence = {"reviewed_from_revision": data.expected_revision}
            if resolved["mode_id"]:
                label_evidence["mode_registry_version"] = resolved["registry_version"]
            label_evidence.update({
                key: current_region["evidence"][key]
                for key in ("detector_proposal", "signal_presence_review")
                if key in current_region["evidence"]
            })
            if classifier_evidence is not None:
                label_evidence["classifier_proposal"] = classifier_evidence
            elif "classifier_proposal" in current_region["evidence"]:
                label_evidence["classifier_proposal"] = current_region["evidence"]["classifier_proposal"]
            store.revise_label(region_id, SignalLabel(
                family=resolved["family"], method=resolved["method"], protocol=resolved["protocol"], variant=resolved["variant"],
                mode_id=resolved["mode_id"], notes=data.notes,
                source="human" if has_label else "unknown",
                review_state="accepted" if has_label else "pending",
                evidence=label_evidence,
            ), expected_revision=data.expected_revision)
            return next(region for region in store.regions(capture_id) if region["id"] == region_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture or region not found") from exc
        except LabelRevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/captures/{capture_id}/regions/{region_id}/feedback")
    def region_feedback_history(capture_id: str, region_id: str) -> list[dict[str, Any]]:
        try:
            if not any(region["id"] == region_id for region in store.regions(capture_id)):
                raise KeyError("region not in this capture")
            return store.feedback_history(region_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture or region not found") from exc

    @app.get("/api/captures/{capture_id}/regions/{region_id}/geometry")
    def region_geometry_history(capture_id: str, region_id: str) -> list[dict[str, Any]]:
        try:
            if not any(region["id"] == region_id for region in store.regions(capture_id)):
                raise KeyError("region not in this capture")
            return store.geometry_history(region_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture or region not found") from exc

    @app.post("/api/captures/{capture_id}/regions/{region_id}/geometry", status_code=201)
    def revise_region_geometry(capture_id: str, region_id: str, data: GeometryRevisionCreate) -> dict[str, Any]:
        try:
            if not any(region["id"] == region_id for region in store.regions(capture_id)):
                raise KeyError("region not in this capture")
            revision = store.revise_region_geometry(
                region_id, expected_revision=data.expected_revision,
                sample_start=data.sample_start, sample_count=data.sample_count,
                frequency_low_hz=data.frequency_low_hz, frequency_high_hz=data.frequency_high_hz,
            )
            revision["geometry_revision"] = revision.pop("revision")
            return revision
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture or region not found") from exc
        except LabelRevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/captures/{capture_id}/regions/{region_id}/lineage")
    def region_lineage_history(capture_id: str, region_id: str) -> list[dict[str, Any]]:
        try:
            if not any(region["id"] == region_id for region in store.regions(capture_id)):
                raise KeyError("region not in this capture")
            return store.region_lineage(region_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture or region not found") from exc

    @app.post("/api/captures/{capture_id}/regions/{region_id}/split", status_code=201)
    def split_region(capture_id: str, region_id: str, data: RegionSplitCreate) -> dict[str, Any]:
        try:
            return store.split_region(
                capture_id, region_id,
                expected_geometry_revision=data.expected_geometry_revision,
                children=[{
                    "sample_start": child.sample_start,
                    "sample_count": child.sample_count,
                    "frequency_low_hz": child.frequency_low_hz,
                    "frequency_high_hz": child.frequency_high_hz,
                } for child in data.children],
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture or region not found") from exc
        except LabelRevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/captures/{capture_id}/regions/merge", status_code=201)
    def merge_regions(capture_id: str, data: RegionMergeCreate) -> dict[str, Any]:
        try:
            return store.merge_regions(
                capture_id, source_region_ids=data.source_region_ids,
                expected_geometry_revisions=data.expected_geometry_revisions,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture or region not found") from exc
        except LabelRevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/captures/{capture_id}/coverage")
    def list_scene_coverage(capture_id: str) -> list[dict[str, Any]]:
        try:
            return store.scene_coverages(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc

    @app.post("/api/captures/{capture_id}/coverage", status_code=201)
    def create_scene_coverage(capture_id: str, data: SceneCoverageCreate) -> dict[str, Any]:
        try:
            return store.create_scene_coverage(capture_id, **data.model_dump())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/captures/{capture_id}/coverage/{coverage_id}/history")
    def scene_coverage_history(capture_id: str, coverage_id: str) -> list[dict[str, Any]]:
        try:
            return store.scene_coverage_history(capture_id, coverage_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture or scene coverage not found") from exc

    @app.post("/api/captures/{capture_id}/coverage/{coverage_id}", status_code=201)
    def revise_scene_coverage(capture_id: str, coverage_id: str, data: SceneCoverageRevision) -> dict[str, Any]:
        try:
            values = data.model_dump()
            expected_revision = values.pop("expected_revision")
            return store.revise_scene_coverage(
                capture_id, coverage_id, expected_revision=expected_revision, **values,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture or scene coverage not found") from exc
        except LabelRevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/captures/{capture_id}/regions/{region_id}/feedback", status_code=201)
    def create_region_feedback(capture_id: str, region_id: str, data: AnnotationFeedbackCreate) -> dict[str, Any]:
        try:
            if not any(region["id"] == region_id for region in store.regions(capture_id)):
                raise KeyError("region not in this capture")
            resolved = resolve_mode_label(data.corrected_mode_id, {
                "family": data.corrected_family, "method": data.corrected_method,
                "protocol": data.corrected_protocol, "variant": data.corrected_variant,
            }, additional_modes=store.accepted_mode_definitions())
            correction = {key: resolved[key] for key in ("mode_id", "family", "method", "protocol", "variant") if resolved[key] is not None}
            if resolved["mode_id"]:
                correction["registry_version"] = resolved["registry_version"]
            return store.record_feedback(
                region_id, expected_revision=data.expected_revision, action=data.action,
                expected_geometry_revision=data.expected_geometry_revision,
                correction=correction, note=data.note,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture or region not found") from exc
        except LabelRevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/captures/{capture_id}/regions", status_code=201)
    def create_region(capture_id: str, data: RegionCreate) -> dict[str, Any]:
        try:
            capture = store.capture(capture_id)
            detector_evidence = data.detector_evidence
            if data.signal_presence_confirmed and data.origin != "detector_proposal":
                raise ValueError("signal presence confirmation is only valid for a detector proposal")
            if detector_evidence is not None:
                if data.origin != "detector_proposal":
                    raise ValueError("detector evidence is only valid for a detector proposal")
                if detector_evidence.sample_rate != int(capture["sample_rate"]):
                    raise ValueError("detector sample rate does not match the capture")
                window_start = detector_evidence.window_sample_start
                window_end = window_start + detector_evidence.window_sample_count
                if window_end > int(capture["sample_count"]):
                    raise ValueError("detector window exceeds the capture sample range")
                if data.sample_start < window_start or data.sample_start + data.sample_count > window_end:
                    raise ValueError("proposal region must remain inside its detector window")
                for source_window in detector_evidence.supporting_windows:
                    source_start = source_window.sample_start
                    source_end = source_start + source_window.sample_count
                    if source_end > int(capture["sample_count"]):
                        raise ValueError("supporting detector window exceeds the capture sample range")
                    if min(source_end, data.sample_start + data.sample_count) <= max(source_start, data.sample_start):
                        raise ValueError("supporting detector window must overlap the proposed region")
            elif data.signal_presence_confirmed:
                raise ValueError("signal presence confirmation requires detector evidence")
            resolved = resolve_mode_label(data.mode_id, {
                "family": data.family, "method": data.method,
                "protocol": data.protocol, "variant": data.variant,
            }, additional_modes=store.accepted_mode_definitions())
            has_label = any((resolved["family"], resolved["method"], resolved["protocol"], resolved["variant"]))
            accepted_presence = data.signal_presence_confirmed
            classifier_evidence = None
            if data.classifier_proposal is not None:
                proposal = data.classifier_proposal
                if proposal.sample_start != data.sample_start or proposal.selection_sample_count != data.sample_count:
                    raise ValueError("classifier proposal selection must match the saved region geometry")
                if proposal.frequency_low_hz is not None and (
                    proposal.frequency_low_hz != data.frequency_low_hz
                    or proposal.frequency_high_hz != data.frequency_high_hz
                ):
                    raise ValueError("classifier proposal frequency bounds must match the saved region geometry")
                classifier_evidence = classifier_proposal_for_range(
                    capture_id, capture, proposal.sample_start,
                    proposal.sample_count, proposal.selection_sample_count,
                    proposal.frequency_low_hz, proposal.frequency_high_hz,
                )
            region_id = store.add_region(
                capture_id,
                sample_start=data.sample_start,
                sample_count=data.sample_count,
                frequency_low_hz=data.frequency_low_hz,
                frequency_high_hz=data.frequency_high_hz,
                label=SignalLabel(
                    family=resolved["family"],
                    method=resolved["method"],
                    protocol=resolved["protocol"],
                    variant=resolved["variant"],
                    mode_id=resolved["mode_id"],
                    source="human" if has_label or accepted_presence else "unknown",
                    review_state="accepted" if has_label or accepted_presence else "pending",
                    evidence={
                        **({"mode_registry_version": resolved["registry_version"]} if resolved["mode_id"] else {}),
                        **({"detector_proposal": detector_evidence.model_dump()} if detector_evidence else {}),
                        **({"classifier_proposal": classifier_evidence} if classifier_evidence is not None else {}),
                        **({"signal_presence_review": "confirmed_unknown_identity"} if accepted_presence else {}),
                    } or None,
                    notes=data.notes,
                ),
                origin=data.origin,
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

    @app.get("/api/exports/bit-annotations.jsonl")
    def export_bit_annotations() -> FileResponse:
        destination = root / "exports" / "bit-annotations.jsonl"
        store.export_bit_annotations_jsonl(destination)
        return FileResponse(
            destination,
            media_type="application/x-ndjson",
            filename="signal-bit-annotations.jsonl",
        )

    @app.get("/api/exports/cv-proposal-decisions.jsonl")
    def export_cv_proposal_decisions() -> FileResponse:
        destination = root / "exports" / "cv-proposal-decisions.jsonl"
        store.export_cv_proposal_decisions_jsonl(destination)
        return FileResponse(
            destination,
            media_type="application/x-ndjson",
            filename="cv-proposal-review-decisions.jsonl",
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

    @app.get("/bit-inspector.mjs", include_in_schema=False)
    def bit_inspector_module() -> FileResponse:
        return FileResponse(WEB_DIR / "bit-inspector.mjs", media_type="text/javascript")

    @app.get("/protocol-stencils.mjs", include_in_schema=False)
    def protocol_stencils_module() -> FileResponse:
        return FileResponse(WEB_DIR / "protocol-stencils.mjs", media_type="text/javascript")

    @app.get("/mode-candidates.mjs", include_in_schema=False)
    def mode_candidates_module() -> FileResponse:
        return FileResponse(WEB_DIR / "mode-candidates.mjs", media_type="text/javascript")

    @app.get("/signal-overlays.mjs", include_in_schema=False)
    def signal_overlays_module() -> FileResponse:
        return FileResponse(WEB_DIR / "signal-overlays.mjs", media_type="text/javascript")

    @app.get("/live-waterfall.mjs", include_in_schema=False)
    def live_waterfall_module() -> FileResponse:
        return FileResponse(WEB_DIR / "live-waterfall.mjs", media_type="text/javascript")

    @app.get("/style.css", include_in_schema=False)
    def stylesheet() -> FileResponse:
        return FileResponse(WEB_DIR / "style.css", media_type="text/css")

    return app
