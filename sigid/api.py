"""HTTP API and small browser workbench for WAV replay and manual signal labels."""

from __future__ import annotations

import os
import asyncio
import base64
import fcntl
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
import logging
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, model_validator

from .dataset import CaptureStore, SignalLabel, LabelRevisionConflict
from .decoder_backend import DecoderBackendRegistry, DecoderRequest, NativeDecoderBackend
from .classifier import (
    CLASSIFIER_WINDOW_SECONDS, FEATURE_VERSION, FIXED_WINDOW_FEATURE_VERSIONS,
    ROI_FEATURE_VERSIONS, extract_features, load_model, propose_loaded,
)
from .cv_detector import (
    FFT_SIZE,
    MAX_FREQUENCY_HZ,
    OVERLAP_DEDUP_IOU,
    PREPROCESSING_VERSION,
    WINDOW_SECONDS,
    WINDOW_STRIDE_SECONDS,
    YoloSignalDetector,
    spectrogram_image,
    proposal_id as cv_proposal_id,
)
from .cv_embeddings import load_embedding_index, read_audio_range, region_image
from .laya import LayaError, classify as classify_with_laya, configured as laya_configured, load_reference_examples
from .demod import demodulate_2fsk, demodulate_bpsk, demodulate_mfsk, demodulate_ook, demodulate_qpsk
from .detection import detect_tracks
from .fldigi import FldigiClient
from .fldigi_backend import FldigiNativeBackend
from .fldigi_replay import IsolatedFldigiReplay
from .wsjtx_backend import WsjtxUdpBackend
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
RE_REVIEW_INTERVAL = timedelta(days=90)
WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_MFSK_RATE_SEARCH_WORK = 200_000_000


def _raw_attempt_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep a useful, bounded comparison preview without duplicating large vectors."""
    truncated: list[str] = []

    def compact(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {name: compact(item, name) for name, item in value.items()}
        if isinstance(value, list):
            if len(value) > 256:
                truncated.append(key)
                return {"preview": [compact(item, key) for item in value[:256]], "total_items": len(value)}
            return [compact(item, key) for item in value]
        if isinstance(value, str) and len(value) > 4096:
            truncated.append(key)
            return {"preview": value[:4096], "total_characters": len(value)}
        return value

    compacted = compact(result)
    if truncated:
        compacted["comparison_preview_truncated"] = sorted(set(truncated))
    return compacted


def _persist_raw_attempt(
    store: CaptureStore, capture_id: str, mode: str, settings: BaseModel,
    result: dict[str, Any],
) -> dict[str, Any]:
    attempt = store.add_raw_demod_attempt(
        capture_id, mode=mode, sample_start=int(result["sample_start"]),
        sample_count=int(result["sample_count"]), settings=settings.model_dump(),
        result=_raw_attempt_result(result),
    )
    result["attempt_id"] = attempt["id"]
    return result


def _classify_signal_crop(
    classifier: dict[str, Any], samples: np.ndarray, sample_rate: int,
    frequency_low_hz: float, frequency_high_hz: float,
) -> dict[str, Any]:
    """Run the active classifier on a bounded crop associated with a CV box."""
    metadata = classifier["metadata"]
    window_seconds = float(metadata.get("classifier_window_seconds") or CLASSIFIER_WINDOW_SECONDS)
    features = extract_features(
        samples, sample_rate, feature_version=metadata["feature_version"],
        classifier_window_seconds=window_seconds,
        frequency_range_hz=(
            (frequency_low_hz, frequency_high_hz)
            if metadata.get("feature_version") in ROI_FEATURE_VERSIONS else None
        ),
    )
    return {
        "available": True,
        **propose_loaded(features, metadata, classifier["parameters"]),
    }


def default_track_fft_size(sample_rate: int) -> int:
    """Choose a track-analysis window near 42 ms, capped by the API FFT limit."""
    target = max(32, int(np.ceil(sample_rate / 24)))
    return min(8192, 1 << int(np.ceil(np.log2(target))))


def tune_rigctl_frequency(host: str, port: int, frequency_hz: int) -> None:
    """Set only the receiver frequency through an existing rigctld service."""
    executable = shutil.which(os.environ.get("SIG_ID_RIGCTL", "rigctl"))
    if not executable:
        raise RuntimeError("Hamlib rigctl was not found on the server")
    try:
        result = subprocess.run(
            [executable, "-m", "2", "-r", f"{host}:{port}", "F", str(frequency_hz)],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("rigctl timed out while tuning the receiver") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not run rigctl: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"rigctl exited with status {result.returncode}"
        raise RuntimeError(detail[-500:])


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
    crop_count: int = Field(default=1, ge=1, le=20)

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
    identity_unknown_confirmed: bool = False
    prediction_only: bool = False


class WindowAnnotationRegionCreate(BaseModel):
    client_id: str = Field(min_length=1, max_length=80)
    sample_start: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    frequency_low_hz: float = Field(ge=0, le=96000)
    frequency_high_hz: float = Field(gt=0, le=96000)
    family: str | None = Field(default=None, max_length=200)
    method: str | None = Field(default=None, max_length=200)
    protocol: str | None = Field(default=None, max_length=200)
    variant: str | None = Field(default=None, max_length=200)
    mode_id: str | None = Field(default=None, max_length=80)
    notes: str = Field(default="", max_length=20000)
    origin: Literal["manual", "detector_proposal", "user_added_missed_detection"] = "manual"
    detector_evidence: DetectorProposalEvidence | None = None
    classifier_proposal: ClassifierProposalRequest | None = None
    signal_presence_confirmed: bool = False
    identity_unknown_confirmed: bool = False

    @model_validator(mode="after")
    def validate_box(self) -> "WindowAnnotationRegionCreate":
        if self.frequency_low_hz >= self.frequency_high_hz:
            raise ValueError("frequency bounds must be increasing")
        if self.detector_evidence is not None and self.origin != "detector_proposal":
            raise ValueError("detector evidence is only valid for a detector proposal")
        if self.signal_presence_confirmed and (self.origin != "detector_proposal" or self.detector_evidence is None):
            raise ValueError("signal presence confirmation requires detector evidence")
        if self.classifier_proposal is not None:
            proposal = self.classifier_proposal
            if proposal.sample_start != self.sample_start or proposal.selection_sample_count != self.sample_count:
                raise ValueError("classifier proposal selection must match the saved region geometry")
            if proposal.frequency_low_hz is not None and (
                proposal.frequency_low_hz != self.frequency_low_hz
                or proposal.frequency_high_hz != self.frequency_high_hz
            ):
                raise ValueError("classifier proposal frequency bounds must match the saved region geometry")
        if self.identity_unknown_confirmed and any((self.mode_id, self.family, self.method, self.protocol, self.variant)):
            raise ValueError("an explicitly unknown label cannot also contain a mode or taxonomy")
        return self


class WindowAnnotationRegionUpdate(BaseModel):
    region_id: str = Field(min_length=1, max_length=80)
    expected_revision: int = Field(ge=1)
    expected_geometry_revision: int = Field(ge=1)
    sample_start: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    frequency_low_hz: float = Field(ge=0, le=96000)
    frequency_high_hz: float = Field(gt=0, le=96000)
    family: str | None = Field(default=None, max_length=200)
    method: str | None = Field(default=None, max_length=200)
    protocol: str | None = Field(default=None, max_length=200)
    variant: str | None = Field(default=None, max_length=200)
    mode_id: str | None = Field(default=None, max_length=80)
    notes: str = Field(default="", max_length=20000)
    identity_unknown_confirmed: bool = False

    @model_validator(mode="after")
    def validate_update(self) -> "WindowAnnotationRegionUpdate":
        if self.frequency_low_hz >= self.frequency_high_hz:
            raise ValueError("frequency bounds must be increasing")
        if self.identity_unknown_confirmed and any((self.mode_id, self.family, self.method, self.protocol, self.variant)):
            raise ValueError("an explicitly unknown label cannot also contain a mode or taxonomy")
        return self


class WindowAnnotationRegionDelete(BaseModel):
    region_id: str = Field(min_length=1, max_length=80)
    expected_revision: int = Field(ge=1)
    expected_geometry_revision: int = Field(ge=1)


class WindowHeuristicMergeProposal(BaseModel):
    candidate_id: str = Field(min_length=1, max_length=80)
    sample_start: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    frequency_low_hz: float = Field(ge=0, le=96000)
    frequency_high_hz: float = Field(gt=0, le=96000)
    modulation_hypothesis: str = Field(min_length=1, max_length=80)
    tone_count_candidate: int = Field(ge=1, le=64)
    tone_centers_hz: list[float] = Field(default_factory=list, max_length=64)
    peak_snr_db: float | None = None

    @model_validator(mode="after")
    def validate_candidate(self) -> "WindowHeuristicMergeProposal":
        if self.frequency_low_hz >= self.frequency_high_hz:
            raise ValueError("heuristic candidate frequency bounds must be increasing")
        if any(not np.isfinite(tone) or not 0 <= tone <= 96000 for tone in self.tone_centers_hz):
            raise ValueError("heuristic candidate tone centers must be non-negative")
        if self.peak_snr_db is not None and not np.isfinite(self.peak_snr_db):
            raise ValueError("heuristic candidate SNR must be finite")
        return self


class WindowAnnotationMerge(BaseModel):
    client_id: str = Field(min_length=1, max_length=80)
    source_region_ids: list[str] = Field(default_factory=list, max_length=32)
    expected_revisions: dict[str, int] = Field(default_factory=dict)
    expected_geometry_revisions: dict[str, int] = Field(default_factory=dict)
    cv_proposals: list[CVProposalDecisionCreate] = Field(default_factory=list, max_length=32)
    heuristic_proposals: list[WindowHeuristicMergeProposal] = Field(default_factory=list, max_length=32)
    target_region_id: str | None = Field(default=None, max_length=80)
    sample_start: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    frequency_low_hz: float = Field(ge=0, le=96000)
    frequency_high_hz: float = Field(gt=0, le=96000)
    family: str | None = Field(default=None, max_length=200)
    method: str | None = Field(default=None, max_length=200)
    protocol: str | None = Field(default=None, max_length=200)
    variant: str | None = Field(default=None, max_length=200)
    mode_id: str | None = Field(default=None, max_length=80)
    notes: str = Field(default="", max_length=20000)
    identity_unknown_confirmed: bool = False

    @model_validator(mode="after")
    def validate_merge(self) -> "WindowAnnotationMerge":
        if not 1 <= len(self.source_region_ids) + len(self.cv_proposals) + len(self.heuristic_proposals) <= 32:
            raise ValueError("a staged merge needs one to 32 saved or provisional source boxes")
        if len(set(self.source_region_ids)) != len(self.source_region_ids):
            raise ValueError("merge source regions must be distinct")
        if set(self.expected_geometry_revisions) != set(self.source_region_ids):
            raise ValueError("expected geometry revisions must identify each saved merge source")
        if set(self.expected_revisions) != set(self.source_region_ids):
            raise ValueError("expected label revisions must identify each saved merge source")
        if self.target_region_id and self.target_region_id not in self.source_region_ids:
            raise ValueError("the saved merge target must also be included as a source")
        if self.frequency_low_hz >= self.frequency_high_hz:
            raise ValueError("merge result frequency bounds must be increasing")
        if self.identity_unknown_confirmed and any((self.mode_id, self.family, self.method, self.protocol, self.variant)):
            raise ValueError("an explicitly unknown merge result cannot also contain a mode or taxonomy")
        if len({item.proposal_id for item in self.cv_proposals}) != len(self.cv_proposals):
            raise ValueError("CV proposal ids must be unique within a merge")
        if len({item.candidate_id for item in self.heuristic_proposals}) != len(self.heuristic_proposals):
            raise ValueError("heuristic candidate ids must be unique within a merge")
        return self


class WindowAnnotationIdentityLink(BaseModel):
    signal_id: str | None = Field(default=None, pattern="^S[1-9][0-9]*$", max_length=40)
    target_region_id: str | None = Field(default=None, max_length=80)
    target_client_id: str | None = Field(default=None, max_length=80)
    region_ids: list[str] = Field(default_factory=list, max_length=128)
    client_ids: list[str] = Field(default_factory=list, max_length=128)
    expected_signal_ids: dict[str, str] = Field(default_factory=dict)
    expected_revisions: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity_link(self) -> "WindowAnnotationIdentityLink":
        if not self.region_ids and not self.client_ids:
            raise ValueError("an identity link needs at least one saved region or draft box")
        if len(set(self.region_ids)) != len(self.region_ids) or len(set(self.client_ids)) != len(self.client_ids):
            raise ValueError("identity link sources must be unique")
        direct_id = self.signal_id is not None
        target = self.target_region_id is not None or self.target_client_id is not None
        if direct_id == target or (self.target_region_id and self.target_client_id):
            raise ValueError("choose either an existing signal ID or one selected box as the identity target")
        if self.target_region_id and self.target_region_id not in self.region_ids:
            raise ValueError("the target saved region must be included in the linked boxes")
        if self.target_client_id and self.target_client_id not in self.client_ids:
            raise ValueError("the target draft box must be included in the linked boxes")
        if set(self.expected_signal_ids) != set(self.region_ids):
            raise ValueError("expected signal IDs must identify every saved box being linked")
        if set(self.expected_revisions) != set(self.region_ids):
            raise ValueError("expected label revisions must identify every saved box being linked")
        if any(not re.fullmatch(r"S[1-9][0-9]*", signal_id) for signal_id in self.expected_signal_ids.values()):
            raise ValueError("expected signal IDs must be valid S-numbers")
        return self


class WindowAnnotationBatchCreate(BaseModel):
    idempotency_key: str = Field(min_length=8, max_length=80)
    window_start_sample: int = Field(ge=0)
    window_sample_count: int = Field(gt=0)
    regions: list[WindowAnnotationRegionCreate] = Field(default_factory=list, max_length=128)
    updates: list[WindowAnnotationRegionUpdate] = Field(default_factory=list, max_length=128)
    deletions: list[WindowAnnotationRegionDelete] = Field(default_factory=list, max_length=128)
    merges: list[WindowAnnotationMerge] = Field(default_factory=list, max_length=32)
    identity_links: list[WindowAnnotationIdentityLink] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def validate_window(self) -> "WindowAnnotationBatchCreate":
        operation_count = len(self.regions) + len(self.updates) + len(self.deletions) + len(self.merges) + len(self.identity_links)
        if not 1 <= operation_count <= 128:
            raise ValueError("a window save must include one to 128 annotation changes")
        if len({item.client_id for item in self.regions}) != len(self.regions):
            raise ValueError("annotation client ids must be unique")
        region_ids = [item.region_id for item in [*self.updates, *self.deletions]]
        if len(set(region_ids)) != len(region_ids):
            raise ValueError("each saved region can be updated or deleted only once per window save")
        merged_region_ids = [region_id for merge in self.merges for region_id in merge.source_region_ids]
        if len(set(merged_region_ids)) != len(merged_region_ids):
            raise ValueError("a saved region can be used in only one merge per window save")
        if len({merge.client_id for merge in self.merges}) != len(self.merges):
            raise ValueError("merge client ids must be unique")
        merged_proposal_ids = [proposal.proposal_id for merge in self.merges for proposal in merge.cv_proposals]
        if len(set(merged_proposal_ids)) != len(merged_proposal_ids):
            raise ValueError("a CV proposal can be used in only one merge per window save")
        merged_heuristic_ids = [proposal.candidate_id for merge in self.merges for proposal in merge.heuristic_proposals]
        if len(set(merged_heuristic_ids)) != len(merged_heuristic_ids):
            raise ValueError("a heuristic candidate can be used in only one merge per window save")
        if set(merged_region_ids) & set(region_ids):
            raise ValueError("merge sources cannot also be updated or deleted in the same window save")
        linked_region_ids = [region_id for link in self.identity_links for region_id in link.region_ids]
        linked_client_ids = [client_id for link in self.identity_links for client_id in link.client_ids]
        if len(set(linked_region_ids)) != len(linked_region_ids) or len(set(linked_client_ids)) != len(linked_client_ids):
            raise ValueError("a box can participate in only one identity link per window save")
        if set(linked_region_ids) & {item.region_id for item in self.deletions}:
            raise ValueError("a box cannot be linked and deleted in the same window save")
        if set(linked_region_ids) & set(merged_region_ids):
            raise ValueError("a merge source cannot also be identity-linked in the same window save")
        if set(linked_client_ids) - {item.client_id for item in self.regions}:
            raise ValueError("identity links may reference only draft boxes included in the same save")
        window_end = self.window_start_sample + self.window_sample_count
        if any(item.sample_start < self.window_start_sample or item.sample_start + item.sample_count > window_end for item in self.regions):
            raise ValueError("each annotation must remain inside this 10-second window")
        if any(item.sample_start < self.window_start_sample or item.sample_start + item.sample_count > window_end for item in self.updates):
            raise ValueError("each updated annotation must remain inside this 10-second window")
        if any(item.sample_start < self.window_start_sample or item.sample_start + item.sample_count > window_end for item in self.merges):
            raise ValueError("each merge result must remain inside this 10-second window")
        return self


class DecodeRunCollection(BaseModel):
    notes: str = Field(default="", max_length=20000)


class LiveTrainingAnnotation(BaseModel):
    session_id: str = Field(min_length=1, max_length=80)
    sample_start: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    frequency_low_hz: float = Field(ge=0, le=96000)
    frequency_high_hz: float = Field(gt=0, le=96000)
    family: str | None = Field(default=None, max_length=200)
    method: str | None = Field(default=None, max_length=200)
    protocol: str | None = Field(default=None, max_length=200)
    variant: str | None = Field(default=None, max_length=200)
    mode_id: str | None = Field(default=None, max_length=80)
    notes: str = Field(default="", max_length=20000)


class LabelRevisionCreate(BaseModel):
    expected_revision: int = Field(ge=1)
    family: str | None = Field(default=None, max_length=200)
    method: str | None = Field(default=None, max_length=200)
    protocol: str | None = Field(default=None, max_length=200)
    variant: str | None = Field(default=None, max_length=200)
    mode_id: str | None = Field(default=None, max_length=80)
    notes: str = Field(default="", max_length=20000)
    classifier_proposal: ClassifierProposalRequest | None = None
    identity_unknown_confirmed: bool = False


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
    source_region_ids: list[str] = Field(default_factory=list, max_length=32)
    expected_geometry_revisions: dict[str, int]
    cv_proposals: list[CVProposalDecisionCreate] = Field(default_factory=list, max_length=32)
    result_bounds: RegionBounds | None = None

    @model_validator(mode="after")
    def validate_merge_sources(self) -> "RegionMergeCreate":
        if not 1 <= len(self.source_region_ids) + len(self.cv_proposals) <= 32:
            raise ValueError("a merge needs one to 32 saved boxes or CV proposals")
        if not self.result_bounds and (len(self.source_region_ids) + len(self.cv_proposals) < 2):
            raise ValueError("a single source needs drawn result bounds")
        if len({proposal.proposal_id for proposal in self.cv_proposals}) != len(self.cv_proposals):
            raise ValueError("CV proposal ids must be unique")
        return self


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
    max_duration_seconds: int = Field(default=600, ge=1, le=28800)
    capture_mode: Literal["record", "scan_only", "unattended"] = "record"
    noise_interval_seconds: int = Field(default=300, ge=60, le=3600)
    band_hop_frequencies_hz: list[int] = Field(default_factory=list, max_length=32)
    band_hop_dwell_seconds: int = Field(default=300, ge=30, le=3600)
    rigctl_host: str = Field(default="127.0.0.1", min_length=1, max_length=253)
    rigctl_port: int = Field(default=4532, ge=1, le=65535)

    @model_validator(mode="after")
    def validate_band_hopping(self) -> "LiveCaptureStart":
        if self.band_hop_frequencies_hz:
            if self.capture_mode != "unattended":
                raise ValueError("band hopping is available only in unattended mode")
            if len(self.band_hop_frequencies_hz) < 2:
                raise ValueError("band hopping needs at least two receiver frequencies")
            if len(set(self.band_hop_frequencies_hz)) != len(self.band_hop_frequencies_hz):
                raise ValueError("band-hop frequencies must be unique")
            if any(not 100_000 <= value <= 6_000_000_000 for value in self.band_hop_frequencies_hz):
                raise ValueError("band-hop frequencies must be between 0.1 MHz and 6000 MHz")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}", self.rigctl_host):
                raise ValueError("rigctld host must be a hostname or IP address")
        return self


class RigctlFrequencyRequest(BaseModel):
    host: str = Field(default="127.0.0.1", min_length=1, max_length=253)
    port: int = Field(default=4532, ge=1, le=65535)

    @model_validator(mode="after")
    def validate_host(self) -> "RigctlFrequencyRequest":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}", self.host):
            raise ValueError("host must be a hostname or IP address without a scheme or path")
        return self


class FldigiModeSelect(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class FldigiTrialStart(BaseModel):
    track_id: str = Field(min_length=1, max_length=128)
    modulation_hypothesis: str = Field(default="unknown", max_length=80)
    sample_start: int | None = Field(default=None, ge=0)
    sample_count: int | None = Field(default=None, gt=0, le=192_000 * 600)


class FldigiReplayRequest(BaseModel):
    mode: str = Field(min_length=1, max_length=80)
    auto_identify: bool = False
    candidate_modes: list[str] = Field(default_factory=list, max_length=6)
    sample_start: int = Field(ge=0)
    sample_count: int = Field(gt=0, le=192_000 * 600)
    track_id: str | None = Field(default=None, max_length=128)
    carrier_hz: float | None = Field(default=None, ge=0, le=20000)
    region_id: str | None = Field(default=None, max_length=80)
    detector_evidence: DetectorProposalEvidence | None = None
    frequency_low_hz: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    frequency_high_hz: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_frequency_band(self) -> "FldigiReplayRequest":
        if (self.frequency_low_hz is None) != (self.frequency_high_hz is None):
            raise ValueError("supply both selected frequency bounds")
        if self.frequency_low_hz is not None and self.frequency_high_hz <= self.frequency_low_hz:
            raise ValueError("selected frequency bounds must increase")
        return self

    def selection_evidence(self, sample_rate: int) -> dict[str, Any]:
        if self.frequency_low_hz is None:
            return {}
        if self.frequency_high_hz > sample_rate / 2:
            raise HTTPException(status_code=422, detail="selected frequency bounds exceed source Nyquist range")
        return {"selection_bounds": {"source": "operator_selection",
                "frequency_low_hz": self.frequency_low_hz, "frequency_high_hz": self.frequency_high_hz}}


def _decode_text_coherence(text: str) -> float:
    """Rank decoder outputs for human-readable text; never treat score as a label."""
    if not text:
        return 0.0
    printable = [char for char in text if char.isprintable() or char in "\r\n\t"]
    visible = "".join(char for char in printable if char not in "\r\n\t")
    if not visible:
        return 0.0
    letters = sum(char.isalpha() for char in visible)
    spaces = sum(char.isspace() for char in visible)
    trigrams = {
        "the", "and", "ing", "ion", "tio", "ent", "her", "for", "tha", "nth",
        "int", "ere", "ter", "est", "ers", "ati", "hat", "ate", "all", "eth",
        "hes", "ver", "his", "oft", "sth", "ith", "fth", "men", "rea", "you",
    }
    lowered = "".join(char.lower() if char.isalpha() else " " for char in visible)
    grams = [lowered[index:index + 3] for index in range(max(0, len(lowered) - 2))
             if lowered[index:index + 3].isalpha()]
    common = sum(gram in trigrams for gram in grams) / max(1, len(grams))
    length_factor = min(1.0, len(visible) / 48.0)
    printable_ratio = len(printable) / max(1, len(text))
    letter_ratio = letters / len(visible)
    word_spacing = min(1.0, spaces / max(1, len(visible) / 12.0))
    return round(100.0 * length_factor * (
        0.20 * printable_ratio + 0.25 * letter_ratio + 0.15 * word_spacing + 0.40 * common
    ), 1)


def _checked_replay_modes(settings: FldigiReplayRequest, available: list[str]) -> list[str]:
    candidates = list(dict.fromkeys(settings.candidate_modes)) if settings.auto_identify else [settings.mode]
    if settings.auto_identify and not candidates:
        candidates = [settings.mode]
    if any(mode not in available for mode in candidates):
        raise ValueError("one or more candidate modes are not available in the Fldigi catalog")
    return candidates


class NativeDecoderRequestBody(BaseModel):
    mode_id: str = Field(min_length=1, max_length=100)
    sample_start: int = Field(ge=0)
    sample_count: int = Field(gt=0, le=192_000 * 600)
    carrier_hz: float | None = Field(default=None, ge=0, le=96_000)
    region_id: str | None = Field(default=None, max_length=80)


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
    crop_count: int = Field(default=1, ge=1, le=20)

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
    decoder_backends: tuple[NativeDecoderBackend, ...] = (),
) -> FastAPI:
    root = Path(data_dir or os.environ.get("SIG_ID_DATA_DIR", "~/.local/share/sig-id")).expanduser()
    capture_dir = root / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    store = CaptureStore(root / "dataset.sqlite3")
    # Field-guide reference transmissions are kept in their own store so they
    # cannot be swept into live review or the normal training dataset.
    field_guide_store = CaptureStore(root / "field-guide" / "dataset.sqlite3")
    live_capture = LiveCaptureManager(store, capture_dir, audio_backend=audio_backend)
    receive_products = ReceiveProductStore(root / "receive-products")
    fldigi = fldigi_client or FldigiClient(
        url=os.environ.get("SIG_ID_FLDIGI_URL", "http://127.0.0.1:7362/RPC2"),
        timeout=float(os.environ.get("SIG_ID_FLDIGI_TIMEOUT", "0.75")),
    )
    fldigi_replay = fldigi_replay_backend or IsolatedFldigiReplay()
    configured_backends = list(decoder_backends)
    if not any(str(getattr(backend, "backend_id", "")).casefold() == "fldigi" for backend in configured_backends):
        configured_backends.append(FldigiNativeBackend(fldigi, fldigi_replay))
    if not any(str(getattr(backend, "backend_id", "")).casefold() == "wsjtx" for backend in configured_backends):
        wsjtx_port_value = os.environ.get("SIG_ID_WSJTX_UDP_PORT", "").strip()
        try:
            wsjtx_port = int(wsjtx_port_value) if wsjtx_port_value else None
        except ValueError:
            wsjtx_port = None
            logging.getLogger(__name__).warning("Ignoring invalid SIG_ID_WSJTX_UDP_PORT")
        configured_backends.append(WsjtxUdpBackend(port=wsjtx_port))
    decoder_backends_registry = DecoderBackendRegistry.discover(tuple(configured_backends))
    decode_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sig-id-fldigi-replay")
    model_training_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sig-id-model-training")
    decode_job_lock = Lock()
    model_training_lock = Lock()
    # The active pair belongs to the project model catalog, not the user's
    # capture store. Keep direct `sig-id-serve` starts aligned with the
    # canonical pointer written by the training pipeline and run_sig_id.sh.
    active_models_file = Path(os.environ.get(
        "SIGID_ACTIVE_MODELS_JSON",
        Path(__file__).resolve().parents[1] / "data" / "models" / "active-models.json",
    )).expanduser()
    active_models: dict[str, Any] = {}
    try:
        active_models = json.loads(active_models_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    configured_model = classifier_model_path or active_models.get("classifier") or os.environ.get("SIG_ID_MODEL_PATH")
    classifier = None
    classifier_error = None
    if configured_model:
        try:
            model_metadata, model_parameters = load_model(configured_model)
            classifier = {"metadata": model_metadata, "parameters": model_parameters}
        except Exception as exc:
            classifier_error = str(exc)

    configured_cv_model = active_models.get("cv_detector") or os.environ.get("SIG_ID_CV_MODEL_PATH")
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
            with application.state.band_hop_lock:
                hopping = list(application.state.band_hop_threads.values())
            for stop_event, thread in hopping:
                stop_event.set()
                thread.join(timeout=5)
            application.state.fldigi_trials.cancel()
            decoder_backends_registry.close()
            close_replay = getattr(fldigi_replay, "close", None)
            if callable(close_replay):
                close_replay()
            decode_executor.shutdown(wait=True, cancel_futures=True)
            model_training_executor.shutdown(wait=False, cancel_futures=True)
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
    app.state.decoder_backends = decoder_backends_registry
    app.state.classifier = classifier
    app.state.classifier_error = classifier_error
    app.state.classifier_model_path = str(Path(configured_model).expanduser().resolve()) if configured_model else None
    app.state.cv_detector = cv_detector
    app.state.cv_detector_error = cv_detector_error
    app.state.cv_model_path = configured_cv_model
    app.state.model_training_job = {"status": "idle", "progress": 0, "stage": "", "error": None, "result": None}
    app.state.model_training_lock = model_training_lock
    app.state.model_training_executor = model_training_executor
    app.state.cv_embedding_vectors = cv_embedding_vectors
    app.state.cv_embedding_entries = cv_embedding_entries
    app.state.cv_embedding_index_path = configured_cv_index
    app.state.cv_embedding_error = cv_embedding_error
    app.state.band_hop_threads = {}
    app.state.band_hop_lock = Lock()
    app.state.laya_references = load_reference_examples()
    app.state.live_track_registry = None
    app.state.fldigi_trials = FldigiTrialRunner(fldigi, run_store=store)

    def run_region_replay(
        run_id: str, settings: FldigiReplayRequest, sample_rate: int,
        sample_reader: Any,
    ) -> None:
        temporary_audio: Path | None = None
        attempts: list[dict[str, Any]] = []
        modes = settings.candidate_modes if settings.auto_identify else [settings.mode]
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
                "attempted_modes": [], "current_mode": modes[0] if modes else settings.mode,
                "output_offset": 0, "output_end": None, "text": "",
                "status": "running", "note": "Preparing the selected sample window for isolated receive replay.",
            })
            for index, candidate in enumerate(modes):
                app.state.capture_store.update_decoder_run(run_id, {
                    "attempted_modes": [item["mode"] for item in attempts],
                    "current_mode": candidate, "output_offset": 0, "output_end": None,
                    "text": "", "status": "running",
                    "note": ("Checking for receive-side Fldigi RSID/TXID, then testing candidate text decode."
                             if settings.auto_identify and index == 0 else f"Testing {candidate} against the selected audio."),
                })
                replay_options = {
                    "mode": candidate,
                    "carrier_hz": settings.carrier_hz,
                }
                if settings.auto_identify:
                    replay_options["detect_rsid"] = index == 0
                result = fldigi_replay.decode(temporary_audio, **replay_options)
                text = str(result.get("text", ""))
                result_mode = str(result.get("mode") or candidate)
                attempts.append({
                    "mode": result_mode, "text": text,
                    "score": _decode_text_coherence(text),
                    "rsid_detected": bool(result.get("rsid_detected")),
                })
                if result.get("rsid_detected"):
                    break
            best = max(attempts, key=lambda item: item["score"], default={"mode": settings.mode, "text": "", "score": 0.0})
            text = str(best["text"])
            attempted_modes = [item["mode"] for item in attempts]
            if settings.auto_identify:
                summary = "; ".join(f"{item['mode']} {item['score']:.1f}" for item in attempts)
                if any(item["rsid_detected"] for item in attempts):
                    note = f"Fldigi receive-side RSID identified {best['mode']}; tentative decode score {best['score']:.1f}."
                elif text:
                    note = f"Best text-coherence candidate: {best['mode']} ({best['score']:.1f}/100). Candidate scores: {summary}. Review decoded text; no label was assigned."
                else:
                    note = f"No candidate produced text. Tested: {summary or 'none'}."
            else:
                note = f"{result.get('note', '')} Carrier setting: {settings.carrier_hz if settings.carrier_hz is not None else 'unset'} Hz."
            app.state.capture_store.update_decoder_run(run_id, {
                "attempted_modes": attempted_modes, "current_mode": best["mode"],
                "output_offset": 0, "output_end": len(text.encode("utf-8")), "text": text,
                "status": "tentative_output" if settings.auto_identify and text else (
                    "completed_no_text" if settings.auto_identify else result.get("status", "completed")
                ),
                "note": note,
            })
        except Exception as exc:
            logging.getLogger(__name__).exception("Isolated Fldigi region replay failed")
            try:
                app.state.capture_store.update_decoder_run(run_id, {
                    "attempted_modes": [item["mode"] for item in attempts],
                    "current_mode": attempts[-1]["mode"] if attempts else (modes[0] if modes else settings.mode),
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

    @app.post("/api/rigctl/frequency")
    def read_rigctl_frequency(settings: RigctlFrequencyRequest) -> dict[str, Any]:
        executable = shutil.which(os.environ.get("SIG_ID_RIGCTL", "rigctl"))
        if not executable:
            return {"available": False, "connected": False, "error": "Hamlib rigctl was not found on the server."}
        try:
            result = subprocess.run(
                [executable, "-m", "2", "-r", f"{settings.host}:{settings.port}", "f"],
                capture_output=True, text=True, timeout=3, check=False,
            )
        except subprocess.TimeoutExpired:
            return {"available": True, "connected": False, "error": "rigctl timed out while querying rigctld."}
        except OSError as exc:
            return {"available": True, "connected": False, "error": f"Could not run rigctl: {exc}"}
        output = result.stdout.strip().splitlines()
        if result.returncode != 0 or not output:
            detail = result.stderr.strip() or result.stdout.strip() or f"rigctl exited with status {result.returncode}"
            return {"available": True, "connected": False, "error": detail[-500:]}
        try:
            frequency_hz = float(output[0].strip())
        except ValueError:
            return {"available": True, "connected": False, "error": f"Unexpected rigctl frequency response: {output[0][:120]}"}
        if not np.isfinite(frequency_hz) or frequency_hz <= 0:
            return {"available": True, "connected": False, "error": f"rigctl returned an invalid frequency: {output[0][:120]}"}
        return {
            "available": True, "connected": True, "frequency_hz": frequency_hz,
            "host": settings.host, "port": settings.port,
        }

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
            "classifier_window_seconds": metadata.get("classifier_window_seconds"),
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

    def _run_incremental_model_training() -> None:
        job = app.state.model_training_job
        release = root / "models" / "releases" / datetime.now(timezone.utc).strftime("incremental-%Y%m%dT%H%M%SZ")
        try:
            if app.state.classifier is None or not app.state.classifier_model_path:
                raise RuntimeError("Configure a classifier model before starting in-app training")
            if app.state.cv_detector is None or not app.state.cv_model_path:
                raise RuntimeError("Configure a CV detector before starting in-app training")
            release.mkdir(parents=True, exist_ok=False)

            # Resolve the exact synthetic split used to train this classifier by
            # its recorded digest, then find the dataset store containing its IDs.
            job.update(status="running", progress=2, stage="Finding classifier training sources", error=None)
            classifier_meta_path = Path(app.state.classifier_model_path).expanduser().resolve()
            classifier_meta = app.state.classifier["metadata"]
            wanted_split_hash = classifier_meta.get("split_manifest_sha256")
            repo_root = Path(__file__).resolve().parents[1]
            split_path = None
            recorded_split = classifier_meta.get("split_manifest_path")
            if isinstance(recorded_split, str) and Path(recorded_split).is_file():
                candidate = Path(recorded_split).expanduser().resolve()
                if hashlib.sha256(candidate.read_bytes()).hexdigest() == wanted_split_hash:
                    split_path = candidate
            for candidate in (repo_root / "data").rglob("*.json"):
                if split_path is not None:
                    break
                try:
                    if hashlib.sha256(candidate.read_bytes()).hexdigest() == wanted_split_hash:
                        split_path = candidate
                        break
                except OSError:
                    continue
            if split_path is None:
                raise RuntimeError("Could not find the classifier's original split manifest; set up its training sources first")
            split_payload = json.loads(split_path.read_text(encoding="utf-8"))
            required_ids = {row.get("capture_id") for row in split_payload.get("records", [])}
            classifier_data_dir = None
            recorded_data_dir = classifier_meta.get("classifier_data_dir")
            if isinstance(recorded_data_dir, str) and (Path(recorded_data_dir).expanduser() / "dataset.sqlite3").is_file():
                candidate_store = CaptureStore(Path(recorded_data_dir).expanduser() / "dataset.sqlite3")
                if required_ids.issubset({row["id"] for row in candidate_store.list_captures()}):
                    classifier_data_dir = Path(recorded_data_dir).expanduser()
            for database in (repo_root / "data").rglob("dataset.sqlite3"):
                if classifier_data_dir is not None:
                    break
                try:
                    candidate_store = CaptureStore(database)
                    if required_ids.issubset({row["id"] for row in candidate_store.list_captures()}):
                        classifier_data_dir = database.parent
                        break
                except Exception:
                    continue
            if classifier_data_dir is None:
                raise RuntimeError("Could not find the classifier's original training dataset")
            previous_reviewed_examples: set[tuple[str, str, int, str | None, int, int]] = set()
            reviewed_manifest_name = classifier_meta.get("reviewed_training_manifest")
            if isinstance(reviewed_manifest_name, str):
                reviewed_manifest = classifier_meta_path.with_name(reviewed_manifest_name)
                if reviewed_manifest.is_file():
                    expected = classifier_meta.get("reviewed_training_manifest_sha256")
                    if expected and hashlib.sha256(reviewed_manifest.read_bytes()).hexdigest() != expected:
                        raise RuntimeError("Classifier reviewed-training manifest failed its hash check")
                    previous_reviewed_examples = {
                        (row["capture_id"], row["region_id"], int(row.get("window_start", 0)),
                         row.get("mode"), int(row.get("sample_start", 0)), int(row.get("sample_count", 0)))
                        for row in json.loads(reviewed_manifest.read_text(encoding="utf-8")).get("examples", [])
                        if isinstance(row.get("capture_id"), str) and isinstance(row.get("region_id"), str)
                    }

            cv_meta_path = Path(app.state.cv_model_path).expanduser().resolve().with_suffix(
                Path(app.state.cv_model_path).suffix + ".json"
            )
            cv_meta = json.loads(cv_meta_path.read_text(encoding="utf-8"))
            old_dataset = Path(cv_meta["dataset_path"]).expanduser()
            old_report = json.loads((old_dataset / "export-report.json").read_text(encoding="utf-8"))
            reviewed_data_dirs = [Path(path).expanduser() for path in old_report.get("source_data_dirs", [])]
            reviewed_data_dirs = [path for path in reviewed_data_dirs if (path / "dataset.sqlite3").is_file()]
            if root not in reviewed_data_dirs:
                reviewed_data_dirs.append(root)

            job.update(progress=5, stage="Preparing classifier training data")
            from .hierarchical_training import train_hierarchical
            classifier_output = release / "classifier.json"
            split_copy = release / "classifier-splits.json"
            shutil.copy2(split_path, split_copy)
            classifier_kwargs: dict[str, Any] = {}
            if os.environ.get("SIG_ID_OOD_CORPUS"):
                classifier_kwargs["ood_corpus"] = os.environ["SIG_ID_OOD_CORPUS"]
            elif classifier_meta.get("ood_corpus_path") and Path(classifier_meta["ood_corpus_path"]).is_file():
                classifier_kwargs["ood_corpus"] = classifier_meta["ood_corpus_path"]
            elif classifier_meta.get("ood_validation_sources", 0):
                from .ood_corpus import load_ood_corpus
                for candidate in (repo_root / "data").rglob("ood-corpus.json"):
                    try:
                        validation_count = sum(row.get("split") == "validation" for row in load_ood_corpus(candidate))
                        if validation_count == classifier_meta["ood_validation_sources"]:
                            classifier_kwargs["ood_corpus"] = candidate
                            break
                    except Exception:
                        continue
            classifier_metadata = train_hierarchical(
                split_copy, classifier_data_dir, classifier_output,
                reviewed_data_dirs=reviewed_data_dirs,
                initial_model_path=classifier_meta_path,
                epochs=int(os.environ.get("SIG_ID_INCREMENTAL_CLASSIFIER_EPOCHS", "1200")),
                feature_version=classifier_meta["feature_version"],
                classifier_window_seconds=float(classifier_meta.get("classifier_window_seconds", 2.0)),
                progress_callback=lambda fraction, detail: job.update(
                    progress=5 + int(43 * fraction), stage=f"Training classifier · {detail}",
                ),
                **classifier_kwargs,
            )
            reviewed_manifest_name = classifier_metadata.get("reviewed_training_manifest")
            if not isinstance(reviewed_manifest_name, str) or Path(reviewed_manifest_name).name != reviewed_manifest_name:
                raise RuntimeError("Classifier trainer did not produce a valid reviewed-training manifest name")
            new_reviewed_manifest = classifier_output.with_name(reviewed_manifest_name)
            if not new_reviewed_manifest.is_file():
                raise RuntimeError("Classifier trainer did not produce its reviewed-training manifest")
            expected_new_manifest_hash = classifier_metadata.get("reviewed_training_manifest_sha256")
            if not isinstance(expected_new_manifest_hash, str) or hashlib.sha256(
                new_reviewed_manifest.read_bytes()
            ).hexdigest() != expected_new_manifest_hash:
                raise RuntimeError("New classifier reviewed-training manifest failed its hash check")
            new_reviewed_payload = json.loads(new_reviewed_manifest.read_text(encoding="utf-8"))
            if not isinstance(new_reviewed_payload, dict):
                raise RuntimeError("New classifier reviewed-training manifest is invalid")
            new_reviewed_examples = new_reviewed_payload.get("examples", [])
            if not isinstance(new_reviewed_examples, list):
                raise RuntimeError("New classifier reviewed-training manifest has an invalid examples list")
            new_reviewed = [
                row for row in new_reviewed_examples
                if isinstance(row, dict)
                and (row.get("capture_id"), row.get("region_id"), int(row.get("window_start", 0)),
                    row.get("mode"), int(row.get("sample_start", 0)), int(row.get("sample_count", 0)))
                not in previous_reviewed_examples
            ]
            new_capture_ids = {row.get("capture_id") for row in new_reviewed}
            if not new_reviewed:
                raise RuntimeError("No new, fully reviewed labeled live captures are eligible for classifier training")

            job.update(progress=50, stage="Building reviewed CV dataset")
            source_dirs = list(reviewed_data_dirs)
            if root not in source_dirs:
                source_dirs.append(root)
            if not source_dirs:
                raise RuntimeError("CV training source stores are unavailable")
            from .cv_dataset import export_cv_dataset
            from .cv_training import train_cv_detector
            cv_dataset = release / "cv-dataset"
            export_cv_dataset(source_dirs[0], cv_dataset, seed=int(os.environ.get("SIG_ID_INCREMENTAL_CV_SEED", "4192026")),
                              additional_data_dirs=[str(path) for path in source_dirs[1:]])
            job.update(progress=58, stage="Training CV detector")
            cv_model_path = release / "cv-signal.pt"
            train_cv_detector(
                cv_dataset, output_model=cv_model_path,
                base_model=app.state.cv_model_path,
                epochs=int(os.environ.get("SIG_ID_INCREMENTAL_CV_EPOCHS", "30")),
                batch=int(os.environ.get("SIG_ID_INCREMENTAL_CV_BATCH", "8")),
                device=os.environ.get("SIG_ID_INCREMENTAL_CV_DEVICE") or None,
                progress_callback=lambda epoch, total: job.update(
                    progress=58 + int(30 * epoch / max(1, total)),
                    stage=f"Training CV detector · epoch {epoch}/{total}",
                ),
            )

            # Load and validate both artifacts before atomically changing the
            # durable pointer or either in-memory model.
            job.update(progress=90, stage="Validating paired models")
            new_classifier_meta, new_classifier_parameters = load_model(classifier_output)
            new_detector = YoloSignalDetector(cv_model_path)
            pointer = {"classifier": str(classifier_output.resolve()), "cv_detector": str(cv_model_path.resolve())}
            active_models_file.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=active_models_file.parent, delete=False) as stream:
                json.dump(pointer, stream, indent=2)
                stream.write("\n")
                temporary_pointer = Path(stream.name)
            temporary_pointer.replace(active_models_file)
            app.state.classifier = {"metadata": new_classifier_meta, "parameters": new_classifier_parameters}
            app.state.classifier_model_path = pointer["classifier"]
            app.state.classifier_error = None
            app.state.cv_detector = new_detector
            app.state.cv_detector_error = None
            app.state.cv_model_path = pointer["cv_detector"]
            app.state.cv_embedding_vectors = None
            app.state.cv_embedding_entries = None
            app.state.cv_embedding_index_path = None
            app.state.cv_embedding_error = "Embedding index needs rebuilding for the newly trained CV model"
            job.update(status="complete", progress=100, stage="Active", error=None, result={
                "classifier": pointer["classifier"], "cv_detector": pointer["cv_detector"],
                "new_classifier_examples": len(new_reviewed),
                "new_classifier_captures": len(new_capture_ids),
                "cv_images": json.loads((cv_dataset / "export-report.json").read_text(encoding="utf-8")).get("split_images"),
            })
        except Exception as exc:
            logging.getLogger(__name__).exception("In-app incremental model training failed")
            job.update(status="failed", stage="failed", error=str(exc), result=None)
        finally:
            model_training_lock.release()

    @app.get("/api/models/training/status")
    def model_training_status() -> dict[str, Any]:
        return dict(app.state.model_training_job)

    @app.post("/api/models/train", status_code=202)
    def start_model_training() -> dict[str, Any]:
        if not model_training_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Model training is already running")
        app.state.model_training_job = {"status": "running", "progress": 0, "stage": "Queued", "error": None, "result": None}
        try:
            model_training_executor.submit(_run_incremental_model_training)
        except Exception:
            model_training_lock.release()
            raise
        return dict(app.state.model_training_job)

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
            "active_models_file": str(active_models_file),
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

    @app.get("/api/library/signals")
    def list_library_signals() -> list[dict[str, Any]]:
        return store.list_library_signals()

    @app.get("/api/audio/devices")
    def input_devices() -> list[dict[str, Any]]:
        try:
            return live_capture.devices()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/live")
    def live_status() -> dict[str, Any]:
        return live_capture.status()

    @app.get("/api/live/capabilities")
    def live_capabilities() -> dict[str, bool]:
        return {"band_hopping": True}

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

    @app.get("/api/live/cv-detections")
    def live_cv_detections(
        session_id: str = Query(min_length=1, max_length=128),
        confidence: float | None = Query(default=None, ge=0.0, le=1.0),
        include_classifier: bool = Query(default=True),
    ) -> dict[str, Any]:
        detector = app.state.cv_detector
        if detector is None:
            raise HTTPException(status_code=503, detail="CV detector is unavailable")
        snapshot = live_capture.analysis_audio()
        if snapshot is None or not snapshot["active"] or snapshot["session_id"] != session_id:
            raise HTTPException(status_code=409, detail="live capture session is no longer active")
        channels = int(snapshot["channels"])
        pcm = np.frombuffer(snapshot["pcm"], dtype="<i2").reshape(-1, channels)
        mono = pcm.astype(np.float64).mean(axis=1) / 32768.0
        if len(mono) < FFT_SIZE:
            return {"session_id": session_id, "warming_up": True, "observed_seconds": len(mono) / snapshot["sample_rate"], "detections": []}
        try:
            detections = detector.detect_live_partial(
                mono, sample_start=int(snapshot["sample_offset"]),
                total_sample_count=int(snapshot["sample_count"]),
                sample_rate=int(snapshot["sample_rate"]), confidence=confidence,
            )
        except Exception as exc:
            logging.getLogger(__name__).exception("Live CV detection failed")
            raise HTTPException(status_code=503, detail=f"live CV detection failed: {exc}") from exc
        classifier = app.state.classifier
        classifier_available = classifier is not None
        if classifier is not None and include_classifier:
            sample_rate = int(snapshot["sample_rate"])
            model_window_seconds = float(
                classifier["metadata"].get("classifier_window_seconds") or CLASSIFIER_WINDOW_SECONDS
            )
            target_count = max(1, round(model_window_seconds * sample_rate))
            ring_start = int(snapshot["sample_offset"])
            max_scored = set(id(item) for item in sorted(
                detections, key=lambda item: float(item["confidence"]), reverse=True
            )[:32])
            for detection in detections:
                if id(detection) not in max_scored:
                    detection["classifier_proposal"] = {
                        "available": False, "reason": "32-box live classification limit reached",
                    }
                    continue
                local_start = max(0, int(detection["sample_start"]) - ring_start)
                local_end = min(len(mono), local_start + int(detection["sample_count"]))
                if len(mono) < target_count:
                    detection["classifier_proposal"] = {
                        "available": False,
                        "reason": f"waiting for a {model_window_seconds:g}-second classifier crop",
                    }
                    continue
                midpoint = (local_start + local_end) // 2
                crop_start = max(0, min(len(mono) - target_count, midpoint - target_count // 2))
                crop = mono[crop_start : crop_start + target_count]
                try:
                    proposal = _classify_signal_crop(
                        classifier, crop, sample_rate,
                        float(detection["frequency_low_hz"]),
                        float(detection["frequency_high_hz"]),
                    )
                    detection["classifier_proposal"] = {
                        "sample_start": ring_start + crop_start,
                        "sample_count": target_count,
                        **proposal,
                    }
                except Exception as exc:
                    logging.getLogger(__name__).exception("Live CV classifier proposal failed")
                    detection["classifier_proposal"] = {"available": False, "reason": str(exc)}
        return {
            "session_id": session_id, "warming_up": False,
            "observed_seconds": len(mono) / snapshot["sample_rate"],
            "window_seconds": WINDOW_SECONDS, "padding": "leading_noise",
            "model_sha256": detector.model_sha256,
            "preprocessing_version": PREPROCESSING_VERSION,
            "classifier_available": classifier_available,
            "classifier_scoring": classifier is not None and include_classifier,
            "detections": detections,
        }

    @app.get("/api/live/analysis")
    def live_analysis(
        fft_size: int = Query(default=4096, ge=32, le=8192),
        hop_size: int = Query(default=512, ge=1, le=8192),
        max_rows: int = Query(default=640, ge=1, le=8192),
        threshold_db: float = Query(default=18.0, ge=1.0, le=60.0),
        classifier_crop_count: int = Query(default=1, ge=1, le=20),
        include_tracks: bool = Query(default=True),
        include_classifier: bool = Query(default=True),
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
                "sample_offset": snapshot["sample_offset"],
                "sample_rate": snapshot["sample_rate"],
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
            tracks = []
            if include_tracks:
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
            model = app.state.classifier if include_classifier else None
            if include_tracks and model is not None and snapshot["sample_rate"] >= 8000:
                # Score bounded 2-second crops across the candidate's time and
                # frequency area. The user's crop count controls the consensus.
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
                    if segment_end - segment_start < 1024:
                        track["model_proposal"] = {
                            "available": False,
                            "reason": "candidate has less than 1024 in-ring samples",
                        }
                        continue
                    selected = mono[segment_start - ring_start : segment_end - ring_start]
                    sample_rate = int(snapshot["sample_rate"])
                    model_window_seconds = float(model["metadata"].get("classifier_window_seconds") or CLASSIFIER_WINDOW_SECONDS)
                    target_count = max(1, round(model_window_seconds * sample_rate))
                    if len(selected) > target_count:
                        last_start = len(selected) - target_count
                        available_crops = max(1, len(selected) // target_count)
                        offsets = np.asarray([last_start]) if classifier_crop_count == 1 else (
                            np.linspace(0, available_crops - 1, min(classifier_crop_count, available_crops))
                            .round().astype(int) * target_count
                        )
                        crop_offsets = list(dict.fromkeys(int(offset) for offset in offsets))
                    else:
                        crop_offsets = [0]
                    try:
                        crop_results = []
                        for offset in crop_offsets:
                            crop_values = selected[offset : offset + target_count]
                            features = extract_features(
                                crop_values,
                                sample_rate,
                                feature_version=model["metadata"]["feature_version"],
                                classifier_window_seconds=model_window_seconds,
                                frequency_range_hz=(
                                    (float(track["frequency_low_hz"]), float(track["frequency_high_hz"]))
                                    if model["metadata"]["feature_version"] in ROI_FEATURE_VERSIONS
                                    and track.get("frequency_low_hz") is not None
                                    and track.get("frequency_high_hz") is not None
                                    else None
                                ),
                            )
                            crop_results.append({
                                "sample_start": segment_start + offset,
                                "sample_count": target_count,
                                **propose_loaded(features, model["metadata"], model["parameters"]),
                            })
                        proposal = max(crop_results, key=lambda item: item["confidence"]).copy()
                        method_votes: dict[str, list[dict[str, Any]]] = {}
                        family_votes: dict[str, list[dict[str, Any]]] = {}
                        for result in crop_results:
                            if result.get("method"):
                                method_votes.setdefault(result["method"], []).append(result)
                            if result.get("family"):
                                family_votes.setdefault(result["family"], []).append(result)
                        majority = len(crop_results) // 2 + 1
                        method_consensus = next(((name, votes) for name, votes in method_votes.items() if len(votes) >= majority), (None, []))
                        family_consensus = next(((name, votes) for name, votes in family_votes.items() if len(votes) >= majority), (None, []))
                        if method_consensus[0]:
                            proposal.update(method_consensus[1][0])
                            proposal["method"] = method_consensus[0]
                            proposal["family"] = method_consensus[1][0]["family"]
                            proposal["consensus_label"] = method_consensus[0]
                            proposal["consensus_count"] = len(method_consensus[1])
                            proposal["confidence"] = float(np.mean([item["confidence"] for item in method_consensus[1]]))
                        elif family_consensus[0]:
                            proposal["method"] = None
                            proposal["method_id"] = None
                            proposal["family"] = family_consensus[0]
                            proposal["family_abstained"] = False
                            proposal["abstained"] = True
                            proposal["consensus_label"] = family_consensus[0]
                            proposal["consensus_count"] = len(family_consensus[1])
                            proposal["confidence"] = float(np.mean([item["confidence"] for item in family_consensus[1]]))
                        else:
                            proposal["method"] = None
                            proposal["method_id"] = None
                            proposal["family"] = None
                            proposal["family_abstained"] = True
                            proposal["abstained"] = True
                            proposal["consensus_label"] = None
                            proposal["consensus_count"] = 0
                        proposal["crop_count"] = len(crop_results)
                        proposal["requested_crop_count"] = classifier_crop_count
                        proposal["crop_results"] = [
                            {key: result.get(key) for key in ("sample_start", "sample_count", "method", "method_id", "family", "confidence", "top_candidate", "novelty_rejected")}
                            for result in crop_results
                        ]
                        track["model_proposal"] = {
                            "available": True,
                            **proposal,
                            "model_sha256": model["metadata"]["model_sha256"],
                            "sample_start": crop_results[0]["sample_start"],
                            "sample_count": target_count,
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
            if auto_trial and include_tracks and include_classifier and model is not None:
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
                "sample_offset": snapshot["sample_offset"],
                "sample_count": snapshot["sample_count"],
                "window_sample_count": snapshot["window_sample_count"],
                "sample_rate": snapshot["sample_rate"],
                "spectrum": spectrum if include_spectrum else None,
                "tracks": tracks,
                "automatic_trial": automatic_trial,
                "analysis": {"threshold_db": threshold_db, "method": "row-median threshold with persistent peak clusters"},
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/live/training-examples", status_code=201)
    def save_live_training_example(data: LiveTrainingAnnotation) -> dict[str, Any]:
        snapshot = live_capture.analysis_audio()
        if snapshot is None or not snapshot["active"] or snapshot["session_id"] != data.session_id:
            raise HTTPException(status_code=409, detail="that live capture is no longer active")
        rate = int(snapshot["sample_rate"])
        window_count = rate * WINDOW_SECONDS
        if int(snapshot["window_sample_count"]) < window_count:
            raise HTTPException(status_code=409, detail="wait until a complete 10-second live window is available")
        ring_start = int(snapshot["sample_offset"])
        ring_end = ring_start + int(snapshot["window_sample_count"])
        if data.sample_start < ring_start or data.sample_start + data.sample_count > ring_end:
            raise HTTPException(status_code=422, detail="the selection has moved outside the current 10-second live window; redraw it")
        if data.frequency_high_hz > rate / 2 or data.frequency_low_hz >= data.frequency_high_hz:
            raise HTTPException(status_code=422, detail="frequency bounds must fit below the live audio Nyquist frequency")
        if int(snapshot["window_sample_count"]) != window_count:
            pcm = snapshot["pcm"][-window_count * int(snapshot["channels"]) * 2:]
            ring_start = ring_end - window_count
        else:
            pcm = snapshot["pcm"]
        try:
            resolved = resolve_mode_label(data.mode_id, {
                "family": data.family, "method": data.method,
                "protocol": data.protocol, "variant": data.variant,
            }, additional_modes=store.accepted_mode_definitions())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        capture_id = str(uuid4())
        audio_path = capture_dir / f"{capture_id}.wav"
        try:
            with wave.open(str(audio_path), "wb") as wav:
                wav.setnchannels(int(snapshot["channels"]))
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(pcm)
            new_capture_id = store.add_wav_capture(
                audio_path, source_kind="live",
                source_metadata={
                    "auto_review_queue": True,
                    "window_seconds": WINDOW_SECONDS,
                    "auto_capture_session_id": data.session_id,
                    "window_start_sample": ring_start,
                    "created_from_live_annotation": True,
                    "training_candidate_requires_coverage_review": True,
                },
            )
            region_id = store.add_region(
                new_capture_id,
                sample_start=data.sample_start - ring_start,
                sample_count=data.sample_count,
                frequency_low_hz=data.frequency_low_hz,
                frequency_high_hz=data.frequency_high_hz,
                label=SignalLabel(
                    family=resolved["family"], method=resolved["method"],
                    protocol=resolved["protocol"], variant=resolved["variant"],
                    mode_id=resolved["mode_id"], source="manual", review_state="pending",
                    notes=data.notes,
                    evidence={"source": "live_classify", "live_session_id": data.session_id,
                              "window_start_sample": ring_start, "registry_version": resolved["registry_version"]},
                ),
                origin="manual",
            )
            return {"capture_id": new_capture_id, "region_id": region_id,
                    "sample_rate": rate, "sample_count": window_count,
                    "message": "Saved a 10-second training candidate. Review full-window coverage before export."}
        except (ValueError, OSError) as exc:
            audio_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/live/start")
    def start_live_capture(data: LiveCaptureStart) -> dict[str, Any]:
        if data.capture_mode == "unattended" and app.state.cv_detector is None:
            if app.state.cv_detector_error:
                detail = f"CV detector failed to load: {app.state.cv_detector_error}"
            else:
                detail = f"No CV detector is loaded; checked active model configuration at {active_models_file}"
            raise HTTPException(status_code=409, detail=detail)

        def analyze_unattended_window(samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
            detector = app.state.cv_detector
            if detector is None:
                return {"detected": False, "cv_detection_count": 0}
            image = spectrogram_image(samples, sample_rate)
            detections = detector.detect(
                image, sample_start=0, sample_count=len(samples), sample_rate=sample_rate,
            )
            methods: list[str] = []
            classifier = app.state.classifier
            if classifier is not None:
                target = max(1, round(float(classifier["metadata"].get("classifier_window_seconds") or CLASSIFIER_WINDOW_SECONDS) * sample_rate))
                for detection in detections[:4]:
                    local_start = max(0, int(detection["sample_start"]))
                    local_end = min(len(samples), local_start + int(detection["sample_count"]))
                    if local_end - local_start < target:
                        midpoint = (local_start + local_end) // 2
                        local_start = max(0, min(len(samples) - target, midpoint - target // 2))
                        local_end = min(len(samples), local_start + target)
                    if local_end - local_start < target:
                        continue
                    features = extract_features(
                        samples[local_start:local_start + target], sample_rate,
                        feature_version=classifier["metadata"]["feature_version"],
                        classifier_window_seconds=float(classifier["metadata"].get("classifier_window_seconds") or CLASSIFIER_WINDOW_SECONDS),
                        frequency_range_hz=(
                            (float(detection["frequency_low_hz"]), float(detection["frequency_high_hz"]))
                            if classifier["metadata"].get("feature_version") in ROI_FEATURE_VERSIONS else None
                        ),
                    )
                    proposal = propose_loaded(features, classifier["metadata"], classifier["parameters"])
                    if proposal.get("method"):
                        methods.append(str(proposal["method"]))
            return {
                "detected": bool(detections),
                "cv_detection_count": len(detections),
                "top_confidences": sorted((float(item["confidence"]) for item in detections), reverse=True),
                "classifier_methods": methods,
            }

        try:
            frequencies = data.band_hop_frequencies_hz
            if frequencies:
                if live_capture.status().get("active"):
                    raise HTTPException(status_code=409, detail="a live capture is already active")
                try:
                    tune_rigctl_frequency(data.rigctl_host, data.rigctl_port, frequencies[0])
                except RuntimeError as exc:
                    raise HTTPException(status_code=503, detail=f"Could not start band hopping: {exc}") from exc
            started = live_capture.start(
                device_id=data.device_id,
                sample_rate=data.sample_rate,
                channels=data.channels,
                max_duration_seconds=data.max_duration_seconds,
                capture_mode=data.capture_mode,
                noise_interval_seconds=data.noise_interval_seconds,
                window_analyzer=analyze_unattended_window if data.capture_mode == "unattended" else None,
                initial_frequency_hz=frequencies[0] if frequencies else None,
                band_hop_count=len(frequencies),
                band_hop_dwell_seconds=data.band_hop_dwell_seconds if frequencies else None,
            )
            if frequencies:
                dwell = data.band_hop_dwell_seconds
                stop_hopping = threading.Event()

                def band_hop_worker() -> None:
                    index = 0
                    try:
                        while not stop_hopping.wait(dwell):
                            status = live_capture.status()
                            if not status.get("active") or status.get("session_id") != started["session_id"]:
                                return
                            index = (index + 1) % len(frequencies)
                            frequency_hz = frequencies[index]
                            try:
                                tune_rigctl_frequency(data.rigctl_host, data.rigctl_port, frequency_hz)
                            except RuntimeError as exc:
                                current = live_capture.status()
                                live_capture.set_band_hop_frequency(
                                    started["session_id"], int(current.get("current_frequency_hz") or frequencies[0]),
                                    int(current.get("band_hop_index") or 0), len(frequencies), dwell, str(exc),
                                )
                                logging.getLogger(__name__).warning("Band hop to %s Hz failed: %s", frequency_hz, exc)
                                continue
                            if not live_capture.set_band_hop_frequency(
                                started["session_id"], frequency_hz, index, len(frequencies), dwell,
                            ):
                                return
                    finally:
                        with app.state.band_hop_lock:
                            existing = app.state.band_hop_threads.get(started["session_id"])
                            if existing and existing[0] is stop_hopping:
                                app.state.band_hop_threads.pop(started["session_id"], None)

                thread = threading.Thread(
                    target=band_hop_worker, name=f"sig-id-band-hop-{started['session_id'][:8]}", daemon=True,
                )
                with app.state.band_hop_lock:
                    app.state.band_hop_threads[started["session_id"]] = (stop_hopping, thread)
                thread.start()
                started["current_frequency_hz"] = frequencies[0]
                started["band_hop_index"] = 0
                started["band_hop_count"] = len(frequencies)
                started["band_hop_dwell_seconds"] = dwell
            app.state.live_track_registry = LiveTrackRegistry(started["session_id"])
            return started
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"could not start audio input: {exc}") from exc

    @app.post("/api/live/stop")
    def stop_live_capture() -> dict[str, Any]:
        try:
            status = live_capture.status()
            with app.state.band_hop_lock:
                hopping = app.state.band_hop_threads.get(status.get("session_id"))
            if hopping:
                hopping[0].set()
                hopping[1].join(timeout=5)
            result = live_capture.stop()
            app.state.fldigi_trials.cancel()
            if result.get("capture"):
                store.associate_decoder_runs(result["session_id"], result["capture"]["id"])
            return result
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/live/review-queue")
    def live_review_queue() -> dict[str, Any]:
        pending = []
        manually_reviewed = []
        coverage_reviewed = 0
        re_review_due = 0
        now = datetime.now(timezone.utc)
        for capture in store.list_captures():
            if not capture["source_metadata"].get("auto_review_queue"):
                continue
            manual_reviewed_at = capture["source_metadata"].get("auto_review_queue_reviewed_utc")
            review_start = capture["source_metadata"].get("review_window_start_sample", 0)
            review_count = capture["source_metadata"].get("review_window_sample_count", capture["sample_count"])
            records = store.scene_coverages(capture["id"])
            valid_reviews = [record for record in records if (
                not record["is_stale"] and record["task"] == "detection"
                and record["sample_start"] <= review_start
                and record["sample_start"] + record["sample_count"] >= review_start + review_count
                and record["state"] in {"complete", "partial"}
                and (record["frequency_low_hz"] is None
                     or (record["frequency_low_hz"] <= 0 and record["frequency_high_hz"] >= MAX_FREQUENCY_HZ))
            )]
            complete_review = bool(valid_reviews)
            if manual_reviewed_at:
                last_reviewed_at = manual_reviewed_at
            elif complete_review:
                last_reviewed_at = max(record["created_utc"] for record in valid_reviews)
            else:
                last_reviewed_at = None
            review_is_due = False
            if last_reviewed_at:
                try:
                    reviewed_utc = datetime.fromisoformat(last_reviewed_at.replace("Z", "+00:00"))
                    review_is_due = now - reviewed_utc >= RE_REVIEW_INTERVAL
                except (TypeError, ValueError):
                    review_is_due = True
            if review_is_due:
                re_review_due += 1
                pending.append({**capture, "source_metadata": {
                    **capture["source_metadata"], "re_review_due": True,
                    "last_reviewed_utc": last_reviewed_at,
                }})
                continue
            if manual_reviewed_at:
                manually_reviewed.append(capture)
                continue
            if complete_review:
                coverage_reviewed += 1
            else:
                pending.append(capture)
        pending.sort(key=lambda item: item["created_utc"])
        manually_reviewed.sort(key=lambda item: item["source_metadata"].get("auto_review_queue_reviewed_utc", ""), reverse=True)
        return {
            "pending": pending,
            "pending_count": len(pending),
            "reviewed_count": coverage_reviewed + len(manually_reviewed),
            "re_review_due_count": re_review_due,
            "re_review_interval_days": RE_REVIEW_INTERVAL.days,
            "manually_reviewed": manually_reviewed,
        }

    def hub_request(path: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> tuple[bytes, str]:
        base_url = os.environ.get("SIG_ID_HUB_URL", "").rstrip("/")
        token = os.environ.get("SIG_ID_HUB_REVIEWER_TOKEN", "")
        if not base_url or not token:
            raise HTTPException(status_code=503, detail="Hub review is not configured; set SIG_ID_HUB_URL and SIG_ID_HUB_REVIEWER_TOKEN for the local workbench service")
        request_headers = {"Authorization": f"Bearer {token}"}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        hub_request_obj = urllib.request.Request(base_url + path, data=body, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(hub_request_obj, timeout=8) as response:
                return response.read(), response.headers.get("Content-Type", "application/octet-stream")
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", errors="replace")
            raise HTTPException(status_code=exc.code, detail=f"Hub request failed: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise HTTPException(status_code=502, detail=f"Hub is unavailable: {exc}") from exc

    @app.get("/api/hub/review-queue")
    def hub_review_queue() -> list[dict[str, Any]]:
        payload, _ = hub_request("/api/review/queue")
        try:
            result = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=502, detail="Hub returned invalid review queue data") from exc
        if not isinstance(result, list):
            raise HTTPException(status_code=502, detail="Hub returned an unexpected review queue")
        return result

    @app.get("/api/hub/submissions/{submission_id}/audio")
    def hub_review_audio(submission_id: str) -> Response:
        if not submission_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in submission_id):
            raise HTTPException(status_code=422, detail="invalid Hub submission ID")
        payload, content_type = hub_request(f"/api/submissions/{urllib.parse.quote(submission_id, safe='')}/audio")
        return Response(content=payload, media_type=content_type)

    @app.post("/api/hub/submissions/{submission_id}/annotations/{annotation_id}/reviews")
    def submit_hub_review(submission_id: str, annotation_id: str, data: dict[str, Any]) -> dict[str, Any]:
        for value in (submission_id, annotation_id):
            if not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in value):
                raise HTTPException(status_code=422, detail="invalid Hub review ID")
        path = f"/api/submissions/{urllib.parse.quote(submission_id, safe='')}/annotations/{urllib.parse.quote(annotation_id, safe='')}/reviews"
        payload, _ = hub_request(path, method="POST", payload=data)
        try:
            result = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=502, detail="Hub returned invalid review data") from exc
        if not isinstance(result, dict):
            raise HTTPException(status_code=502, detail="Hub returned an unexpected review response")
        return result

    @app.post("/api/live/review-queue/{capture_id}/mark-reviewed")
    def mark_live_reviewed(capture_id: str) -> dict[str, Any]:
        try:
            capture = store.set_live_reviewed(capture_id, True)
            return {"reviewed": True, "capture_id": capture["id"]}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/live/review-queue/{capture_id}/reopen")
    def reopen_live_review(capture_id: str) -> dict[str, Any]:
        try:
            capture = store.set_live_reviewed(capture_id, False)
            return {"reviewed": False, "capture_id": capture["id"]}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        except ValueError as exc:
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

    @app.get("/api/decoder-backends")
    def decoder_backends_status() -> dict[str, Any]:
        return {
            "backends": decoder_backends_registry.info(),
            "plugin_errors": decoder_backends_registry.load_errors,
            "extension_group": decoder_backends_registry.ENTRY_POINT_GROUP,
        }

    @app.get("/api/decoder-backends/{backend_id}/events")
    def decoder_backend_events(backend_id: str, limit: int = Query(default=100, ge=1, le=1000)) -> list[dict[str, Any]]:
        try:
            events = decoder_backends_registry.drain_events(backend_id, limit=limit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"decoder event source failed: {exc}") from exc
        result = []
        for event in events:
            products = []
            for product in event.products:
                if isinstance(product.content, bytes):
                    if len(product.content) > 8 * 1024 * 1024:
                        continue
                    content, encoding = base64.b64encode(product.content).decode("ascii"), "base64"
                else:
                    content, encoding = product.content, "utf-8"
                products.append({"kind": product.kind, "content": content, "encoding": encoding,
                                 "media_type": product.media_type, "metadata": product.metadata})
            result.append({"backend_id": event.backend_id, "mode_id": event.mode_id,
                           "native_mode": event.native_mode, "observed_utc": event.observed_utc,
                           "frequency_hz": event.frequency_hz, "snr_db": event.snr_db,
                           "metadata": event.metadata, "products": products})
        return result

    @app.post("/api/captures/{capture_id}/decoder-backends/{backend_id}/decode")
    def decode_with_native_backend(
        capture_id: str, backend_id: str, settings: NativeDecoderRequestBody,
    ) -> dict[str, Any]:
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        if settings.sample_start + settings.sample_count > int(capture["sample_count"]):
            raise HTTPException(status_code=422, detail="selected region exceeds the saved capture")
        if settings.region_id is not None:
            region = next((item for item in store.regions(capture_id) if item["id"] == settings.region_id), None)
            if region is None or region["lineage_state"] != "active":
                raise HTTPException(status_code=422, detail="decoder region must be active in this capture")
            if (settings.sample_start, settings.sample_count) != (region["sample_start"], region["sample_count"]):
                raise HTTPException(status_code=422, detail="decoder sample bounds must match the linked region")
        backend_status = next((item for item in decoder_backends_registry.info() if item["backend_id"] == backend_id.casefold()), None)
        if backend_status is None:
            raise HTTPException(status_code=404, detail="decoder backend is not registered")
        if not backend_status.get("connected"):
            raise HTTPException(status_code=503, detail=backend_status.get("note") or "decoder backend is unavailable")
        binding = decoder_backends_registry.mode_binding(backend_id, settings.mode_id)
        if binding is None or not binding.selectable:
            raise HTTPException(status_code=422, detail="selected label is not an available mode for this decoder backend")
        try:
            sample_rate, samples = read_audio_range(capture["audio_path"], settings.sample_start, settings.sample_count)
            with tempfile.TemporaryDirectory(prefix="sig-id-native-decoder-") as temporary:
                audio_path = Path(temporary) / "selected-window.wav"
                pcm = np.clip(np.rint(samples * 32767.0), -32768, 32767).astype("<i2")
                with wave.open(str(audio_path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(pcm.tobytes())
                result = decoder_backends_registry.decode(backend_id, DecoderRequest(
                    audio_path=audio_path,
                    sample_rate=sample_rate,
                    mode_id=settings.mode_id,
                    native_mode=binding.native_mode,
                    sample_start=settings.sample_start,
                    sample_count=settings.sample_count,
                    carrier_hz=settings.carrier_hz,
                    metadata={"capture_id": capture_id, "capture_sha256": capture["sha256"],
                              "region_id": settings.region_id},
                ))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logging.getLogger(__name__).exception("Native decoder backend %s failed", backend_id)
            raise HTTPException(status_code=503, detail=f"native decoder failed: {exc}") from exc
        products = []
        for product in result.products:
            if isinstance(product.content, bytes):
                if len(product.content) > 8 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="decoder output artifact exceeds the 8 MiB response limit")
                content = base64.b64encode(product.content).decode("ascii")
                encoding = "base64"
            else:
                content = product.content
                encoding = "utf-8"
            products.append({"kind": product.kind, "content": content, "encoding": encoding,
                             "media_type": product.media_type, "metadata": product.metadata})
        text_output = "\n".join(
            product.content for product in result.products
            if product.kind == "text" and isinstance(product.content, str)
        )
        product_kinds = sorted({product.kind for product in result.products})
        run_id = store.create_decoder_run(
            session_id=f"saved-capture:{capture_id}",
            capture_id=capture_id,
            region_id=settings.region_id,
            track_id=f"native:{backend_id}:{settings.sample_start}:{settings.sample_count}",
            sample_start=settings.sample_start,
            sample_count=settings.sample_count,
            sample_rate=sample_rate,
            carrier_hz=settings.carrier_hz,
            modulation_hypothesis="manual native-backend selection",
            backend=result.backend_id,
            modes=[result.native_mode],
            attempted_modes=[result.native_mode],
            current_mode=result.native_mode,
            output_offset=0,
            status=result.status,
            note=result.note or f"Native decoder returned {', '.join(product_kinds) or 'no'} products; review before treating as a label.",
            output_scope="isolated_native_window",
            association="exact_selected_capture_window",
        )
        store.update_decoder_run(run_id, {
            "attempted_modes": [result.native_mode], "current_mode": result.native_mode,
            "output_offset": 0, "output_end": len(text_output), "text": text_output,
            "status": result.status,
            "note": result.note or f"Native decoder returned {', '.join(product_kinds) or 'no'} products; review before treating as a label.",
        })
        return {
            "run_id": run_id,
            "backend_id": result.backend_id, "application": result.application,
            "mode_id": result.mode_id, "native_mode": result.native_mode,
            "status": result.status, "version": result.version, "note": result.note,
            "capture_id": capture_id, "sample_start": settings.sample_start,
            "sample_count": settings.sample_count, "sample_rate": sample_rate,
            "products": products,
        }

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

    decode_collection_lock = Lock()

    @contextmanager
    def decode_collection_guard():
        # Serialize the durable lookup/create pair across workers and app instances.
        # Keep this inode: unlinking a flock file would let new callers bypass it.
        with decode_collection_lock, (capture_dir / ".decode-collection.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def validate_collectable_run(run: dict[str, Any], *, live: bool, sample_rate: int) -> None:
        if run["status"] not in {"completed", "completed_no_text", "completed_empty", "tentative_output", "failed", "cancelled", "exhausted"}:
            raise HTTPException(status_code=409, detail="decoder attempt is not finished")
        expected = ({("isolated_live_window_replay", "exact_live_track_window"),
                     ("isolated_live_mode_sweep", "exact_live_track_window")} if live
                    else {("isolated_region_replay", "exact_capture_window"),
                          ("isolated_native_window", "exact_selected_capture_window")})
        if (run["output_scope"], run["association"]) not in expected:
            raise HTTPException(status_code=422, detail="decoder attempt does not have an exact source association")
        if (type(run["sample_start"]) is not int or run["sample_start"] < 0
                or type(run["sample_count"]) is not int or run["sample_count"] <= 0):
            raise HTTPException(status_code=422, detail="decoder attempt requires exact positive sample bounds")
        if run["sample_rate"] != sample_rate or sample_rate <= 0:
            raise HTTPException(status_code=422, detail="decoder attempt sample rate differs from its source")
        band = run["detector_evidence"].get("selection_bounds", run["detector_evidence"])
        low = band.get("frequency_low_hz")
        high = band.get("frequency_high_hz")
        if low is not None or high is not None:
            if (not isinstance(low, (int, float)) or not isinstance(high, (int, float))
                    or not math.isfinite(low) or not math.isfinite(high)
                    or not 0 <= low < high <= sample_rate / 2):
                raise HTTPException(status_code=422, detail="decoder attempt frequency bounds must fit the source Nyquist range")

    def existing_decode_collection(run_id: str) -> dict[str, Any] | None:
        for capture in store.list_captures():
            if capture["source_metadata"].get("collected_decoder_run_id") != run_id:
                continue
            if not Path(capture["audio_path"]).is_file():
                raise HTTPException(status_code=409, detail="collected audio is missing; restore or explicitly delete the existing review item")
            regions = store.regions(capture["id"])
            region = next((item for item in regions if item["evidence"].get("decoder_run", {}).get("id") == run_id), None)
            return {"capture_id": capture["id"], "region_id": region["id"] if region else None,
                    "run_id": run_id, "sample_rate": capture["sample_rate"],
                    "sample_count": capture["sample_count"], "deduplicated": True,
                    "message": "This attempt has already been collected."}
        return None

    def save_decode_collection(
        run: dict[str, Any], data: DecodeRunCollection,
        audio_path: Path, *, source_capture_id: str | None, sample_offset: int = 0,
    ) -> dict[str, Any]:
        capture_id = None
        try:
            metadata = WavSource(audio_path).metadata()
            validate_collectable_run(run, live=source_capture_id is None, sample_rate=metadata["sample_rate"])
            review_count = int(metadata["sample_rate"] * WINDOW_SECONDS)
            review_start = ((run["sample_start"] - sample_offset) // review_count) * review_count
            capture_id = store.add_wav_capture(
                audio_path, source_kind="reference" if source_capture_id else "live",
                source_metadata={
                    "auto_review_queue": True,
                    "training_candidate_requires_coverage_review": True,
                    "collected_decoder_run_id": run["id"],
                    "decoder_run": run,
                    "source_capture_id": source_capture_id,
                    "auto_capture_session_id": run["session_id"],
                    "window_start_sample": sample_offset,
                    "review_window_start_sample": review_start,
                    "review_window_sample_count": min(review_count, metadata["sample_count"] - review_start),
                },
            )
            band = run["detector_evidence"].get("selection_bounds", run["detector_evidence"])
            region_id = store.add_region(
                capture_id, sample_start=run["sample_start"] - sample_offset,
                sample_count=run["sample_count"],
                frequency_low_hz=band.get("frequency_low_hz"),
                frequency_high_hz=band.get("frequency_high_hz"),
                label=SignalLabel(source="decoder", review_state="pending", notes=data.notes,
                                  evidence={"source": "decoder_run_collection", "decoder_run": run,
                                            "source_capture_id": source_capture_id,
                                            "window_start_sample": sample_offset}),
            )
            capture = store.capture(capture_id)
            return {"capture_id": capture_id, "region_id": region_id, "run_id": run["id"],
                    "sample_rate": capture["sample_rate"], "sample_count": capture["sample_count"], "deduplicated": False,
                    "message": "Collected a pending candidate. Review labels and coverage before export."}
        except Exception:
            if capture_id is not None:
                store.delete_capture(capture_id)
            audio_path.unlink(missing_ok=True)
            raise

    @app.post("/api/captures/{capture_id}/decode-runs/{run_id}/collect", status_code=201)
    def collect_capture_decode_run(capture_id: str, run_id: str, data: DecodeRunCollection) -> dict[str, Any]:
        try:
            capture = store.capture(capture_id)
            run = next((item for item in store.decoder_runs(capture_id=capture_id) if item["id"] == run_id), None)
            if run is None:
                raise HTTPException(status_code=404, detail="decoder run not found for this capture")
            with decode_collection_guard():
                existing = existing_decode_collection(run_id)
                if existing is not None:
                    return existing
                validate_collectable_run(run, live=False, sample_rate=capture["sample_rate"])
                audio_path = capture_dir / f"{uuid4()}.wav"
                try:
                    try:
                        os.link(capture["audio_path"], audio_path)
                    except OSError:
                        shutil.copyfile(capture["audio_path"], audio_path)
                    return save_decode_collection(run, data, audio_path, source_capture_id=capture_id)
                except Exception:
                    audio_path.unlink(missing_ok=True)
                    raise
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="source audio not found") from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/live/sessions/{session_id}/decode-runs/{run_id}/collect", status_code=201)
    def collect_live_decode_run(session_id: str, run_id: str, data: DecodeRunCollection) -> dict[str, Any]:
        run = next((item for item in store.decoder_runs(session_id=session_id) if item["id"] == run_id), None)
        if run is None:
            raise HTTPException(status_code=404, detail="decoder run not found for this session")
        with decode_collection_guard():
            existing = existing_decode_collection(run_id)
            if existing is not None:
                return existing
            snapshot = live_capture.analysis_audio()
            if snapshot is None or not snapshot["active"] or snapshot["session_id"] != session_id:
                raise HTTPException(status_code=409, detail="that live capture is no longer active")
            validate_collectable_run(run, live=True, sample_rate=int(snapshot["sample_rate"]))
            window_start = int(snapshot["sample_offset"])
            window_count = int(snapshot["window_sample_count"])
            channels = int(snapshot["channels"])
            if (window_start < 0 or window_count <= 0 or channels <= 0
                    or len(snapshot["pcm"]) != window_count * channels * 2):
                raise HTTPException(status_code=422, detail="live audio snapshot has inconsistent sample bounds")
            if (run["sample_start"] < window_start
                    or run["sample_start"] + run["sample_count"] > window_start + window_count):
                raise HTTPException(status_code=422, detail="decoder attempt samples are no longer in the live buffer")
            audio_path = capture_dir / f"{uuid4()}.wav"
            try:
                with wave.open(str(audio_path), "wb") as wav:
                    wav.setnchannels(int(snapshot["channels"]))
                    wav.setsampwidth(2)
                    wav.setframerate(int(snapshot["sample_rate"]))
                    wav.writeframes(snapshot["pcm"])
                return save_decode_collection(run, data, audio_path, source_capture_id=None,
                                              sample_offset=int(snapshot["sample_offset"]))
            except (ValueError, OSError) as exc:
                audio_path.unlink(missing_ok=True)
                raise HTTPException(status_code=422, detail=str(exc)) from exc

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
        selection_evidence = settings.selection_evidence(int(snapshot["sample_rate"]))
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
        try:
            candidates = _checked_replay_modes(settings, modes)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
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
                backend="fldigi-isolated-replay", modes=candidates, attempted_modes=[],
                current_mode=candidates[0], output_offset=0, status="queued",
                note=("Waiting for isolated RSID-assisted mode identification." if settings.auto_identify
                      else "Waiting to replay the selected live track window in an isolated Fldigi instance."),
                carrier_hz=settings.carrier_hz,
                auto_identify=settings.auto_identify,
                detector_evidence=selection_evidence,
                output_scope="isolated_live_window_replay", association="exact_live_track_window",
            )
            decode_executor.submit(
                run_region_replay, run_id, settings.model_copy(update={"candidate_modes": candidates}), int(snapshot["sample_rate"]),
                lambda: samples.tolist(),
            )
            return {"run_id": run_id, "status": "queued", "mode": candidates[0], "modes": candidates, "auto_identify": settings.auto_identify, "carrier_hz": settings.carrier_hz}
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
        selection_evidence = settings.selection_evidence(int(capture["sample_rate"]))
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
        try:
            candidates = _checked_replay_modes(settings, modes)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not decode_job_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="an isolated Fldigi replay is already running")
        track_id = settings.track_id or f"capture:{capture_id}:samples:{settings.sample_start}:{settings.sample_count}"
        try:
            run_id = store.create_decoder_run(
                session_id=f"capture:{capture_id}", capture_id=capture_id,
                track_id=track_id, sample_start=settings.sample_start,
                sample_count=settings.sample_count, sample_rate=int(capture["sample_rate"]),
                modulation_hypothesis="unknown", backend="fldigi-isolated-replay",
                modes=candidates, attempted_modes=[], current_mode=candidates[0],
                output_offset=0, status="queued",
                note=("Waiting for isolated RSID-assisted mode identification." if settings.auto_identify
                      else "Waiting to replay the selected saved-capture region."),
                carrier_hz=settings.carrier_hz,
                auto_identify=settings.auto_identify,
                region_id=settings.region_id,
                classifier_evidence=(linked_region["evidence"].get("classifier_proposal") if linked_region else None),
                detector_evidence={**(saved_detector_evidence or {}), **selection_evidence},
                output_scope="isolated_region_replay", association="exact_capture_window",
            )
            decode_executor.submit(
                run_saved_replay, run_id, capture,
                settings.model_copy(update={"candidate_modes": candidates}),
            )
            return {"run_id": run_id, "status": "queued", "mode": candidates[0], "modes": candidates, "auto_identify": settings.auto_identify, "carrier_hz": settings.carrier_hz}
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

    @app.delete("/api/captures/{capture_id}")
    def delete_capture(capture_id: str) -> dict[str, bool]:
        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        if not capture["source_metadata"].get("auto_review_queue"):
            raise HTTPException(status_code=409, detail="only queued live review samples can be deleted here")
        audio_path = Path(capture["audio_path"]).resolve()
        try:
            audio_path.relative_to(capture_dir.resolve())
        except ValueError as exc:
            raise HTTPException(status_code=409, detail="review sample is outside the managed capture folder") from exc
        try:
            audio_path.unlink(missing_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"could not remove review sample audio: {exc}") from exc
        if not store.delete_capture(capture_id):
            raise HTTPException(status_code=404, detail="capture not found")
        return {"deleted": True}

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
        sample_start: int = 0,
        window_sample_count: int | None = None,
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
                sample_start=sample_start,
                window_sample_count=window_sample_count,
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
        classifier = app.state.classifier
        classifier_available = classifier is not None
        if classifier is not None:
            sample_rate = int(capture["sample_rate"])
            window_seconds = float(
                classifier["metadata"].get("classifier_window_seconds") or CLASSIFIER_WINDOW_SECONDS
            )
            target_count = max(1, round(window_seconds * sample_rate))
            max_scored = set(id(item) for item in sorted(
                detections, key=lambda item: float(item["confidence"]), reverse=True
            )[:32])
            for detection in detections:
                if id(detection) not in max_scored:
                    detection["classifier_proposal"] = {
                        "available": False, "reason": "32-box classification limit reached",
                    }
                    continue
                if int(capture["sample_count"]) < target_count:
                    detection["classifier_proposal"] = {
                        "available": False,
                        "reason": f"capture is shorter than a {window_seconds:g}-second classifier crop",
                    }
                    continue
                midpoint = int(detection["sample_start"]) + int(detection["sample_count"]) // 2
                crop_start = max(0, min(int(capture["sample_count"]) - target_count,
                                        midpoint - target_count // 2))
                try:
                    crop_rate, crop = read_audio_range(capture["audio_path"], crop_start, target_count)
                    detection["classifier_proposal"] = {
                        "sample_start": crop_start,
                        "sample_count": target_count,
                        **_classify_signal_crop(
                            classifier, crop, crop_rate,
                            float(detection["frequency_low_hz"]),
                            float(detection["frequency_high_hz"]),
                        ),
                    }
                except Exception as exc:
                    logging.getLogger(__name__).exception(
                        "Classifier proposal failed for CV detection in capture %s", capture_id,
                    )
                    detection["classifier_proposal"] = {"available": False, "reason": str(exc)}
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
            "classifier_available": classifier_available,
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
        crop_count: int = 1,
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
        model = app.state.classifier
        feature_version = model["metadata"]["feature_version"]
        if not 1 <= crop_count <= 20:
            raise ValueError("classifier crop count must be between 1 and 20")
        end = sample_start + classified_sample_count
        selected_values: list[float] = []
        for block in WavSource(capture["audio_path"]).blocks():
            block_end = block.first_sample + block.sample_count
            if block_end <= sample_start:
                continue
            if block.first_sample >= end:
                break
            left = max(0, sample_start - block.first_sample)
            right = min(block.sample_count, end - block.first_sample)
            selected_values.extend(block.samples[left:right])
        sample_rate = int(capture["sample_rate"])
        classifier_window_seconds = float(model["metadata"].get("classifier_window_seconds") or CLASSIFIER_WINDOW_SECONDS)
        target_count = max(1, round(sample_rate * classifier_window_seconds))
        crop_ranges: list[tuple[int, list[float]]] = []
        if model["metadata"].get("classifier_window_seconds") is not None:
            if len(selected_values) > target_count:
                last_start = len(selected_values) - target_count
                available_crops = max(1, len(selected_values) // target_count)
                offsets = np.asarray([last_start // 2]) if crop_count == 1 else (
                    np.linspace(0, available_crops - 1, min(crop_count, available_crops))
                    .round().astype(int) * target_count
                )
                unique_offsets = list(dict.fromkeys(int(offset) for offset in offsets))
                crop_ranges = [(sample_start + offset, selected_values[offset : offset + target_count]) for offset in unique_offsets]
            else:
                crop_ranges = [(sample_start, selected_values)]
        else:
            crop_ranges = [(sample_start, selected_values)]
        classifier_frequency_range = None
        if feature_version in ROI_FEATURE_VERSIONS:
            classifier_frequency_range = (
                (frequency_low_hz, frequency_high_hz)
                if frequency_low_hz is not None
                else (200.0, min(3800.0, int(capture["sample_rate"]) / 2 - 20.0))
            )
        crop_results = []
        for crop_start, crop_values in crop_ranges:
            features = extract_features(
                crop_values,
                sample_rate,
                feature_version=feature_version,
                frequency_range_hz=classifier_frequency_range,
                classifier_window_seconds=classifier_window_seconds,
            )
            crop_results.append({
                "sample_start": crop_start,
                "sample_count": target_count if feature_version in FIXED_WINDOW_FEATURE_VERSIONS else len(crop_values),
                **propose_loaded(features, model["metadata"], model["parameters"]),
            })
        result = max(crop_results, key=lambda item: item["confidence"]).copy()
        method_votes: dict[str, list[dict[str, Any]]] = {}
        family_votes: dict[str, list[dict[str, Any]]] = {}
        for crop_result in crop_results:
            if crop_result["method"]:
                method_votes.setdefault(crop_result["method"], []).append(crop_result)
            if crop_result["family"]:
                family_votes.setdefault(crop_result["family"], []).append(crop_result)
        majority = len(crop_results) // 2 + 1
        method_consensus = next(((name, votes) for name, votes in method_votes.items() if len(votes) >= majority), (None, []))
        family_consensus = next(((name, votes) for name, votes in family_votes.items() if len(votes) >= majority), (None, []))
        if method_consensus[0]:
            result.update(method_consensus[1][0])
            result["method"] = method_consensus[0]
            result["method_id"] = method_consensus[1][0]["method_id"]
            result["family"] = method_consensus[1][0]["family"]
            result["abstained"] = False
            result["consensus_label"] = method_consensus[0]
            result["consensus_count"] = len(method_consensus[1])
            result["confidence"] = float(np.mean([vote["confidence"] for vote in method_consensus[1]]))
        elif family_consensus[0]:
            result["method"] = None
            result["method_id"] = None
            result["family"] = family_consensus[0]
            result["family_confidence"] = float(np.mean([vote["family_confidence"] for vote in family_consensus[1]]))
            result["family_abstained"] = False
            result["abstained"] = True
            result["consensus_label"] = family_consensus[0]
            result["consensus_count"] = len(family_consensus[1])
            result["confidence"] = float(np.mean([vote["confidence"] for vote in family_consensus[1]]))
        else:
            result["method"] = None
            result["method_id"] = None
            result["family"] = None
            result["family_abstained"] = True
            result["abstained"] = True
            result["consensus_label"] = None
            result["consensus_count"] = 0
        result["crop_count"] = len(crop_results)
        result["requested_crop_count"] = crop_count
        result["crop_results"] = [
            {key: crop.get(key) for key in ("sample_start", "sample_count", "method", "method_id", "family", "confidence", "top_candidate", "novelty_rejected")}
            for crop in crop_results
        ]
        result["consensus"] = bool(result.get("method") or result.get("family"))
        return {
            **result,
            "model_sha256": model["metadata"]["model_sha256"],
            "feature_version": model["metadata"]["feature_version"],
            "capture_id": capture_id,
            "sample_start": sample_start,
            "sample_count": classified_sample_count,
            "selection_sample_count": selection_sample_count,
            "selection_frequency_range_hz": [frequency_low_hz, frequency_high_hz],
            "feature_frequency_scope": (
                "roi_bandpass_expanded" if feature_version in ROI_FEATURE_VERSIONS and frequency_low_hz is not None
                else "full_audio_passband" if feature_version not in ROI_FEATURE_VERSIONS
                else "full_range_roi_bandpass"
            ),
            "classifier_input_duration_seconds": (
                float(model["metadata"].get("classifier_window_seconds") or CLASSIFIER_WINDOW_SECONDS)
                if model["metadata"].get("classifier_window_seconds") is not None else len(selected_values) / sample_rate
            ),
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
                settings.crop_count,
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
            return _persist_raw_attempt(store, capture_id, "2fsk", settings, result)
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
            return _persist_raw_attempt(store, capture_id, "ook", settings, result)
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
            return _persist_raw_attempt(store, capture_id, "mfsk", settings, result)
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
            return _persist_raw_attempt(store, capture_id, "bpsk", settings, result)
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
            return _persist_raw_attempt(store, capture_id, "qpsk", settings, result)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/captures/{capture_id}/demod-attempts")
    def get_raw_demod_attempts(capture_id: str) -> list[dict[str, Any]]:
        try:
            return store.raw_demod_attempts(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc

    @app.get("/api/captures/{capture_id}/regions")
    def get_regions(capture_id: str) -> list[dict[str, Any]]:
        try:
            return store.regions(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc

    @app.get("/api/signal-identities")
    def get_signal_identities() -> list[dict[str, Any]]:
        return store.signal_identities()

    @app.delete("/api/captures/{capture_id}/regions/{region_id}")
    def delete_region(
        capture_id: str, region_id: str,
        expected_revision: int = Query(ge=1),
        expected_geometry_revision: int = Query(ge=1),
    ) -> dict[str, Any]:
        try:
            store.delete_region(
                capture_id, region_id,
                expected_revision=expected_revision,
                expected_geometry_revision=expected_geometry_revision,
            )
            return {"deleted": True, "region_id": region_id}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture or region not found") from exc
        except LabelRevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

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
            if not store.region_exists(capture_id, region_id):
                raise KeyError("region not in this capture")
            return store.label_history(region_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture or region not found") from exc

    @app.get("/api/modes")
    def search_modes(q: str = Query(min_length=1, max_length=80), limit: int = Query(default=20, ge=1, le=50)) -> dict[str, Any]:
        modes = find_modes(q, limit=limit, additional_modes=store.accepted_mode_definitions())
        backends = {item["backend_id"]: item for item in decoder_backends_registry.info()}
        for mode in modes:
            backend = backends.get(mode.get("native_backend_id"))
            if backend is None:
                continue
            binding = next((item for item in backend.get("modes", []) if item.get("mode_id") == mode["id"]), None)
            if binding and backend.get("connected"):
                mode["decoder_status"] = "native decoder ready" if binding.get("selectable") else "native live decoder linked"
                mode["backend_selectable"] = bool(binding.get("selectable"))
            else:
                mode["decoder_status"] = "native decoder registered; this mode is not selectable"
        return {"registry_version": MODE_REGISTRY_VERSION, "modes": modes}

    @app.get("/api/field-guide")
    def signal_field_guide() -> dict[str, Any]:
        """Return the curated field guide plus every locally catalogued Fldigi mode."""
        from .field_guide import build_field_guide
        from .sigidwiki_images import read_wiki_image_manifest

        guide = build_field_guide()
        available = read_wiki_image_manifest(root)
        for entry in guide["entries"]:
            if entry["id"] in available:
                entry["wiki_image"] = f"/api/field-guide/wiki-images/{entry['id']}"
                entry["wiki_image_source"] = available[entry["id"]].get("source_attribution_page")
        return guide

    @app.get("/api/field-guide/wiki-images/{entry_id}")
    def get_field_guide_wiki_image(entry_id: str) -> FileResponse:
        from .sigidwiki_images import read_wiki_image_manifest

        asset = read_wiki_image_manifest(root).get(entry_id)
        if not asset or not isinstance(asset.get("path"), str):
            raise HTTPException(status_code=404, detail="No cached SigIDWiki reference image for this entry")
        image_path = (root / "field-guide" / "wiki-images" / asset["path"]).resolve()
        asset_root = (root / "field-guide" / "wiki-images").resolve()
        if image_path.parent != asset_root or image_path.suffix.lower() != ".png" or not image_path.is_file():
            raise HTTPException(status_code=404, detail="Field-guide image is unavailable")
        return FileResponse(image_path, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/field-guide/examples")
    def list_field_guide_examples() -> list[dict[str, Any]]:
        """List isolated Fldigi-generated reference signals, separate from live captures."""
        return [
            item for item in field_guide_store.list_library_signals()
            if item["source_metadata"].get("generator") == "sig-id-fldigi"
        ]

    @app.get("/api/field-guide/examples/{capture_id}/waterfall")
    def get_field_guide_example_waterfall(
        capture_id: str,
        fft_size: int = 4096,
        hop_size: int = 512,
        max_rows: int = 512,
        sample_start: int = 0,
        window_sample_count: int | None = None,
    ) -> dict[str, Any]:
        try:
            capture = field_guide_store.capture(capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="field-guide example not found") from exc
        if capture["source_metadata"].get("generator") != "sig-id-fldigi":
            raise HTTPException(status_code=404, detail="capture is not a stored Fldigi reference example")
        try:
            return waterfall(
                capture["audio_path"], fft_size=fft_size, hop_size=hop_size,
                max_rows=max_rows, sample_start=sample_start,
                window_sample_count=window_sample_count,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

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
                    proposal.crop_count,
                )
            resolved = resolve_mode_label(data.mode_id, {
                "family": data.family, "method": data.method,
                "protocol": data.protocol, "variant": data.variant,
            }, additional_modes=store.accepted_mode_definitions())
            has_label = any((resolved["family"], resolved["method"], resolved["protocol"], resolved["variant"]))
            if data.identity_unknown_confirmed and has_label:
                raise ValueError("an explicitly unknown label cannot also contain a mode or taxonomy")
            label_evidence = {"reviewed_from_revision": data.expected_revision}
            if data.identity_unknown_confirmed:
                label_evidence["identity_unknown_confirmed"] = True
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
                source="human" if has_label or data.identity_unknown_confirmed else "unknown",
                review_state="accepted" if has_label or data.identity_unknown_confirmed else "pending",
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
            if not store.region_exists(capture_id, region_id):
                raise KeyError("region not in this capture")
            return store.feedback_history(region_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture or region not found") from exc

    @app.get("/api/captures/{capture_id}/regions/{region_id}/geometry")
    def region_geometry_history(capture_id: str, region_id: str) -> list[dict[str, Any]]:
        try:
            if not store.region_exists(capture_id, region_id):
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
            if not store.region_exists(capture_id, region_id):
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
            capture = store.capture(capture_id)
            cv_proposals = []
            for proposal in data.cv_proposals:
                evidence = proposal.detector_evidence.model_dump()
                if proposal.decision != "dismissed":
                    raise HTTPException(status_code=422, detail="merged CV proposals must be dismissed from the current scan")
                if proposal.proposal_id != cv_proposal_id({
                    "model_sha256": evidence["model_sha256"],
                    "sample_start": proposal.sample_start,
                    "sample_count": proposal.sample_count,
                    "frequency_low_hz": proposal.frequency_low_hz,
                    "frequency_high_hz": proposal.frequency_high_hz,
                }):
                    raise HTTPException(status_code=422, detail="proposal id does not match its model and geometry")
                if evidence["sample_rate"] != capture["sample_rate"]:
                    raise HTTPException(status_code=422, detail="proposal sample rate does not match the capture")
                cv_proposals.append({**proposal.model_dump(), "detector_evidence": evidence})
            return store.merge_regions(
                capture_id, source_region_ids=data.source_region_ids,
                expected_geometry_revisions=data.expected_geometry_revisions,
                cv_proposals=cv_proposals,
                result_bounds=data.result_bounds.model_dump() if data.result_bounds else None,
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
                if data.sample_start + data.sample_count > int(capture["sample_count"]):
                    raise ValueError("proposal region exceeds the capture sample range")
                for source_window in detector_evidence.supporting_windows:
                    source_start = source_window.sample_start
                    source_end = source_start + source_window.sample_count
                    if source_end > int(capture["sample_count"]):
                        raise ValueError("supporting detector window exceeds the capture sample range")
            elif data.signal_presence_confirmed:
                raise ValueError("signal presence confirmation requires detector evidence")
            resolved = resolve_mode_label(data.mode_id, {
                "family": data.family, "method": data.method,
                "protocol": data.protocol, "variant": data.variant,
            }, additional_modes=store.accepted_mode_definitions())
            has_label = any((resolved["family"], resolved["method"], resolved["protocol"], resolved["variant"]))
            if data.identity_unknown_confirmed and has_label:
                raise ValueError("an explicitly unknown label cannot also contain a mode or taxonomy")
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
                    proposal.crop_count,
                )
            if data.prediction_only:
                if classifier_evidence is None:
                    raise ValueError("a classifier proposal is required for a pending annotation")
                if data.mode_id or data.signal_presence_confirmed or data.identity_unknown_confirmed:
                    raise ValueError("pending predictions cannot claim a reviewed mode or signal presence")
                resolved = {
                    "family": classifier_evidence.get("family"),
                    "method": classifier_evidence.get("method"),
                    "protocol": None, "variant": None, "mode_id": None,
                    "registry_version": None,
                }
                has_label = bool(resolved["family"] or resolved["method"])
                for existing in store.regions(capture_id):
                    same_geometry = (existing["sample_start"] == data.sample_start
                                     and existing["sample_count"] == data.sample_count
                                     and existing["frequency_low_hz"] == data.frequency_low_hz
                                     and existing["frequency_high_hz"] == data.frequency_high_hz)
                    same_proposal = (detector_evidence is not None and detector_evidence.proposal_id is not None
                                     and existing["evidence"].get("detector_proposal", {}).get("proposal_id")
                                     == detector_evidence.proposal_id)
                    if not same_geometry and not same_proposal:
                        continue
                    if not same_geometry:
                        return existing
                    if existing["review_state"] == "pending" and existing["lineage_state"] == "active":
                        evidence = {**existing["evidence"], "classifier_proposal": classifier_evidence}
                        if detector_evidence is not None:
                            evidence["detector_proposal"] = detector_evidence.model_dump()
                        store.revise_label(existing["id"], SignalLabel(
                            family=resolved["family"], method=resolved["method"],
                            source="prediction" if has_label else "unknown", review_state="pending", evidence=evidence,
                            notes=existing["notes"] or data.notes,
                        ), expected_revision=existing["revision"])
                        return next(region for region in store.regions(capture_id) if region["id"] == existing["id"])
                    return existing
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
                    source=("prediction" if has_label else "unknown") if data.prediction_only else "human" if has_label or accepted_presence or data.identity_unknown_confirmed else "unknown",
                    review_state="pending" if data.prediction_only else "accepted" if has_label or accepted_presence or data.identity_unknown_confirmed else "pending",
                    evidence={
                        **({"mode_registry_version": resolved["registry_version"]} if resolved["mode_id"] else {}),
                        **({"detector_proposal": detector_evidence.model_dump()} if detector_evidence else {}),
                        **({"classifier_proposal": classifier_evidence} if classifier_evidence is not None else {}),
                        **({"signal_presence_review": "confirmed_unknown_identity"} if accepted_presence else {}),
                        **({"identity_unknown_confirmed": True} if data.identity_unknown_confirmed else {}),
                    } or None,
                    notes=data.notes,
                ),
                origin=data.origin,
            )
            return next(region for region in store.regions(capture_id) if region["id"] == region_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        except LabelRevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/captures/{capture_id}/regions/batch", status_code=201)
    def create_annotation_batch(capture_id: str, data: WindowAnnotationBatchCreate) -> dict[str, Any]:
        try:
            capture = store.capture(capture_id)
            if data.window_start_sample + data.window_sample_count > int(capture["sample_count"]):
                raise ValueError("annotation window exceeds the capture sample range")
            if data.window_sample_count > int(capture["sample_rate"]) * 10 + 1:
                raise ValueError("annotation batches must cover one 10-second window")
            additional_modes = store.accepted_mode_definitions()
            batch_regions = []
            for item in data.regions:
                detector_evidence = item.detector_evidence
                if item.signal_presence_confirmed and item.origin != "detector_proposal":
                    raise ValueError("signal presence confirmation is only valid for a detector proposal")
                if detector_evidence is not None:
                    if item.origin != "detector_proposal":
                        raise ValueError("detector evidence is only valid for a detector proposal")
                    if detector_evidence.sample_rate != int(capture["sample_rate"]):
                        raise ValueError("detector sample rate does not match the capture")
                    if detector_evidence.window_sample_start + detector_evidence.window_sample_count > int(capture["sample_count"]):
                        raise ValueError("detector window exceeds the capture sample range")
                    if item.sample_start + item.sample_count > int(capture["sample_count"]):
                        raise ValueError("proposal region exceeds the capture sample range")
                    if any(
                        window.sample_start + window.sample_count > int(capture["sample_count"])
                        for window in detector_evidence.supporting_windows
                    ):
                        raise ValueError("supporting detector window exceeds the capture sample range")
                resolved = resolve_mode_label(item.mode_id, {
                    "family": item.family, "method": item.method,
                    "protocol": item.protocol, "variant": item.variant,
                }, additional_modes=additional_modes)
                has_identity = any(resolved[key] for key in ("family", "method", "protocol", "variant"))
                accepted_presence = item.signal_presence_confirmed
                human_reviewed = has_identity or accepted_presence or item.identity_unknown_confirmed
                classifier_evidence = None
                if item.classifier_proposal is not None:
                    proposal = item.classifier_proposal
                    if proposal.sample_start != item.sample_start or proposal.selection_sample_count != item.sample_count:
                        raise ValueError("classifier proposal selection must match the saved region geometry")
                    if proposal.frequency_low_hz is not None and (
                        proposal.frequency_low_hz != item.frequency_low_hz
                        or proposal.frequency_high_hz != item.frequency_high_hz
                    ):
                        raise ValueError("classifier proposal frequency bounds must match the saved region geometry")
                    classifier_evidence = classifier_proposal_for_range(
                        capture_id, capture, proposal.sample_start,
                        proposal.sample_count, proposal.selection_sample_count,
                        proposal.frequency_low_hz, proposal.frequency_high_hz,
                        proposal.crop_count,
                    )
                batch_regions.append({
                    "client_id": item.client_id,
                    "sample_start": item.sample_start,
                    "sample_count": item.sample_count,
                    "frequency_low_hz": item.frequency_low_hz,
                    "frequency_high_hz": item.frequency_high_hz,
                    "origin": item.origin,
                    "label": SignalLabel(
                        family=resolved["family"], method=resolved["method"],
                        protocol=resolved["protocol"], variant=resolved["variant"],
                        mode_id=resolved["mode_id"],
                        source="human" if human_reviewed else "unknown",
                        review_state="accepted" if human_reviewed else "pending",
                        evidence={
                        "annotation_window": {
                            "sample_start": data.window_start_sample,
                            "sample_count": data.window_sample_count,
                            "duration_seconds": data.window_sample_count / int(capture["sample_rate"]),
                        },
                        **({"detector_proposal": detector_evidence.model_dump()} if detector_evidence else {}),
                        **({"classifier_proposal": classifier_evidence} if classifier_evidence is not None else {}),
                        **({"signal_presence_review": "confirmed_unknown_identity"} if accepted_presence else {}),
                        **({"identity_unknown_confirmed": True} if item.identity_unknown_confirmed else {}),
                        **({"mode_registry_version": resolved["registry_version"]} if resolved["mode_id"] else {}),
                    },
                        notes=item.notes,
                    ),
                })
            current_regions = {region["id"]: region for region in store.regions(capture_id)}
            batch_updates = []
            for item in data.updates:
                current_region = current_regions.get(item.region_id)
                if current_region is None:
                    raise KeyError(f"unknown region in capture: {item.region_id}")
                resolved = resolve_mode_label(item.mode_id, {
                    "family": item.family, "method": item.method,
                    "protocol": item.protocol, "variant": item.variant,
                }, additional_modes=additional_modes)
                has_identity = any(resolved[key] for key in ("family", "method", "protocol", "variant"))
                human_reviewed = has_identity or item.identity_unknown_confirmed
                evidence = {
                    **current_region.get("evidence", {}),
                    "reviewed_from_revision": item.expected_revision,
                    "annotation_window": {
                        "sample_start": data.window_start_sample,
                        "sample_count": data.window_sample_count,
                        "duration_seconds": data.window_sample_count / int(capture["sample_rate"]),
                    },
                }
                if item.identity_unknown_confirmed:
                    evidence["identity_unknown_confirmed"] = True
                else:
                    evidence.pop("identity_unknown_confirmed", None)
                if resolved["mode_id"]:
                    evidence["mode_registry_version"] = resolved["registry_version"]
                else:
                    evidence.pop("mode_registry_version", None)
                batch_updates.append({
                    "region_id": item.region_id,
                    "expected_revision": item.expected_revision,
                    "expected_geometry_revision": item.expected_geometry_revision,
                    "sample_start": item.sample_start,
                    "sample_count": item.sample_count,
                    "frequency_low_hz": item.frequency_low_hz,
                    "frequency_high_hz": item.frequency_high_hz,
                    "label": SignalLabel(
                        family=resolved["family"], method=resolved["method"],
                        protocol=resolved["protocol"], variant=resolved["variant"],
                        mode_id=resolved["mode_id"],
                        source="human" if human_reviewed else "unknown",
                        review_state="accepted" if human_reviewed else "pending",
                        evidence=evidence,
                        notes=item.notes,
                    ),
                })
            batch_deletions = [item.model_dump() for item in data.deletions]
            batch_merges = []
            for item in data.merges:
                resolved = resolve_mode_label(item.mode_id, {
                    "family": item.family, "method": item.method,
                    "protocol": item.protocol, "variant": item.variant,
                }, additional_modes=additional_modes)
                has_identity = any(resolved[key] for key in ("family", "method", "protocol", "variant"))
                human_reviewed = has_identity or item.identity_unknown_confirmed
                evidence = {
                    "annotation_window": {
                        "sample_start": data.window_start_sample,
                        "sample_count": data.window_sample_count,
                        "duration_seconds": data.window_sample_count / int(capture["sample_rate"]),
                    },
                    "merge_target_region_id": item.target_region_id,
                    **({"identity_unknown_confirmed": True} if item.identity_unknown_confirmed else {}),
                    **({"mode_registry_version": resolved["registry_version"]} if resolved["mode_id"] else {}),
                }
                if item.target_region_id and item.target_region_id in current_regions:
                    target = current_regions[item.target_region_id]
                    evidence["merged_target_label_evidence"] = target.get("evidence", {})
                cv_proposals = []
                for proposal in item.cv_proposals:
                    proposal_data = proposal.model_dump()
                    detector = proposal_data["detector_evidence"]
                    if proposal_data["decision"] != "dismissed":
                        raise HTTPException(status_code=422, detail="merged CV proposals must be dismissed from the current scan")
                    if proposal_data["proposal_id"] != cv_proposal_id({
                        "model_sha256": detector["model_sha256"],
                        "sample_start": proposal_data["sample_start"],
                        "sample_count": proposal_data["sample_count"],
                        "frequency_low_hz": proposal_data["frequency_low_hz"],
                        "frequency_high_hz": proposal_data["frequency_high_hz"],
                    }):
                        raise HTTPException(status_code=422, detail="proposal id does not match its model and geometry")
                    if detector["sample_rate"] != capture["sample_rate"]:
                        raise HTTPException(status_code=422, detail="proposal sample rate does not match the capture")
                    cv_proposals.append(proposal_data)
                heuristic_proposals = [proposal.model_dump() for proposal in item.heuristic_proposals]
                batch_merges.append({
                    "client_id": item.client_id,
                    "source_region_ids": item.source_region_ids,
                    "expected_revisions": item.expected_revisions,
                    "expected_geometry_revisions": item.expected_geometry_revisions,
                    "cv_proposals": cv_proposals,
                    "heuristic_proposals": heuristic_proposals,
                    "target_region_id": item.target_region_id,
                    "sample_start": item.sample_start,
                    "sample_count": item.sample_count,
                    "frequency_low_hz": item.frequency_low_hz,
                    "frequency_high_hz": item.frequency_high_hz,
                    "label": SignalLabel(
                        family=resolved["family"], method=resolved["method"],
                        protocol=resolved["protocol"], variant=resolved["variant"],
                        mode_id=resolved["mode_id"],
                        source="human" if human_reviewed else "unknown",
                        review_state="accepted" if human_reviewed else "pending",
                        evidence=evidence,
                        notes=item.notes,
                    ),
                })
            batch_identity_links = [item.model_dump() for item in data.identity_links]
            region_ids, replayed = store.add_regions_batch(
                capture_id, data.idempotency_key, batch_regions,
                updates=batch_updates,
                deletions=batch_deletions,
                merges=batch_merges,
                identity_links=batch_identity_links,
                window_start_sample=data.window_start_sample,
                window_sample_count=data.window_sample_count,
            )
            current = {region["id"]: region for region in store.regions(capture_id)}
            return {
                "capture_id": capture_id,
                "idempotency_key": data.idempotency_key,
                "replayed": replayed,
                "regions": [current[region_id] for region_id in region_ids if region_id in current],
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        except LabelRevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
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

    def web_asset(name: str, media_type: str | None = None) -> FileResponse:
        return FileResponse(WEB_DIR / name, media_type=media_type, headers={"Cache-Control": "no-store"})

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return web_asset("index.html")

    @app.get("/app.js", include_in_schema=False)
    def javascript() -> FileResponse:
        return web_asset("app.js", "text/javascript")

    @app.get("/waterfall-window.mjs", include_in_schema=False)
    def waterfall_window_module() -> FileResponse:
        return web_asset("waterfall-window.mjs", "text/javascript")

    @app.get("/demod-export.mjs", include_in_schema=False)
    def demod_export_module() -> FileResponse:
        return web_asset("demod-export.mjs", "text/javascript")

    @app.get("/bit-inspector.mjs", include_in_schema=False)
    def bit_inspector_module() -> FileResponse:
        return web_asset("bit-inspector.mjs", "text/javascript")

    @app.get("/protocol-stencils.mjs", include_in_schema=False)
    def protocol_stencils_module() -> FileResponse:
        return web_asset("protocol-stencils.mjs", "text/javascript")

    @app.get("/mode-candidates.mjs", include_in_schema=False)
    def mode_candidates_module() -> FileResponse:
        return web_asset("mode-candidates.mjs", "text/javascript")

    @app.get("/signal-overlays.mjs", include_in_schema=False)
    def signal_overlays_module() -> FileResponse:
        return web_asset("signal-overlays.mjs", "text/javascript")

    @app.get("/decode-context.mjs", include_in_schema=False)
    def decode_context_module() -> FileResponse:
        return web_asset("decode-context.mjs", "text/javascript")

    @app.get("/live-waterfall.mjs", include_in_schema=False)
    def live_waterfall_module() -> FileResponse:
        return web_asset("live-waterfall.mjs", "text/javascript")

    @app.get("/style.css", include_in_schema=False)
    def stylesheet() -> FileResponse:
        return web_asset("style.css", "text/css")

    return app
