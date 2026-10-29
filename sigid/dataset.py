"""Versioned SQLite metadata for recordings, signal regions and label history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any
from uuid import uuid4

from .wav import WavSource
from .modes import mode_name_conflicts, resolve_mode_label


SOURCE_KINDS = {"live", "wav", "synthetic", "reference"}
LABEL_SOURCES = {"unknown", "prediction", "rsid", "decoder", "human", "synthetic", "external"}
REVIEW_STATES = {"pending", "accepted", "rejected"}


class LabelRevisionConflict(ValueError):
    """The operator reviewed an older revision than the current saved label."""


@dataclass(frozen=True, slots=True)
class SignalLabel:
    family: str | None = None
    method: str | None = None
    protocol: str | None = None
    variant: str | None = None
    source: str = "unknown"
    review_state: str = "pending"
    evidence: dict[str, Any] | None = None
    notes: str = ""
    mode_id: str | None = None


class CaptureStore:
    """Local dataset index. Audio stays as lossless files beside the database."""

    def __init__(self, database: str | Path):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22):
                raise RuntimeError(f"unsupported dataset schema version: {version}")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS captures (
                    id TEXT PRIMARY KEY,
                    created_utc TEXT NOT NULL,
                    source_kind TEXT NOT NULL CHECK(source_kind IN ('live','wav','synthetic','reference')),
                    audio_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    sample_rate INTEGER NOT NULL CHECK(sample_rate > 0),
                    channels INTEGER NOT NULL CHECK(channels > 0),
                    sample_count INTEGER NOT NULL CHECK(sample_count >= 0),
                    source_metadata_json TEXT NOT NULL,
                    parent_capture_id TEXT REFERENCES captures(id)
                );
                CREATE TABLE IF NOT EXISTS signal_identities (
                    id TEXT PRIMARY KEY,
                    created_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS regions (
                    id TEXT PRIMARY KEY,
                    capture_id TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
                    sample_start INTEGER NOT NULL CHECK(sample_start >= 0),
                    sample_count INTEGER NOT NULL CHECK(sample_count > 0),
                    frequency_low_hz REAL,
                    frequency_high_hz REAL,
                    origin TEXT NOT NULL DEFAULT 'manual',
                    signal_id TEXT REFERENCES signal_identities(id),
                    created_utc TEXT NOT NULL,
                    CHECK((frequency_low_hz IS NULL AND frequency_high_hz IS NULL) OR
                          (frequency_low_hz IS NOT NULL AND frequency_high_hz IS NOT NULL AND
                           frequency_low_hz < frequency_high_hz))
                );
                CREATE INDEX IF NOT EXISTS regions_by_capture ON regions(capture_id, sample_start);
                CREATE TABLE IF NOT EXISTS region_signal_identity_revisions (
                    region_id TEXT NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
                    revision INTEGER NOT NULL CHECK(revision > 0),
                    signal_id TEXT NOT NULL REFERENCES signal_identities(id),
                    created_utc TEXT NOT NULL,
                    PRIMARY KEY(region_id, revision)
                );
                CREATE INDEX IF NOT EXISTS signal_identity_history_by_signal
                    ON region_signal_identity_revisions(signal_id, created_utc);
                CREATE INDEX IF NOT EXISTS captures_by_sha256 ON captures(sha256);
                CREATE TABLE IF NOT EXISTS region_geometry_revisions (
                    region_id TEXT NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
                    revision INTEGER NOT NULL CHECK(revision > 0),
                    sample_start INTEGER NOT NULL CHECK(sample_start >= 0),
                    sample_count INTEGER NOT NULL CHECK(sample_count > 0),
                    frequency_low_hz REAL,
                    frequency_high_hz REAL,
                    created_utc TEXT NOT NULL,
                    UNIQUE(region_id, revision),
                    CHECK((frequency_low_hz IS NULL AND frequency_high_hz IS NULL) OR
                          (frequency_low_hz IS NOT NULL AND frequency_high_hz IS NOT NULL AND frequency_low_hz < frequency_high_hz))
                );
                CREATE INDEX IF NOT EXISTS geometry_by_region ON region_geometry_revisions(region_id, revision DESC);
                CREATE TABLE IF NOT EXISTS label_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    region_id TEXT NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
                    revision INTEGER NOT NULL CHECK(revision > 0),
                    created_utc TEXT NOT NULL,
                    family TEXT,
                    method TEXT,
                    protocol TEXT,
                    variant TEXT,
                    mode_id TEXT,
                    source TEXT NOT NULL CHECK(source IN ('unknown','prediction','rsid','decoder','human','synthetic','external')),
                    review_state TEXT NOT NULL CHECK(review_state IN ('pending','accepted','rejected')),
                    evidence_json TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    UNIQUE(region_id, revision)
                );
                CREATE INDEX IF NOT EXISTS labels_by_region ON label_revisions(region_id, revision DESC);
                CREATE TABLE IF NOT EXISTS annotation_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    region_id TEXT NOT NULL,
                    label_revision INTEGER NOT NULL,
                    geometry_revision INTEGER NOT NULL,
                    resulting_revision INTEGER,
                    action TEXT NOT NULL CHECK(action IN ('correct','incorrect_identity','not_signal')),
                    correction_json TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_utc TEXT NOT NULL,
                    FOREIGN KEY(region_id, label_revision) REFERENCES label_revisions(region_id, revision) ON DELETE CASCADE,
                    FOREIGN KEY(region_id, geometry_revision) REFERENCES region_geometry_revisions(region_id, revision) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS feedback_by_region ON annotation_feedback(region_id, id);
                CREATE TABLE IF NOT EXISTS region_lineage (
                    relation_id TEXT NOT NULL,
                    relation TEXT NOT NULL CHECK(relation IN ('split','merge')),
                    source_region_id TEXT NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
                    result_region_id TEXT NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
                    source_index INTEGER NOT NULL CHECK(source_index >= 0),
                    result_index INTEGER NOT NULL CHECK(result_index >= 0),
                    created_utc TEXT NOT NULL,
                    PRIMARY KEY(relation_id, source_region_id, result_region_id)
                );
                CREATE INDEX IF NOT EXISTS lineage_by_source ON region_lineage(source_region_id, created_utc);
                CREATE INDEX IF NOT EXISTS lineage_by_result ON region_lineage(result_region_id, created_utc);
                CREATE TABLE IF NOT EXISTS region_removals (
                    region_id TEXT PRIMARY KEY REFERENCES regions(id) ON DELETE CASCADE,
                    capture_id TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
                    label_revision INTEGER NOT NULL CHECK(label_revision > 0),
                    geometry_revision INTEGER NOT NULL CHECK(geometry_revision > 0),
                    reason TEXT NOT NULL,
                    created_utc TEXT NOT NULL,
                    UNIQUE(region_id,capture_id)
                );
                CREATE INDEX IF NOT EXISTS removals_by_capture ON region_removals(capture_id, created_utc);
                CREATE TABLE IF NOT EXISTS scene_coverage_revisions (
                    coverage_id TEXT NOT NULL,
                    capture_id TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
                    revision INTEGER NOT NULL CHECK(revision > 0),
                    sample_start INTEGER NOT NULL CHECK(sample_start >= 0),
                    sample_count INTEGER NOT NULL CHECK(sample_count > 0),
                    frequency_low_hz REAL,
                    frequency_high_hz REAL,
                    task TEXT NOT NULL CHECK(task IN ('detection','modulation','protocol','decoder')),
                    taxonomy_scope_json TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('complete','partial')),
                    annotation_set_digest TEXT NOT NULL CHECK(length(annotation_set_digest)=64),
                    ignore_regions_json TEXT NOT NULL,
                    uncertain_regions_json TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    created_utc TEXT NOT NULL,
                    PRIMARY KEY(coverage_id,revision),
                    CHECK((frequency_low_hz IS NULL AND frequency_high_hz IS NULL) OR
                          (frequency_low_hz IS NOT NULL AND frequency_high_hz IS NOT NULL AND frequency_low_hz < frequency_high_hz))
                );
                CREATE INDEX IF NOT EXISTS coverage_by_capture ON scene_coverage_revisions(capture_id, coverage_id, revision DESC);
                CREATE TABLE IF NOT EXISTS mode_registry_entries (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    family TEXT NOT NULL,
                    method TEXT NOT NULL,
                    protocol TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    aliases_json TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    distinguishing_evidence TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    created_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mode_registry_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mode_id TEXT NOT NULL REFERENCES mode_registry_entries(id) ON DELETE CASCADE,
                    decision TEXT NOT NULL CHECK(decision IN ('accepted','rejected')),
                    note TEXT NOT NULL,
                    created_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS mode_decisions_by_mode ON mode_registry_decisions(mode_id, id DESC);
                CREATE TABLE IF NOT EXISTS decoder_runs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    capture_id TEXT REFERENCES captures(id) ON DELETE SET NULL,
                    region_id TEXT REFERENCES regions(id) ON DELETE SET NULL,
                    track_id TEXT NOT NULL,
                    sample_start INTEGER,
                    sample_count INTEGER,
                    sample_rate INTEGER,
                    carrier_hz REAL,
                    auto_identify INTEGER,
                    modulation_hypothesis TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    modes_json TEXT NOT NULL,
                    attempted_modes_json TEXT NOT NULL,
                    current_mode TEXT,
                    output_offset INTEGER NOT NULL,
                    output_end INTEGER,
                    output_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    note TEXT NOT NULL,
                    output_scope TEXT NOT NULL DEFAULT 'shared_live_input',
                    association TEXT NOT NULL DEFAULT 'track_window_tentative',
                    classifier_evidence_json TEXT NOT NULL DEFAULT '{}',
                    detector_evidence_json TEXT NOT NULL DEFAULT '{}',
                    created_utc TEXT NOT NULL,
                    updated_utc TEXT NOT NULL,
                    CHECK(sample_start IS NULL OR sample_start >= 0),
                    CHECK(sample_count IS NULL OR sample_count > 0),
                    CHECK(sample_rate IS NULL OR sample_rate > 0)
                );
                CREATE INDEX IF NOT EXISTS decoder_runs_by_capture ON decoder_runs(capture_id, created_utc DESC);
                CREATE INDEX IF NOT EXISTS decoder_runs_by_session ON decoder_runs(session_id, created_utc DESC);
                CREATE TABLE IF NOT EXISTS demod_bitstreams (
                    id TEXT PRIMARY KEY,
                    capture_id TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
                    stream_sha256 TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    sample_start INTEGER NOT NULL CHECK(sample_start >= 0),
                    sample_count INTEGER NOT NULL CHECK(sample_count > 0),
                    settings_json TEXT NOT NULL,
                    hypothesis TEXT NOT NULL,
                    bits TEXT NOT NULL,
                    created_utc TEXT NOT NULL,
                    UNIQUE(capture_id, stream_sha256)
                );
                CREATE INDEX IF NOT EXISTS bitstreams_by_capture ON demod_bitstreams(capture_id, created_utc);
                CREATE TABLE IF NOT EXISTS raw_demod_attempts (
                    id TEXT PRIMARY KEY,
                    capture_id TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
                    mode TEXT NOT NULL,
                    sample_start INTEGER NOT NULL CHECK(sample_start >= 0),
                    sample_count INTEGER NOT NULL CHECK(sample_count > 0),
                    settings_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS raw_attempts_by_capture ON raw_demod_attempts(capture_id, created_utc DESC);
                CREATE TABLE IF NOT EXISTS bit_annotations (
                    id TEXT PRIMARY KEY,
                    bitstream_id TEXT NOT NULL REFERENCES demod_bitstreams(id) ON DELETE CASCADE,
                    bit_start INTEGER NOT NULL CHECK(bit_start >= 0),
                    bit_count INTEGER NOT NULL CHECK(bit_count > 0),
                    label TEXT NOT NULL,
                    color TEXT NOT NULL,
                    created_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS bit_annotations_by_stream ON bit_annotations(bitstream_id, created_utc);
                CREATE TABLE IF NOT EXISTS protocol_stencils (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    field TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    protocols_json TEXT NOT NULL,
                    source_capture_id TEXT REFERENCES captures(id) ON DELETE SET NULL,
                    created_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS protocol_stencils_by_name ON protocol_stencils(name);
                CREATE TABLE IF NOT EXISTS cv_proposal_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    capture_id TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
                    proposal_id TEXT NOT NULL,
                    model_sha256 TEXT NOT NULL CHECK(length(model_sha256)=64),
                    decision TEXT NOT NULL CHECK(decision IN ('dismissed','restored')),
                    sample_start INTEGER NOT NULL CHECK(sample_start >= 0),
                    sample_count INTEGER NOT NULL CHECK(sample_count > 0),
                    frequency_low_hz REAL NOT NULL,
                    frequency_high_hz REAL NOT NULL,
                    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                    detector_evidence_json TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_utc TEXT NOT NULL,
                    CHECK(frequency_low_hz < frequency_high_hz)
                );
                CREATE INDEX IF NOT EXISTS cv_proposal_decisions_by_capture
                    ON cv_proposal_decisions(capture_id, model_sha256, proposal_id, id DESC);
                CREATE TABLE IF NOT EXISTS annotation_save_batches (
                    idempotency_key TEXT PRIMARY KEY,
                    capture_id TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
                    request_sha256 TEXT NOT NULL CHECK(length(request_sha256)=64),
                    result_region_ids_json TEXT NOT NULL,
                    created_utc TEXT NOT NULL
                );
                """
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(decoder_runs)")}
            if "output_scope" not in columns:
                db.execute("ALTER TABLE decoder_runs ADD COLUMN output_scope TEXT NOT NULL DEFAULT 'shared_live_input'")
            if "association" not in columns:
                db.execute("ALTER TABLE decoder_runs ADD COLUMN association TEXT NOT NULL DEFAULT 'track_window_tentative'")
            if "carrier_hz" not in columns:
                db.execute("ALTER TABLE decoder_runs ADD COLUMN carrier_hz REAL")
            if "auto_identify" not in columns:
                db.execute("ALTER TABLE decoder_runs ADD COLUMN auto_identify INTEGER")
            if "region_id" not in columns:
                db.execute("ALTER TABLE decoder_runs ADD COLUMN region_id TEXT REFERENCES regions(id) ON DELETE SET NULL")
            if "classifier_evidence_json" not in columns:
                db.execute("ALTER TABLE decoder_runs ADD COLUMN classifier_evidence_json TEXT NOT NULL DEFAULT '{}'")
            if "detector_evidence_json" not in columns:
                db.execute("ALTER TABLE decoder_runs ADD COLUMN detector_evidence_json TEXT NOT NULL DEFAULT '{}'")
            db.execute("CREATE INDEX IF NOT EXISTS decoder_runs_by_region ON decoder_runs(region_id, created_utc DESC)")
            region_columns = {row[1] for row in db.execute("PRAGMA table_info(regions)")}
            if "origin" not in region_columns:
                db.execute("ALTER TABLE regions ADD COLUMN origin TEXT NOT NULL DEFAULT 'manual'")
            if "signal_id" not in region_columns:
                db.execute("ALTER TABLE regions ADD COLUMN signal_id TEXT REFERENCES signal_identities(id)")
            db.execute("CREATE INDEX IF NOT EXISTS regions_by_signal_identity ON regions(signal_id)")
            for row in db.execute(
                "SELECT id FROM regions WHERE signal_id IS NULL ORDER BY created_utc,id"
            ).fetchall():
                signal_id = self._create_signal_identity(db)
                db.execute("UPDATE regions SET signal_id=? WHERE id=?", (signal_id, row["id"]))
                db.execute(
                    "INSERT OR IGNORE INTO region_signal_identity_revisions (region_id,revision,signal_id,created_utc) VALUES (?,1,?,?)",
                    (row["id"], signal_id, datetime.now(timezone.utc).isoformat()),
                )
            label_columns = {row[1] for row in db.execute("PRAGMA table_info(label_revisions)")}
            if "mode_id" not in label_columns:
                db.execute("ALTER TABLE label_revisions ADD COLUMN mode_id TEXT")
            db.execute("""INSERT OR IGNORE INTO region_geometry_revisions
                (region_id, revision, sample_start, sample_count, frequency_low_hz, frequency_high_hz, created_utc)
                SELECT id, 1, sample_start, sample_count, frequency_low_hz, frequency_high_hz, created_utc FROM regions""")
            feedback_columns = {row[1] for row in db.execute("PRAGMA table_info(annotation_feedback)")}
            if "geometry_revision" not in feedback_columns:
                db.execute("ALTER TABLE annotation_feedback ADD COLUMN geometry_revision INTEGER")
                db.execute("""UPDATE annotation_feedback SET geometry_revision=(
                    SELECT MAX(revision) FROM region_geometry_revisions WHERE region_id=annotation_feedback.region_id
                ) WHERE geometry_revision IS NULL""")
            if "resulting_revision" not in feedback_columns:
                db.execute("ALTER TABLE annotation_feedback ADD COLUMN resulting_revision INTEGER")
            db.execute("PRAGMA user_version = 22")

    @staticmethod
    def _create_signal_identity(db: sqlite3.Connection) -> str:
        existing = db.execute("SELECT id FROM signal_identities").fetchall()
        maximum = max(
            (int(row["id"][1:]) for row in existing if re.fullmatch(r"S[0-9]+", row["id"])),
            default=0,
        )
        signal_id = f"S{maximum + 1}"
        db.execute(
            "INSERT INTO signal_identities (id,created_utc) VALUES (?,?)",
            (signal_id, datetime.now(timezone.utc).isoformat()),
        )
        return signal_id

    @staticmethod
    def _record_region_signal_identity(
        db: sqlite3.Connection, region_id: str, signal_id: str,
    ) -> bool:
        if db.execute("SELECT 1 FROM signal_identities WHERE id=?", (signal_id,)).fetchone() is None:
            raise ValueError(f"unknown signal identity: {signal_id}")
        current = db.execute("SELECT signal_id FROM regions WHERE id=?", (region_id,)).fetchone()
        if current is None:
            raise KeyError(f"unknown region: {region_id}")
        if current["signal_id"] == signal_id:
            return False
        latest = db.execute(
            "SELECT MAX(revision) AS revision FROM region_signal_identity_revisions WHERE region_id=?",
            (region_id,),
        ).fetchone()["revision"] or 0
        revision = int(latest) + 1
        created = datetime.now(timezone.utc).isoformat()
        db.execute("UPDATE regions SET signal_id=? WHERE id=?", (signal_id, region_id))
        db.execute(
            "INSERT INTO region_signal_identity_revisions (region_id,revision,signal_id,created_utc) VALUES (?,?,?,?)",
            (region_id, revision, signal_id, created),
        )
        return True

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as audio:
            for block in iter(lambda: audio.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def add_wav_capture(
        self,
        path: str | Path,
        *,
        source_kind: str = "wav",
        parent_capture_id: str | None = None,
        source_metadata: dict[str, Any] | None = None,
    ) -> str:
        if source_kind not in SOURCE_KINDS:
            raise ValueError(f"unknown source_kind: {source_kind}")
        audio_path = Path(path).resolve(strict=True)
        metadata = WavSource(audio_path).metadata()
        capture_id = str(uuid4())
        extra = dict(source_metadata or {})
        extra.update({"sample_width_bytes": metadata["sample_width_bytes"]})
        with self._connect() as db:
            db.execute(
                """INSERT INTO captures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    capture_id,
                    datetime.now(timezone.utc).isoformat(),
                    source_kind,
                    str(audio_path),
                    self._sha256(audio_path),
                    metadata["sample_rate"],
                    metadata["channels"],
                    metadata["sample_count"],
                    json.dumps(extra, sort_keys=True),
                    parent_capture_id,
                ),
            )
        return capture_id

    def add_region(
        self,
        capture_id: str,
        *,
        sample_start: int,
        sample_count: int,
        frequency_low_hz: float | None = None,
        frequency_high_hz: float | None = None,
        label: SignalLabel | None = None,
        origin: str = "manual",
    ) -> str:
        if sample_start < 0 or sample_count <= 0:
            raise ValueError("region sample_start must be non-negative and sample_count positive")
        if (frequency_low_hz is None) != (frequency_high_hz is None):
            raise ValueError("both frequency bounds must be provided together")
        if frequency_low_hz is not None and frequency_low_hz >= frequency_high_hz:
            raise ValueError("frequency_low_hz must be less than frequency_high_hz")
        if origin not in {"manual", "detector_proposal", "user_added_missed_detection"}:
            raise ValueError("unknown region origin")
        chosen_label = label or SignalLabel()
        self._validate_label(chosen_label, self.accepted_mode_definitions())

        region_id = str(uuid4())
        with self._connect() as db:
            capture = db.execute(
                "SELECT sample_count FROM captures WHERE id = ?", (capture_id,)
            ).fetchone()
            if capture is None:
                raise KeyError(f"unknown capture: {capture_id}")
            if sample_start + sample_count > capture["sample_count"]:
                raise ValueError("region exceeds the parent capture sample range")
            signal_id = self._create_signal_identity(db)
            db.execute(
                "INSERT INTO regions (id, capture_id, sample_start, sample_count, frequency_low_hz, frequency_high_hz, origin, signal_id, created_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    region_id,
                    capture_id,
                    sample_start,
                    sample_count,
                    frequency_low_hz,
                    frequency_high_hz,
                    origin,
                    signal_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            db.execute(
                "INSERT INTO region_signal_identity_revisions (region_id,revision,signal_id,created_utc) VALUES (?,1,?,?)",
                (region_id, signal_id, datetime.now(timezone.utc).isoformat()),
            )
            db.execute(
                """INSERT INTO region_geometry_revisions
                   (region_id, revision, sample_start, sample_count, frequency_low_hz, frequency_high_hz, created_utc)
                   VALUES (?, 1, ?, ?, ?, ?, ?)""",
                (region_id, sample_start, sample_count, frequency_low_hz, frequency_high_hz, datetime.now(timezone.utc).isoformat()),
            )
            self._insert_label(db, region_id, 1, chosen_label)
        return region_id

    def add_regions_batch(
        self, capture_id: str, idempotency_key: str, regions: list[dict[str, Any]], *,
        updates: list[dict[str, Any]] | None = None,
        deletions: list[dict[str, Any]] | None = None,
        merges: list[dict[str, Any]] | None = None,
        identity_links: list[dict[str, Any]] | None = None,
        window_start_sample: int = 0,
        window_sample_count: int | None = None,
    ) -> tuple[list[str], bool]:
        """Save new, edited, and removed annotations atomically for one window."""
        updates = updates or []
        deletions = deletions or []
        merges = merges or []
        identity_links = identity_links or []
        if not idempotency_key or len(idempotency_key) > 80:
            raise ValueError("annotation batch id must contain 1 to 80 characters")
        if not 1 <= len(regions) + len(updates) + len(deletions) + len(merges) + len(identity_links) <= 128:
            raise ValueError("a window save must contain one to 128 changes")

        canonical: list[dict[str, Any]] = []
        canonical_updates: list[dict[str, Any]] = []
        canonical_merges: list[dict[str, Any]] = []
        additional_modes = self.accepted_mode_definitions()
        for item in regions:
            label = item["label"]
            if not isinstance(label, SignalLabel):
                raise ValueError("annotation batch labels must be validated SignalLabel objects")
            self._validate_label(label, additional_modes)
            canonical.append({
                "client_id": str(item["client_id"]),
                "sample_start": item["sample_start"],
                "sample_count": item["sample_count"],
                "frequency_low_hz": item["frequency_low_hz"],
                "frequency_high_hz": item["frequency_high_hz"],
                "origin": item.get("origin", "manual"),
                "label": {
                    "family": label.family, "method": label.method,
                    "protocol": label.protocol, "variant": label.variant,
                    "mode_id": label.mode_id, "source": label.source,
                    "review_state": label.review_state, "evidence": label.evidence,
                    "notes": label.notes,
                },
            })
        for item in updates:
            label = item["label"]
            if not isinstance(label, SignalLabel):
                raise ValueError("window update labels must be validated SignalLabel objects")
            self._validate_label(label, additional_modes)
            canonical_updates.append({
                "region_id": str(item["region_id"]),
                "expected_revision": int(item["expected_revision"]),
                "expected_geometry_revision": int(item["expected_geometry_revision"]),
                "sample_start": item["sample_start"],
                "sample_count": item["sample_count"],
                "frequency_low_hz": item["frequency_low_hz"],
                "frequency_high_hz": item["frequency_high_hz"],
                "label": {
                    "family": label.family, "method": label.method,
                    "protocol": label.protocol, "variant": label.variant,
                    "mode_id": label.mode_id, "source": label.source,
                    "review_state": label.review_state, "evidence": label.evidence,
                    "notes": label.notes,
                },
            })
        canonical_deletions = [{
            "region_id": str(item["region_id"]),
            "expected_revision": int(item["expected_revision"]),
            "expected_geometry_revision": int(item["expected_geometry_revision"]),
        } for item in deletions]
        canonical_identity_links = []
        for item in identity_links:
            link = item if isinstance(item, dict) else item.model_dump()
            region_ids = [str(region_id) for region_id in link.get("region_ids", [])]
            client_ids = [str(client_id) for client_id in link.get("client_ids", [])]
            expected_signal_ids = {str(region_id): str(signal_id) for region_id, signal_id in link.get("expected_signal_ids", {}).items()}
            expected_revisions = {str(region_id): int(revision) for region_id, revision in link.get("expected_revisions", {}).items()}
            if not region_ids and not client_ids:
                raise ValueError("an identity link needs at least one saved region or draft box")
            if len(set(region_ids)) != len(region_ids) or len(set(client_ids)) != len(client_ids):
                raise ValueError("identity link sources must be unique")
            if set(expected_signal_ids) != set(region_ids):
                raise ValueError("expected signal IDs must identify every saved box being linked")
            if set(expected_revisions) != set(region_ids):
                raise ValueError("expected label revisions must identify every saved box being linked")
            if bool(link.get("signal_id")) == bool(link.get("target_region_id") or link.get("target_client_id")):
                raise ValueError("choose an existing signal ID or one selected box as the identity target")
            canonical_identity_links.append({
                "signal_id": link.get("signal_id"),
                "target_region_id": link.get("target_region_id"),
                "target_client_id": link.get("target_client_id"),
                "region_ids": region_ids,
                "client_ids": client_ids,
                "expected_signal_ids": expected_signal_ids,
                "expected_revisions": expected_revisions,
            })
        for item in merges:
            label = item["label"]
            if not isinstance(label, SignalLabel):
                raise ValueError("window merge labels must be validated SignalLabel objects")
            self._validate_label(label, additional_modes)
            source_ids = [str(region_id) for region_id in item.get("source_region_ids", [])]
            expected_revisions = {str(region_id): int(revision) for region_id, revision in item.get("expected_revisions", {}).items()}
            expected_geometry_revisions = {str(region_id): int(revision) for region_id, revision in item.get("expected_geometry_revisions", {}).items()}
            proposals = item.get("cv_proposals", [])
            proposal_ids = [str(proposal.get("proposal_id") or "") for proposal in proposals]
            heuristic_proposals = item.get("heuristic_proposals", [])
            heuristic_ids = [str(proposal.get("candidate_id") or "") for proposal in heuristic_proposals]
            if not 1 <= len(source_ids) + len(proposals) + len(heuristic_proposals) <= 32:
                raise ValueError("a window merge needs one to 32 saved or provisional source boxes")
            if (len(set(source_ids)) != len(source_ids)
                or set(expected_revisions) != set(source_ids)
                or set(expected_geometry_revisions) != set(source_ids)):
                raise ValueError("window merge sources and expected label/geometry revisions must match exactly")
            if len(set(proposal_ids)) != len(proposal_ids) or any(not proposal_id for proposal_id in proposal_ids):
                raise ValueError("CV proposal ids must be present and unique within a merge")
            if len(set(heuristic_ids)) != len(heuristic_ids) or any(not candidate_id for candidate_id in heuristic_ids):
                raise ValueError("heuristic candidate ids must be present and unique within a merge")
            canonical_merges.append({
                "client_id": str(item["client_id"]),
                "source_region_ids": source_ids,
                "expected_geometry_revisions": expected_geometry_revisions,
                "expected_revisions": expected_revisions,
                "cv_proposals": proposals,
                "heuristic_proposals": heuristic_proposals,
                "target_region_id": item.get("target_region_id"),
                "sample_start": item["sample_start"],
                "sample_count": item["sample_count"],
                "frequency_low_hz": item["frequency_low_hz"],
                "frequency_high_hz": item["frequency_high_hz"],
                "label": {
                    "family": label.family, "method": label.method,
                    "protocol": label.protocol, "variant": label.variant,
                    "mode_id": label.mode_id, "source": label.source,
                    "review_state": label.review_state, "evidence": label.evidence,
                    "notes": label.notes,
                },
            })
        if window_start_sample < 0 or (window_sample_count is not None and window_sample_count <= 0):
            raise ValueError("annotation window bounds must be positive")
        digest_payload = {
            "window_start_sample": window_start_sample,
            "window_sample_count": window_sample_count,
            "regions": canonical,
            "updates": canonical_updates,
            "deletions": canonical_deletions,
            "merges": canonical_merges,
            "identity_links": canonical_identity_links,
        }
        digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        region_ids: list[str] = []
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            capture = db.execute("SELECT sample_count, sample_rate FROM captures WHERE id=?", (capture_id,)).fetchone()
            if capture is None:
                raise KeyError(f"unknown capture: {capture_id}")
            existing = db.execute(
                "SELECT capture_id, request_sha256, result_region_ids_json FROM annotation_save_batches WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if existing["capture_id"] != capture_id or existing["request_sha256"] != digest:
                    raise ValueError("this annotation save key was already used for different changes; refresh the window and save again")
                return json.loads(existing["result_region_ids_json"]), True

            window_count = window_sample_count if window_sample_count is not None else int(capture["sample_count"])
            window_end = window_start_sample + window_count
            if window_end > int(capture["sample_count"]) or window_count > int(capture["sample_rate"]) * 10 + 1:
                raise ValueError("annotation saves must stay inside one 10-second capture window")
            merge_source_ids = [region_id for merge in canonical_merges for region_id in merge["source_region_ids"]]
            changed_region_ids = [item["region_id"] for item in canonical_updates + canonical_deletions]
            changed_region_ids.extend(merge_source_ids)
            if len(set(changed_region_ids)) != len(changed_region_ids):
                raise ValueError("each saved region can be edited, removed, or merged only once in a window save")
            identity_region_ids = [region_id for link in canonical_identity_links for region_id in link["region_ids"]]
            if len(set(identity_region_ids)) != len(identity_region_ids):
                raise ValueError("a saved region can participate in only one identity link per window save")
            if set(identity_region_ids) & {item["region_id"] for item in canonical_deletions}:
                raise ValueError("a region cannot be identity-linked and deleted in the same window save")
            if set(identity_region_ids) & set(merge_source_ids):
                raise ValueError("a merge source cannot also be identity-linked in the same window save")
            # Validate link revisions against the original window state before
            # applying label updates in this same transaction.
            for link in canonical_identity_links:
                for region_id in link["region_ids"]:
                    source = db.execute(
                        """SELECT r.signal_id,g.sample_start,g.sample_count,
                                  (SELECT MAX(revision) FROM label_revisions WHERE region_id=r.id) AS label_revision
                           FROM regions r JOIN region_geometry_revisions g ON g.region_id=r.id
                           AND g.revision=(SELECT MAX(revision) FROM region_geometry_revisions WHERE region_id=r.id)
                           WHERE r.id=? AND r.capture_id=?""",
                        (region_id, capture_id),
                    ).fetchone()
                    if source is None:
                        raise KeyError(f"unknown identity-link source in capture: {region_id}")
                    if db.execute("SELECT 1 FROM region_removals WHERE region_id=?", (region_id,)).fetchone():
                        raise LabelRevisionConflict("a removed signal cannot be identity-linked")
                    if db.execute("SELECT 1 FROM region_lineage WHERE source_region_id=? LIMIT 1", (region_id,)).fetchone():
                        raise LabelRevisionConflict("a superseded signal cannot be identity-linked")
                    if source["signal_id"] != link["expected_signal_ids"][region_id]:
                        raise LabelRevisionConflict("a signal identity changed while this window was being edited; refresh it before saving")
                    if source["label_revision"] != link["expected_revisions"][region_id]:
                        raise LabelRevisionConflict("a signal label changed while this window was being edited; refresh it before saving")
                    if (source["sample_start"] < window_start_sample
                        or source["sample_start"] + source["sample_count"] > window_end):
                        raise ValueError("identity-link sources must belong to the selected 10-second window")
            created_regions_by_client: dict[str, str] = {}

            for item in canonical_updates:
                start, count = item["sample_start"], item["sample_count"]
                low, high = item["frequency_low_hz"], item["frequency_high_hz"]
                if not isinstance(start, int) or start < window_start_sample or not isinstance(count, int) or count <= 0 or start + count > window_end:
                    raise ValueError("updated region bounds must stay inside the selected 10-second window")
                if not math.isfinite(low) or not math.isfinite(high) or low < 0 or low >= high or high > capture["sample_rate"] / 2:
                    raise ValueError("updated frequency bounds must be finite, increasing, and below Nyquist")
                current = db.execute(
                    """SELECT r.id,
                              g.revision AS geometry_revision,g.sample_start,g.sample_count,g.frequency_low_hz,g.frequency_high_hz,
                              l.revision AS label_revision,l.family,l.method,l.protocol,l.variant,l.mode_id,l.source,l.review_state,l.evidence_json,l.notes
                       FROM regions r
                       JOIN region_geometry_revisions g ON g.region_id=r.id AND g.revision=(SELECT MAX(revision) FROM region_geometry_revisions WHERE region_id=r.id)
                       JOIN label_revisions l ON l.region_id=r.id AND l.revision=(SELECT MAX(revision) FROM label_revisions WHERE region_id=r.id)
                       WHERE r.id=? AND r.capture_id=?""", (item["region_id"], capture_id),
                ).fetchone()
                if current is None:
                    raise KeyError(f"unknown region in capture: {item['region_id']}")
                if db.execute("SELECT 1 FROM region_removals WHERE region_id=?", (item["region_id"],)).fetchone():
                    raise LabelRevisionConflict("this signal was already removed from the active annotations")
                if current["label_revision"] != item["expected_revision"] or current["geometry_revision"] != item["expected_geometry_revision"]:
                    raise LabelRevisionConflict("a saved signal changed while this window was being edited; refresh it before saving")
                if db.execute("SELECT 1 FROM region_lineage WHERE source_region_id=? LIMIT 1", (item["region_id"],)).fetchone():
                    raise LabelRevisionConflict("a superseded signal cannot be edited")
                if (current["sample_start"] < window_start_sample
                    or current["sample_start"] + current["sample_count"] > window_end):
                    raise ValueError("the saved signal is outside the selected 10-second window")
                if (current["sample_start"], current["sample_count"], current["frequency_low_hz"], current["frequency_high_hz"]) != (start, count, low, high):
                    geometry_revision = int(current["geometry_revision"]) + 1
                    db.execute(
                        """INSERT INTO region_geometry_revisions
                           (region_id,revision,sample_start,sample_count,frequency_low_hz,frequency_high_hz,created_utc)
                           VALUES(?,?,?,?,?,?,?)""",
                        (item["region_id"], geometry_revision, start, count, low, high, now),
                    )
                else:
                    geometry_revision = int(current["geometry_revision"])
                label = SignalLabel(**item["label"])
                prior_evidence = json.loads(current["evidence_json"])
                if (current["family"], current["method"], current["protocol"], current["variant"], current["mode_id"], current["source"], current["review_state"], prior_evidence, current["notes"]) != (
                    label.family, label.method, label.protocol, label.variant, label.mode_id,
                    label.source, label.review_state, label.evidence or {}, label.notes,
                ):
                    self._insert_label(db, item["region_id"], int(current["label_revision"]) + 1, label)
                    identity = db.execute("SELECT signal_id FROM regions WHERE id=?", (item["region_id"],)).fetchone()
                    if identity and identity["signal_id"]:
                        peers = db.execute(
                            """SELECT r.id,l.* FROM regions r JOIN label_revisions l ON l.region_id=r.id
                               AND l.revision=(SELECT MAX(revision) FROM label_revisions WHERE region_id=r.id)
                               WHERE r.signal_id=? AND r.id<>?
                               AND NOT EXISTS(SELECT 1 FROM region_removals rr WHERE rr.region_id=r.id)
                               AND NOT EXISTS(SELECT 1 FROM region_lineage rl WHERE rl.source_region_id=r.id)""",
                            (identity["signal_id"], item["region_id"]),
                        ).fetchall()
                        for peer in peers:
                            shared_fields = (label.family, label.method, label.protocol, label.variant, label.mode_id)
                            prior_fields = (peer["family"], peer["method"], peer["protocol"], peer["variant"], peer["mode_id"])
                            if shared_fields == prior_fields:
                                continue
                            peer_label = SignalLabel(
                                family=label.family, method=label.method, protocol=label.protocol,
                                variant=label.variant, mode_id=label.mode_id,
                                source=peer["source"], review_state=peer["review_state"],
                                evidence=json.loads(peer["evidence_json"]), notes=peer["notes"],
                            )
                            self._insert_label(db, peer["id"], int(peer["revision"]) + 1, peer_label)
                region_ids.append(item["region_id"])

            for item in canonical_deletions:
                current = db.execute(
                    """SELECT (SELECT MAX(revision) FROM label_revisions WHERE region_id=r.id) AS label_revision,
                              g.revision AS geometry_revision,g.sample_start,g.sample_count
                       FROM regions r JOIN region_geometry_revisions g ON g.region_id=r.id AND g.revision=(SELECT MAX(revision) FROM region_geometry_revisions WHERE region_id=r.id)
                       WHERE r.id=? AND r.capture_id=?""", (item["region_id"], capture_id),
                ).fetchone()
                if current is None:
                    raise KeyError(f"unknown region in capture: {item['region_id']}")
                if db.execute("SELECT 1 FROM region_removals WHERE region_id=?", (item["region_id"],)).fetchone():
                    raise LabelRevisionConflict("this signal was already removed from the active annotations")
                if current["label_revision"] != item["expected_revision"] or current["geometry_revision"] != item["expected_geometry_revision"]:
                    raise LabelRevisionConflict("a saved signal changed while this window was being edited; refresh it before saving")
                if (current["sample_start"] < window_start_sample or current["sample_start"] + current["sample_count"] > window_end):
                    raise ValueError("the saved signal is outside the selected 10-second window")
                lineage_source = db.execute(
                    "SELECT 1 FROM region_lineage WHERE source_region_id=? LIMIT 1", (item["region_id"],)
                ).fetchone()
                lineage_result = db.execute(
                    "SELECT 1 FROM region_lineage WHERE result_region_id=? LIMIT 1", (item["region_id"],)
                ).fetchone()
                if lineage_source is not None:
                    raise ValueError("a superseded signal cannot be removed; remove its active replacement instead")
                if lineage_result is not None:
                    db.execute(
                        """INSERT INTO region_removals
                           (region_id,capture_id,label_revision,geometry_revision,reason,created_utc)
                           VALUES(?,?,?,?,?,?)""",
                        (item["region_id"], capture_id, item["expected_revision"],
                         item["expected_geometry_revision"], "Removed during annotation review.", now),
                    )
                else:
                    db.execute("DELETE FROM regions WHERE id=? AND capture_id=?", (item["region_id"], capture_id))

            for item in canonical:
                start = item["sample_start"]
                count = item["sample_count"]
                low = item["frequency_low_hz"]
                high = item["frequency_high_hz"]
                if not isinstance(start, int) or start < 0 or not isinstance(count, int) or count <= 0:
                    raise ValueError("region sample bounds must be non-negative with positive length")
                if start + count > capture["sample_count"]:
                    raise ValueError("region exceeds the parent capture sample range")
                if start < window_start_sample or start + count > window_end:
                    raise ValueError("new region bounds must stay inside the selected 10-second window")
                if low is None or high is None or not math.isfinite(low) or not math.isfinite(high) or low < 0 or low >= high or high > capture["sample_rate"] / 2:
                    raise ValueError("frequency bounds must be finite, increasing, and below Nyquist")
                origin = item["origin"]
                if origin not in {"manual", "detector_proposal", "user_added_missed_detection"}:
                    raise ValueError("unknown region origin")
                region_id = str(uuid4())
                signal_id = self._create_signal_identity(db)
                db.execute(
                    "INSERT INTO regions (id,capture_id,sample_start,sample_count,frequency_low_hz,frequency_high_hz,origin,signal_id,created_utc) VALUES (?,?,?,?,?,?,?,?,?)",
                    (region_id, capture_id, start, count, low, high, origin, signal_id, now),
                )
                db.execute(
                    "INSERT INTO region_signal_identity_revisions (region_id,revision,signal_id,created_utc) VALUES (?,1,?,?)",
                    (region_id, signal_id, now),
                )
                db.execute(
                    "INSERT INTO region_geometry_revisions (region_id,revision,sample_start,sample_count,frequency_low_hz,frequency_high_hz,created_utc) VALUES (?,1,?,?,?,?,?)",
                    (region_id, start, count, low, high, now),
                )
                self._insert_label(db, region_id, 1, SignalLabel(**item["label"]))
                region_ids.append(region_id)
                created_regions_by_client[item["client_id"]] = region_id

            for link in canonical_identity_links:
                target_signal_id = link.get("signal_id")
                if link.get("target_region_id"):
                    target = db.execute(
                        "SELECT signal_id,capture_id,sample_start,sample_count FROM regions WHERE id=?",
                        (link["target_region_id"],),
                    ).fetchone()
                    if target is None or target["capture_id"] != capture_id:
                        raise ValueError("identity target region must belong to this capture")
                    target_signal_id = target["signal_id"]
                elif link.get("target_client_id"):
                    target_region_id = created_regions_by_client.get(link["target_client_id"])
                    if target_region_id is None:
                        raise ValueError("identity target draft box was not included in this window save")
                    target = db.execute("SELECT signal_id FROM regions WHERE id=?", (target_region_id,)).fetchone()
                    target_signal_id = target["signal_id"]
                if not target_signal_id or not re.fullmatch(r"S[1-9][0-9]*", str(target_signal_id)):
                    raise ValueError("identity target does not have a valid signal ID")
                if db.execute("SELECT 1 FROM signal_identities WHERE id=?", (target_signal_id,)).fetchone() is None:
                    raise ValueError(f"unknown signal identity: {target_signal_id}")

                linked_region_ids = list(link["region_ids"])
                linked_region_ids.extend(
                    created_regions_by_client[client_id]
                    for client_id in link["client_ids"]
                    if client_id in created_regions_by_client
                )
                if len(linked_region_ids) != len(link["region_ids"]) + len(link["client_ids"]):
                    raise ValueError("identity link references a draft box not included in this window save")
                for region_id in link["region_ids"]:
                    source = db.execute(
                        """SELECT r.signal_id,g.sample_start,g.sample_count,
                                  (SELECT MAX(revision) FROM label_revisions WHERE region_id=r.id) AS label_revision
                           FROM regions r JOIN region_geometry_revisions g ON g.region_id=r.id
                           AND g.revision=(SELECT MAX(revision) FROM region_geometry_revisions WHERE region_id=r.id)
                           WHERE r.id=? AND r.capture_id=?""",
                        (region_id, capture_id),
                    ).fetchone()
                    if source is None:
                        raise KeyError(f"unknown identity-link source in capture: {region_id}")
                    if db.execute("SELECT 1 FROM region_removals WHERE region_id=?", (region_id,)).fetchone():
                        raise LabelRevisionConflict("a removed signal cannot be identity-linked")
                    if db.execute("SELECT 1 FROM region_lineage WHERE source_region_id=? LIMIT 1", (region_id,)).fetchone():
                        raise LabelRevisionConflict("a superseded signal cannot be identity-linked")
                    if (source["sample_start"] < window_start_sample
                        or source["sample_start"] + source["sample_count"] > window_end):
                        raise ValueError("identity-link sources must belong to the selected 10-second window")
                for region_id in linked_region_ids:
                    if self._record_region_signal_identity(db, region_id, target_signal_id):
                        region_ids.append(region_id)
                # Linking identity also adopts the current taxonomy of that identity.
                representative = db.execute(
                    """SELECT l.* FROM regions r JOIN label_revisions l ON l.region_id=r.id
                       AND l.revision=(SELECT MAX(revision) FROM label_revisions WHERE region_id=r.id)
                       WHERE r.signal_id=?
                       AND NOT EXISTS(SELECT 1 FROM region_removals rr WHERE rr.region_id=r.id)
                       AND NOT EXISTS(SELECT 1 FROM region_lineage rl WHERE rl.source_region_id=r.id)
                       ORDER BY l.review_state='accepted' DESC, l.mode_id IS NOT NULL DESC LIMIT 1""",
                    (target_signal_id,),
                ).fetchone()
                for region_id in linked_region_ids:
                    current = db.execute(
                        "SELECT l.* FROM label_revisions l WHERE l.region_id=? ORDER BY revision DESC LIMIT 1",
                        (region_id,),
                    ).fetchone()
                    if current is None:
                        continue
                    if representative is None:
                        continue
                    fields = ("family", "method", "protocol", "variant", "mode_id")
                    if tuple(current[field] for field in fields) == tuple(representative[field] for field in fields):
                        continue
                    self._insert_label(db, region_id, int(current["revision"]) + 1, SignalLabel(
                        family=representative["family"], method=representative["method"],
                        protocol=representative["protocol"], variant=representative["variant"],
                        mode_id=representative["mode_id"], source=current["source"],
                        review_state=current["review_state"], evidence=json.loads(current["evidence_json"]),
                        notes=current["notes"],
                    ))

            for merge in canonical_merges:
                bounds = {
                    "sample_start": merge["sample_start"],
                    "sample_count": merge["sample_count"],
                    "frequency_low_hz": merge["frequency_low_hz"],
                    "frequency_high_hz": merge["frequency_high_hz"],
                }
                sample_start, sample_count, low, high = self._validate_lineage_geometry(
                    bounds, capture_samples=int(capture["sample_count"]), sample_rate=int(capture["sample_rate"]),
                )
                if sample_start < window_start_sample or sample_start + sample_count > window_end:
                    raise ValueError("merge result bounds must stay inside the selected 10-second window")
                target_region_id = merge["target_region_id"]
                if target_region_id and target_region_id not in merge["source_region_ids"]:
                    raise ValueError("saved merge target must also be one of its source regions")
                all_source_ids = list(merge["source_region_ids"])
                for region_id in merge["source_region_ids"]:
                    source = db.execute(
                        """SELECT g.revision,g.sample_start,g.sample_count,g.frequency_low_hz,g.frequency_high_hz,
                                  l.revision AS label_revision
                           FROM regions r JOIN region_geometry_revisions g ON g.region_id=r.id
                           AND g.revision=(SELECT MAX(revision) FROM region_geometry_revisions WHERE region_id=r.id)
                           JOIN label_revisions l ON l.region_id=r.id
                           AND l.revision=(SELECT MAX(revision) FROM label_revisions WHERE region_id=r.id)
                           WHERE r.id=? AND r.capture_id=?""", (region_id, capture_id),
                    ).fetchone()
                    if source is None:
                        raise KeyError(f"unknown merge source in capture: {region_id}")
                    if db.execute("SELECT 1 FROM region_removals WHERE region_id=?", (region_id,)).fetchone():
                        raise LabelRevisionConflict("a removed signal cannot be used as a merge source")
                    if int(source["revision"]) != merge["expected_geometry_revisions"][region_id]:
                        raise LabelRevisionConflict("a saved merge source changed while this window was being edited; refresh it before saving")
                    if int(source["label_revision"]) != merge["expected_revisions"][region_id]:
                        raise LabelRevisionConflict("a saved merge source label changed while this window was being edited; refresh it before saving")
                    if db.execute("SELECT 1 FROM region_lineage WHERE source_region_id=? LIMIT 1", (region_id,)).fetchone():
                        raise LabelRevisionConflict("a selected merge source is already superseded")
                    if (source["sample_start"] < window_start_sample
                        or source["sample_start"] + source["sample_count"] > window_end):
                        raise ValueError("all saved merge sources must belong to the selected 10-second window")

                for proposal in merge["cv_proposals"]:
                    evidence = proposal["detector_evidence"]
                    proposal_id = proposal["proposal_id"]
                    model_sha256 = evidence.get("model_sha256")
                    proposal_start = int(proposal["sample_start"])
                    proposal_count = int(proposal["sample_count"])
                    proposal_low = float(proposal["frequency_low_hz"])
                    proposal_high = float(proposal["frequency_high_hz"])
                    confidence = float(proposal["confidence"])
                    if proposal.get("decision") != "dismissed":
                        raise ValueError("CV proposals in a merge must be dismissed from the current scan")
                    if not re.fullmatch(r"[0-9a-f]{64}", str(model_sha256)):
                        raise ValueError("CV proposal model hash must be a lowercase SHA-256 digest")
                    if not window_start_sample <= proposal_start or proposal_start + proposal_count > window_end:
                        raise ValueError("CV merge sources must stay inside the selected 10-second window")
                    if not 0 <= proposal_low < proposal_high <= min(3000, int(capture["sample_rate"]) / 2) or not 0 <= confidence <= 1:
                        raise ValueError("CV proposal bounds or confidence are invalid")
                    if evidence.get("sample_rate") != int(capture["sample_rate"]):
                        raise ValueError("CV proposal evidence sample rate does not match the capture")
                    source_windows = [
                        {"sample_start": evidence.get("window_sample_start"), "sample_count": evidence.get("window_sample_count")},
                        *evidence.get("supporting_windows", []),
                    ]
                    if not any(
                        isinstance(window.get("sample_start"), int)
                        and isinstance(window.get("sample_count"), int)
                        and window["sample_start"] <= proposal_start
                        and window["sample_start"] + window["sample_count"] >= proposal_start + proposal_count
                        for window in source_windows
                    ):
                        raise ValueError("CV proposal bounds are outside their supporting detector windows")
                    decision = db.execute(
                        """SELECT decision FROM cv_proposal_decisions
                           WHERE capture_id=? AND proposal_id=? AND model_sha256=? ORDER BY id DESC LIMIT 1""",
                        (capture_id, proposal_id, model_sha256),
                    ).fetchone()
                    if decision is not None and decision["decision"] == "dismissed":
                        raise LabelRevisionConflict("a selected CV proposal was already dismissed; refresh proposals before saving")
                    cv_region_id = self._insert_lineage_region(
                        db, capture_id, (proposal_start, proposal_count, proposal_low, proposal_high),
                        origin="detector_proposal",
                        label=SignalLabel(
                            source="unknown", review_state="pending",
                            evidence={"detector_proposal": evidence},
                            notes="CV proposal merged into a user-selected signal; review evidence only, not a negative label.",
                        ),
                    )
                    db.execute(
                        """INSERT INTO cv_proposal_decisions
                           (capture_id,proposal_id,model_sha256,decision,sample_start,sample_count,
                            frequency_low_hz,frequency_high_hz,confidence,detector_evidence_json,note,created_utc)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (capture_id, proposal_id, model_sha256, "dismissed", proposal_start, proposal_count,
                         proposal_low, proposal_high, confidence, json.dumps(evidence, sort_keys=True),
                         "Merged into a user-selected signal; review evidence only, not a negative label.", now),
                    )
                    all_source_ids.append(cv_region_id)

                for proposal in merge["heuristic_proposals"]:
                    candidate_id = str(proposal["candidate_id"])
                    proposal_start = int(proposal["sample_start"])
                    proposal_count = int(proposal["sample_count"])
                    proposal_low = float(proposal["frequency_low_hz"])
                    proposal_high = float(proposal["frequency_high_hz"])
                    if not candidate_id or len(candidate_id) > 80:
                        raise ValueError("heuristic candidate id must contain 1 to 80 characters")
                    if not window_start_sample <= proposal_start or proposal_start + proposal_count > window_end:
                        raise ValueError("heuristic merge sources must stay inside the selected 10-second window")
                    if not 0 <= proposal_low < proposal_high <= int(capture["sample_rate"]) / 2:
                        raise ValueError("heuristic candidate frequency bounds must fit below Nyquist")
                    if proposal_count <= 0 or proposal_start + proposal_count > int(capture["sample_count"]):
                        raise ValueError("heuristic candidate time bounds exceed the capture")
                    evidence = {
                        "candidate_id": candidate_id,
                        "modulation_hypothesis": str(proposal["modulation_hypothesis"]),
                        "tone_count_candidate": int(proposal["tone_count_candidate"]),
                        "tone_centers_hz": proposal.get("tone_centers_hz", []),
                        "peak_snr_db": proposal.get("peak_snr_db"),
                    }
                    heuristic_region_id = self._insert_lineage_region(
                        db, capture_id,
                        (proposal_start, proposal_count, proposal_low, proposal_high),
                        origin="heuristic_proposal",
                        label=SignalLabel(
                            source="unknown", review_state="pending",
                            evidence={"heuristic_proposal": evidence},
                            notes="Heuristic proposal merged into a user-selected signal; proposal evidence only, not a negative label.",
                        ),
                    )
                    all_source_ids.append(heuristic_region_id)

                label = SignalLabel(**merge["label"])
                relation_id = str(uuid4())
                merge_evidence = {
                    **(label.evidence or {}),
                    "window_annotation_merge": {
                        "client_id": merge["client_id"],
                        "source_region_ids": all_source_ids,
                        "target_region_id": target_region_id,
                    },
                }
                result_label = SignalLabel(
                    family=label.family, method=label.method, protocol=label.protocol,
                    variant=label.variant, mode_id=label.mode_id, source=label.source,
                    review_state=label.review_state, evidence=merge_evidence, notes=label.notes,
                )
                self._validate_label(result_label, additional_modes)
                if target_region_id:
                    target_current = db.execute(
                        """SELECT g.revision AS geometry_revision,g.sample_start,g.sample_count,
                                  g.frequency_low_hz,g.frequency_high_hz,
                                  l.revision AS label_revision,l.family,l.method,l.protocol,l.variant,
                                  l.mode_id,l.source,l.review_state,l.evidence_json,l.notes
                           FROM regions r JOIN region_geometry_revisions g ON g.region_id=r.id
                           AND g.revision=(SELECT MAX(revision) FROM region_geometry_revisions WHERE region_id=r.id)
                           JOIN label_revisions l ON l.region_id=r.id
                           AND l.revision=(SELECT MAX(revision) FROM label_revisions WHERE region_id=r.id)
                           WHERE r.id=? AND r.capture_id=?""",
                        (target_region_id, capture_id),
                    ).fetchone()
                    if target_current is None:
                        raise KeyError(f"unknown merge target in capture: {target_region_id}")
                    if db.execute("SELECT 1 FROM region_lineage WHERE source_region_id=? LIMIT 1", (target_region_id,)).fetchone():
                        raise LabelRevisionConflict("the selected merge target is already superseded")
                    if ((target_current["sample_start"], target_current["sample_count"],
                         target_current["frequency_low_hz"], target_current["frequency_high_hz"])
                            != (sample_start, sample_count, low, high)):
                        db.execute(
                            """INSERT INTO region_geometry_revisions
                               (region_id,revision,sample_start,sample_count,frequency_low_hz,frequency_high_hz,created_utc)
                               VALUES(?,?,?,?,?,?,?)""",
                            (target_region_id, int(target_current["geometry_revision"]) + 1,
                             sample_start, sample_count, low, high, now),
                        )
                    prior_evidence = json.loads(target_current["evidence_json"])
                    if (target_current["family"], target_current["method"], target_current["protocol"],
                        target_current["variant"], target_current["mode_id"], target_current["source"],
                        target_current["review_state"], prior_evidence, target_current["notes"]) != (
                            result_label.family, result_label.method, result_label.protocol,
                            result_label.variant, result_label.mode_id, result_label.source,
                            result_label.review_state, result_label.evidence or {}, result_label.notes,
                        ):
                        self._insert_label(db, target_region_id, int(target_current["label_revision"]) + 1, result_label)
                    result_id = target_region_id
                else:
                    result_id = self._insert_lineage_region(
                        db, capture_id, (sample_start, sample_count, low, high), label=result_label,
                    )
                lineage_sources = [source_id for source_id in all_source_ids if source_id != result_id]
                self._insert_lineage_edges(
                    db, relation_id=relation_id, relation="merge", source_ids=lineage_sources,
                    result_ids=[result_id], created=now,
                )
                region_ids.append(result_id)
            result_region_ids = list(dict.fromkeys(region_ids))
            db.execute(
                "INSERT INTO annotation_save_batches (idempotency_key,capture_id,request_sha256,result_region_ids_json,created_utc) VALUES (?,?,?,?,?)",
                (idempotency_key, capture_id, digest, json.dumps(result_region_ids), now),
            )
        return result_region_ids, False

    def revise_region_geometry(
        self, region_id: str, *, expected_revision: int, sample_start: int, sample_count: int,
        frequency_low_hz: float | None = None, frequency_high_hz: float | None = None,
    ) -> dict[str, Any]:
        if sample_start < 0 or sample_count <= 0:
            raise ValueError("geometry sample_start must be non-negative and sample_count positive")
        if (frequency_low_hz is None) != (frequency_high_hz is None):
            raise ValueError("both frequency bounds must be provided together")
        if frequency_low_hz is not None and (
            not math.isfinite(frequency_low_hz) or not math.isfinite(frequency_high_hz)
            or frequency_low_hz < 0 or frequency_low_hz >= frequency_high_hz
        ):
            raise ValueError("frequency bounds must be finite, non-negative and increasing")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            region = db.execute(
                """SELECT c.sample_count AS capture_samples, c.sample_rate
                   FROM regions r JOIN captures c ON c.id=r.capture_id WHERE r.id=?""", (region_id,)
            ).fetchone()
            if region is None:
                raise KeyError(f"unknown region: {region_id}")
            if sample_start + sample_count > region["capture_samples"]:
                raise ValueError("region exceeds the parent capture sample range")
            if frequency_high_hz is not None and frequency_high_hz > region["sample_rate"] / 2:
                raise ValueError("frequency bounds exceed the capture Nyquist frequency")
            latest = db.execute("SELECT MAX(revision) FROM region_geometry_revisions WHERE region_id=?", (region_id,)).fetchone()[0]
            if latest is None:
                raise KeyError(f"unknown region: {region_id}")
            if latest != expected_revision:
                raise LabelRevisionConflict("geometry changed since it was opened; reopen the region before saving bounds")
            revision = latest + 1
            created = datetime.now(timezone.utc).isoformat()
            db.execute(
                """INSERT INTO region_geometry_revisions
                   (region_id, revision, sample_start, sample_count, frequency_low_hz, frequency_high_hz, created_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (region_id, revision, sample_start, sample_count, frequency_low_hz, frequency_high_hz, created),
            )
            return {"region_id": region_id, "revision": revision, "sample_start": sample_start,
                    "sample_count": sample_count, "frequency_low_hz": frequency_low_hz,
                    "frequency_high_hz": frequency_high_hz, "created_utc": created}

    def geometry_history(self, region_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            exists = db.execute("SELECT 1 FROM regions WHERE id=?", (region_id,)).fetchone()
            if exists is None:
                raise KeyError(f"unknown region: {region_id}")
            rows = db.execute("SELECT * FROM region_geometry_revisions WHERE region_id=? ORDER BY revision", (region_id,)).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _mode_entries(db: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = db.execute(
            """SELECT e.*,
                      (SELECT decision FROM mode_registry_decisions d WHERE d.mode_id=e.id ORDER BY d.id DESC LIMIT 1) AS decision,
                      (SELECT note FROM mode_registry_decisions d WHERE d.mode_id=e.id ORDER BY d.id DESC LIMIT 1) AS decision_note
               FROM mode_registry_entries e ORDER BY e.created_utc,e.id"""
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["aliases"] = json.loads(item.pop("aliases_json"))
            item["evidence_regions"] = json.loads(item.pop("evidence_json"))
            item["status"] = "pending" if item["decision"] is None else f"locally_{item['decision']}"
            item["registry_version"] = f"local:{item['id']}:1"
            result.append(item)
        return result

    def mode_proposals(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            return self._mode_entries(db)

    def accepted_mode_definitions(self) -> list[dict[str, Any]]:
        return [item for item in self.mode_proposals() if item["decision"] == "accepted"]

    def add_mode_proposal(self, definition: dict[str, Any]) -> dict[str, Any]:
        mode_id = definition.get("id")
        if not isinstance(mode_id, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", mode_id):
            raise ValueError("mode id must be lowercase words separated by hyphens")
        text_fields = ("display_name", "family", "method", "protocol", "variant")
        for key in text_fields:
            value = definition.get(key)
            if not isinstance(value, str) or not value.strip() or len(value) > 120:
                raise ValueError(f"{key} must contain 1 to 120 characters")
        aliases = definition.get("aliases")
        if not isinstance(aliases, list) or not 1 <= len(aliases) <= 10 or any(not isinstance(alias, str) or not alias.strip() or len(alias) > 120 for alias in aliases):
            raise ValueError("provide between one and ten non-empty aliases")
        if len({alias.casefold() for alias in aliases}) != len(aliases):
            raise ValueError("aliases must be unique")
        normalized_names = [re.sub(r"[^a-z0-9]+", "", value.casefold()) for value in (definition["display_name"], *aliases)]
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError("display name and aliases must not duplicate each other")
        for key in ("rationale", "distinguishing_evidence"):
            value = definition.get(key)
            if not isinstance(value, str) or len(value.strip()) < 80 or len(value) > 5000:
                raise ValueError(f"{key} must contain 80 to 5000 characters")
        evidence = definition.get("evidence_regions")
        if not isinstance(evidence, list) or not 2 <= len(evidence) <= 8:
            raise ValueError("attach two to eight supporting regions from independent recordings")

        sources = []
        source_hashes = set()
        for item in evidence:
            if not isinstance(item, dict):
                raise ValueError("each evidence region must be an object")
            capture_id, region_id = item.get("capture_id"), item.get("region_id")
            observations = item.get("observations")
            evidence_type = item.get("evidence_type", "on_air")
            if not isinstance(observations, str) or len(observations.strip()) < 40 or len(observations) > 2000:
                raise ValueError("each evidence region needs at least 40 characters of observations")
            if evidence_type not in {"on_air", "controlled_fixture", "independent_decoder"}:
                raise ValueError("unsupported evidence type")
            capture = self.capture(capture_id) if isinstance(capture_id, str) else None
            if capture is None:
                raise ValueError("evidence region references an unknown capture")
            region = next((entry for entry in self.regions(capture_id) if entry["id"] == region_id), None)
            if region is None:
                raise ValueError("evidence region does not belong to its capture")
            source_hash = capture["sha256"]
            source_hashes.add(source_hash)
            sources.append({
                "capture_id": capture_id, "capture_sha256": source_hash,
                "source_kind": capture["source_kind"], "sample_rate": capture["sample_rate"],
                "region_id": region_id, "label_revision": region["revision"],
                "geometry_revision": region["geometry_revision"],
                "sample_start": region["sample_start"], "sample_count": region["sample_count"],
                "frequency_low_hz": region["frequency_low_hz"], "frequency_high_hz": region["frequency_high_hz"],
                "evidence_type": evidence_type, "observations": observations.strip(),
            })
        if len(source_hashes) < 2:
            raise ValueError("supporting evidence must come from two different audio recordings")

        mode = {key: definition[key].strip() for key in text_fields}
        mode.update(id=mode_id, aliases=[alias.strip() for alias in aliases], registry_version=f"local:{mode_id}:1", status="pending")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = self._mode_entries(db)
            if mode_name_conflicts(mode_id, additional_modes=existing):
                raise ValueError("mode id conflicts with an existing registry name or alias")
            for value in (mode["display_name"], *mode["aliases"]):
                if mode_name_conflicts(value, additional_modes=existing):
                    raise ValueError(f"mode name or alias conflicts with an existing registry entry: {value}")
            db.execute(
                """INSERT INTO mode_registry_entries
                   (id,display_name,family,method,protocol,variant,aliases_json,rationale,distinguishing_evidence,evidence_json,created_utc)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (mode_id, mode["display_name"], mode["family"], mode["method"], mode["protocol"], mode["variant"],
                 json.dumps(mode["aliases"], sort_keys=True), definition["rationale"].strip(),
                 definition["distinguishing_evidence"].strip(), json.dumps(sources, sort_keys=True), datetime.now(timezone.utc).isoformat()),
            )
        return next(item for item in self.mode_proposals() if item["id"] == mode_id)

    def decide_mode_proposal(self, mode_id: str, *, decision: str, note: str) -> dict[str, Any]:
        if decision not in {"accepted", "rejected"}:
            raise ValueError("decision must be accepted or rejected")
        if len(note.strip()) < 40 or len(note) > 2000:
            raise ValueError("review note must contain 40 to 2000 characters")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            entries = {item["id"]: item for item in self._mode_entries(db)}
            if mode_id not in entries:
                raise KeyError(f"unknown mode proposal: {mode_id}")
            if decision == "accepted":
                candidate = entries[mode_id]
                others = [item for key, item in entries.items() if key != mode_id and item["decision"] == "accepted"]
                for value in (candidate["id"], candidate["display_name"], *candidate["aliases"]):
                    if mode_name_conflicts(value, additional_modes=others):
                        raise ValueError(f"mode name or alias now conflicts with another accepted definition: {value}")
            db.execute(
                "INSERT INTO mode_registry_decisions(mode_id,decision,note,created_utc) VALUES(?,?,?,?)",
                (mode_id, decision, note.strip(), datetime.now(timezone.utc).isoformat()),
            )
        return next(item for item in self.mode_proposals() if item["id"] == mode_id)

    @staticmethod
    def _validate_lineage_geometry(
        bounds: dict[str, Any], *, capture_samples: int, sample_rate: int,
        parent: dict[str, Any] | None = None,
    ) -> tuple[int, int, float | None, float | None]:
        sample_start = bounds.get("sample_start")
        sample_count = bounds.get("sample_count")
        low = bounds.get("frequency_low_hz")
        high = bounds.get("frequency_high_hz")
        if not isinstance(sample_start, int) or sample_start < 0 or not isinstance(sample_count, int) or sample_count <= 0:
            raise ValueError("geometry sample_start must be non-negative and sample_count positive")
        if sample_start + sample_count > capture_samples:
            raise ValueError("region exceeds the parent capture sample range")
        if (low is None) != (high is None):
            raise ValueError("both frequency bounds must be provided together")
        if low is not None and (
            not isinstance(low, (int, float)) or not isinstance(high, (int, float))
            or not math.isfinite(low) or not math.isfinite(high)
            or low < 0 or low >= high or high > sample_rate / 2
        ):
            raise ValueError("frequency bounds must be finite, non-negative, increasing and below Nyquist")
        if parent is not None:
            if sample_start < parent["sample_start"] or sample_start + sample_count > parent["sample_start"] + parent["sample_count"]:
                raise ValueError("split child time bounds must be contained by the parent region")
            if parent["frequency_low_hz"] is not None and (
                low is None or low < parent["frequency_low_hz"] or high > parent["frequency_high_hz"]
            ):
                raise ValueError("split child frequency bounds must be contained by the parent region")
        return sample_start, sample_count, low, high

    def _insert_lineage_region(
        self, db: sqlite3.Connection, capture_id: str,
        bounds: tuple[int, int, float | None, float | None],
        *, origin: str = "manual", evidence: dict[str, Any] | None = None,
        label: SignalLabel | None = None,
    ) -> str:
        region_id = str(uuid4())
        sample_start, sample_count, low, high = bounds
        created = datetime.now(timezone.utc).isoformat()
        signal_id = self._create_signal_identity(db)
        db.execute(
            """INSERT INTO regions
               (id,capture_id,sample_start,sample_count,frequency_low_hz,frequency_high_hz,origin,signal_id,created_utc)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (region_id, capture_id, sample_start, sample_count, low, high, origin, signal_id, created),
        )
        db.execute(
            "INSERT INTO region_signal_identity_revisions (region_id,revision,signal_id,created_utc) VALUES (?,1,?,?)",
            (region_id, signal_id, created),
        )
        db.execute(
            """INSERT INTO region_geometry_revisions
               (region_id,revision,sample_start,sample_count,frequency_low_hz,frequency_high_hz,created_utc)
               VALUES(?,1,?,?,?,?,?)""",
            (region_id, sample_start, sample_count, low, high, created),
        )
        self._insert_label(db, region_id, 1, label or SignalLabel(
            source="unknown", review_state="pending", evidence=evidence,
        ))
        return region_id

    @staticmethod
    def _insert_lineage_edges(
        db: sqlite3.Connection, *, relation_id: str, relation: str,
        source_ids: list[str], result_ids: list[str], created: str,
    ) -> None:
        db.executemany(
            """INSERT INTO region_lineage
               (relation_id,relation,source_region_id,result_region_id,source_index,result_index,created_utc)
               VALUES(?,?,?,?,?,?,?)""",
            [
                (relation_id, relation, source_id, result_id, source_index, result_index, created)
                for source_index, source_id in enumerate(source_ids)
                for result_index, result_id in enumerate(result_ids)
            ],
        )

    def split_region(
        self, capture_id: str, region_id: str, *, expected_geometry_revision: int,
        children: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not 2 <= len(children) <= 8:
            raise ValueError("a split must create between two and eight child regions")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            parent_row = db.execute(
                """SELECT c.sample_count AS capture_samples,c.sample_rate,
                          g.revision,g.sample_start,g.sample_count,g.frequency_low_hz,g.frequency_high_hz
                   FROM regions r JOIN captures c ON c.id=r.capture_id
                   JOIN region_geometry_revisions g ON g.region_id=r.id AND g.revision=(
                       SELECT MAX(revision) FROM region_geometry_revisions WHERE region_id=r.id
                   ) WHERE r.id=? AND r.capture_id=?""", (region_id, capture_id)
            ).fetchone()
            if parent_row is None:
                raise KeyError(f"unknown region in capture: {region_id}")
            if int(parent_row["revision"]) != expected_geometry_revision:
                raise LabelRevisionConflict("geometry changed since it was opened; reopen the region before splitting")
            if db.execute("SELECT 1 FROM region_lineage WHERE source_region_id=? LIMIT 1", (region_id,)).fetchone():
                raise LabelRevisionConflict("region is already superseded by a lineage operation")
            parent = dict(parent_row)
            normalized = [self._validate_lineage_geometry(
                child, capture_samples=parent["capture_samples"], sample_rate=parent["sample_rate"], parent=parent,
            ) for child in children]
            result_ids = [self._insert_lineage_region(db, capture_id, child) for child in normalized]
            relation_id = str(uuid4())
            created = datetime.now(timezone.utc).isoformat()
            self._insert_lineage_edges(db, relation_id=relation_id, relation="split", source_ids=[region_id], result_ids=result_ids, created=created)
        return {"relation_id": relation_id, "relation": "split", "source_region_ids": [region_id], "result_region_ids": result_ids, "created_utc": created}

    def merge_regions(
        self, capture_id: str, *, source_region_ids: list[str],
        expected_geometry_revisions: dict[str, int],
        cv_proposals: list[dict[str, Any]] | None = None,
        result_bounds: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cv_proposals = cv_proposals or []
        if not 1 <= len(source_region_ids) + len(cv_proposals) <= 32:
            raise ValueError("a merge needs one to 32 saved boxes or CV proposals")
        if not result_bounds and len(source_region_ids) + len(cv_proposals) < 2:
            raise ValueError("a single source needs drawn result bounds")
        if len(set(source_region_ids)) != len(source_region_ids):
            raise ValueError("merge source regions must be distinct")
        if set(expected_geometry_revisions) != set(source_region_ids):
            raise ValueError("expected geometry revisions must identify every source region exactly once")
        proposal_ids = [proposal.get("proposal_id") for proposal in cv_proposals]
        if len(set(proposal_ids)) != len(proposal_ids) or any(not proposal_id for proposal_id in proposal_ids):
            raise ValueError("CV proposal ids must be present and unique")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            capture = db.execute(
                "SELECT sample_count,sample_rate FROM captures WHERE id=?", (capture_id,),
            ).fetchone()
            if capture is None:
                raise KeyError(f"unknown capture: {capture_id}")
            source_rows = []
            for region_id in source_region_ids:
                row = db.execute(
                    """SELECT r.capture_id,c.sample_count AS capture_samples,c.sample_rate,
                              g.revision,g.sample_start,g.sample_count,g.frequency_low_hz,g.frequency_high_hz
                       FROM regions r JOIN captures c ON c.id=r.capture_id
                       JOIN region_geometry_revisions g ON g.region_id=r.id AND g.revision=(
                           SELECT MAX(revision) FROM region_geometry_revisions WHERE region_id=r.id
                       ) WHERE r.id=?""", (region_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown region: {region_id}")
                if row["capture_id"] != capture_id:
                    raise ValueError("merged regions must belong to the requested capture")
                if int(row["revision"]) != expected_geometry_revisions[region_id]:
                    raise LabelRevisionConflict("geometry changed since merge selection; reopen the regions")
                if db.execute("SELECT 1 FROM region_lineage WHERE source_region_id=? LIMIT 1", (region_id,)).fetchone():
                    raise LabelRevisionConflict("a selected region is already superseded by a lineage operation")
                source_rows.append(dict(row))
            all_source_ids = list(source_region_ids)
            for proposal in cv_proposals:
                evidence = proposal["detector_evidence"]
                proposal_id = proposal["proposal_id"]
                model_sha256 = evidence.get("model_sha256")
                sample_start = int(proposal["sample_start"])
                sample_count = int(proposal["sample_count"])
                low = float(proposal["frequency_low_hz"])
                high = float(proposal["frequency_high_hz"])
                confidence = float(proposal["confidence"])
                if not re.fullmatch(r"[0-9a-f]{64}", str(model_sha256)):
                    raise ValueError("CV proposal model hash must be a lowercase SHA-256 digest")
                if not 0 <= sample_start < int(capture["sample_count"]) or sample_start + sample_count > int(capture["sample_count"]):
                    raise ValueError("CV proposal exceeds the capture sample range")
                if not 0 <= low < high <= 3000 or not 0 <= confidence <= 1:
                    raise ValueError("CV proposal bounds or confidence are invalid")
                if evidence.get("model_sha256") != model_sha256 or evidence.get("sample_rate") != int(capture["sample_rate"]):
                    raise ValueError("CV proposal evidence does not match its model or capture")
                source_windows = [
                    {"sample_start": evidence.get("window_sample_start"), "sample_count": evidence.get("window_sample_count")},
                    *evidence.get("supporting_windows", []),
                ]
                if not any(
                    isinstance(window.get("sample_start"), int)
                    and isinstance(window.get("sample_count"), int)
                    and window["sample_start"] <= sample_start
                    and window["sample_start"] + window["sample_count"] >= sample_start + sample_count
                    for window in source_windows
                ):
                    raise ValueError("CV proposal bounds are outside their supporting detector windows")
                if len(json.dumps(evidence)) > 1_000_000:
                    raise ValueError("CV proposal evidence is too large")
                decision = db.execute(
                    """SELECT decision FROM cv_proposal_decisions
                       WHERE capture_id=? AND proposal_id=? AND model_sha256=? ORDER BY id DESC LIMIT 1""",
                    (capture_id, proposal_id, model_sha256),
                ).fetchone()
                if decision is not None and decision["decision"] == "dismissed":
                    raise LabelRevisionConflict("a selected CV proposal was already dismissed; refresh the proposals")
                created = datetime.now(timezone.utc).isoformat()
                region_id = self._insert_lineage_region(
                    db, capture_id, (sample_start, sample_count, low, high),
                    origin="detector_proposal", label=SignalLabel(
                    source="unknown", review_state="pending", evidence={"detector_proposal": evidence},
                    notes="CV proposal included as a merge source; this is not a negative training label.",
                    ),
                )
                db.execute(
                    """INSERT INTO cv_proposal_decisions
                       (capture_id,proposal_id,model_sha256,decision,sample_start,sample_count,
                        frequency_low_hz,frequency_high_hz,confidence,detector_evidence_json,note,created_utc)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (capture_id, proposal_id, model_sha256, "dismissed", sample_start, sample_count,
                     low, high, confidence, json.dumps(evidence, sort_keys=True),
                     "Merged into a user-drawn signal region; review feedback only, not a negative label.", created),
                )
                source_rows.append({
                    "capture_samples": int(capture["sample_count"]), "sample_rate": int(capture["sample_rate"]),
                    "sample_start": sample_start, "sample_count": sample_count,
                    "frequency_low_hz": low, "frequency_high_hz": high, "id": region_id,
                })
                all_source_ids.append(region_id)
            sample_start = min(row["sample_start"] for row in source_rows)
            sample_end = max(row["sample_start"] + row["sample_count"] for row in source_rows)
            bounded_sources = [row for row in source_rows if row["frequency_low_hz"] is not None]
            low = min(row["frequency_low_hz"] for row in bounded_sources) if bounded_sources else None
            high = max(row["frequency_high_hz"] for row in bounded_sources) if bounded_sources else None
            if result_bounds is not None:
                drawn_start, drawn_count, drawn_low, drawn_high = self._validate_lineage_geometry(
                    result_bounds,
                    capture_samples=int(capture["sample_count"]),
                    sample_rate=int(capture["sample_rate"]),
                )
                # The operator's selection is the authoritative result box.
                sample_start, sample_end = drawn_start, drawn_start + drawn_count
                low, high = drawn_low, drawn_high
            result_id = self._insert_lineage_region(db, capture_id, (sample_start, sample_end - sample_start, low, high))
            relation_id = str(uuid4())
            created = datetime.now(timezone.utc).isoformat()
            self._insert_lineage_edges(db, relation_id=relation_id, relation="merge", source_ids=all_source_ids, result_ids=[result_id], created=created)
        return {"relation_id": relation_id, "relation": "merge", "source_region_ids": all_source_ids, "result_region_id": result_id, "created_utc": created}

    def region_lineage(self, region_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            if db.execute("SELECT 1 FROM regions WHERE id=?", (region_id,)).fetchone() is None:
                raise KeyError(f"unknown region: {region_id}")
            rows = db.execute(
                """SELECT rl.*,
                          EXISTS(SELECT 1 FROM region_removals rr WHERE rr.region_id=rl.source_region_id) AS source_removed,
                          EXISTS(SELECT 1 FROM region_removals rr WHERE rr.region_id=rl.result_region_id) AS result_removed
                   FROM region_lineage rl WHERE source_region_id=? OR result_region_id=?
                   ORDER BY created_utc,relation_id,source_index,result_index""", (region_id, region_id)
            ).fetchall()
        operations: dict[str, dict[str, Any]] = {}
        for row in rows:
            operation = operations.setdefault(row["relation_id"], {
                "relation_id": row["relation_id"], "relation": row["relation"], "created_utc": row["created_utc"],
                "source_region_ids": {}, "result_region_ids": {},
                "removed_source_region_ids": set(), "removed_result_region_ids": set(),
            })
            operation["source_region_ids"][row["source_index"]] = row["source_region_id"]
            operation["result_region_ids"][row["result_index"]] = row["result_region_id"]
            if row["source_removed"]:
                operation["removed_source_region_ids"].add(row["source_region_id"])
            if row["result_removed"]:
                operation["removed_result_region_ids"].add(row["result_region_id"])
        return [{
            **operation,
            "source_region_ids": [operation["source_region_ids"][index] for index in sorted(operation["source_region_ids"])],
            "result_region_ids": [operation["result_region_ids"][index] for index in sorted(operation["result_region_ids"])],
            "removed_source_region_ids": sorted(operation["removed_source_region_ids"]),
            "removed_result_region_ids": sorted(operation["removed_result_region_ids"]),
        } for operation in operations.values()]

    @staticmethod
    def _coverage_masks(
        masks: list[dict[str, Any]], *, coverage_bounds: dict[str, Any],
        capture_samples: int, sample_rate: int, field_name: str,
    ) -> list[dict[str, Any]]:
        if len(masks) > 64:
            raise ValueError(f"{field_name} cannot contain more than 64 regions")
        normalized = []
        for item in masks:
            bounds = {key: item.get(key) for key in (
                "sample_start", "sample_count", "frequency_low_hz", "frequency_high_hz",
            )}
            if bounds["frequency_low_hz"] is None and bounds["frequency_high_hz"] is None:
                bounds["frequency_low_hz"] = coverage_bounds["frequency_low_hz"]
                bounds["frequency_high_hz"] = coverage_bounds["frequency_high_hz"]
            checked = CaptureStore._validate_lineage_geometry(
                bounds, capture_samples=capture_samples, sample_rate=sample_rate, parent=coverage_bounds,
            )
            reason = item.get("reason", "")
            if not isinstance(reason, str) or len(reason) > 200:
                raise ValueError(f"{field_name} reasons must be strings of at most 200 characters")
            normalized.append({
                "sample_start": checked[0], "sample_count": checked[1],
                "frequency_low_hz": checked[2], "frequency_high_hz": checked[3], "reason": reason,
            })
        return normalized

    @staticmethod
    def _validate_coverage_fields(
        *, taxonomy_scope: list[str], task: str, state: str, notes: str,
    ) -> None:
        if task not in {"detection", "modulation", "protocol", "decoder"}:
            raise ValueError("unsupported scene coverage task")
        if state not in {"complete", "partial"}:
            raise ValueError("scene coverage state must be complete or partial")
        if len(taxonomy_scope) > 32 or any(not isinstance(item, str) or not item.strip() or len(item) > 80 for item in taxonomy_scope):
            raise ValueError("taxonomy_scope must contain at most 32 non-empty strings of at most 80 characters")
        if len(notes) > 2000:
            raise ValueError("scene coverage notes are too long")

    @staticmethod
    def _annotation_set_digest(
        db: sqlite3.Connection, capture_id: str, bounds: dict[str, Any],
    ) -> str:
        rows = db.execute(
            """SELECT r.id,l.revision AS label_revision,
                      g.revision AS geometry_revision,g.sample_start,g.sample_count,
                      g.frequency_low_hz,g.frequency_high_hz
               FROM regions r
               JOIN label_revisions l ON l.region_id=r.id AND l.revision=(
                   SELECT MAX(revision) FROM label_revisions WHERE region_id=r.id
               )
               JOIN region_geometry_revisions g ON g.region_id=r.id AND g.revision=(
                   SELECT MAX(revision) FROM region_geometry_revisions WHERE region_id=r.id
               )
               WHERE r.capture_id=? AND NOT EXISTS(
                   SELECT 1 FROM region_lineage rl WHERE rl.source_region_id=r.id
               ) AND NOT EXISTS(
                   SELECT 1 FROM region_removals rr WHERE rr.region_id=r.id
               ) ORDER BY r.id""", (capture_id,)
        ).fetchall()
        covered = []
        coverage_start = bounds["sample_start"]
        coverage_end = coverage_start + bounds["sample_count"]
        for row in rows:
            if row["sample_start"] >= coverage_end or row["sample_start"] + row["sample_count"] <= coverage_start:
                continue
            if (
                bounds["frequency_low_hz"] is not None
                and row["frequency_low_hz"] is not None
                and (
                    row["frequency_low_hz"] >= bounds["frequency_high_hz"]
                    or row["frequency_high_hz"] <= bounds["frequency_low_hz"]
                )
            ):
                continue
            covered.append({
                "region_id": row["id"], "label_revision": row["label_revision"],
                "geometry_revision": row["geometry_revision"],
            })
        canonical = json.dumps(covered, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def _coverage_record(row: sqlite3.Row, current_digest: str) -> dict[str, Any]:
        item = dict(row)
        item["id"] = item["coverage_id"]
        item["taxonomy_scope"] = json.loads(item.pop("taxonomy_scope_json"))
        item["ignore_regions"] = json.loads(item.pop("ignore_regions_json"))
        item["uncertain_regions"] = json.loads(item.pop("uncertain_regions_json"))
        item["current_annotation_set_digest"] = current_digest
        item["is_stale"] = item["annotation_set_digest"] != current_digest
        return item

    def _insert_scene_coverage(
        self, db: sqlite3.Connection, *, coverage_id: str, capture_id: str,
        revision: int, bounds: tuple[int, int, float | None, float | None],
        task: str, taxonomy_scope: list[str], state: str, annotation_set_digest: str,
        ignore_regions: list[dict[str, Any]], uncertain_regions: list[dict[str, Any]], notes: str,
    ) -> sqlite3.Row:
        created = datetime.now(timezone.utc).isoformat()
        db.execute(
            """INSERT INTO scene_coverage_revisions
               (coverage_id,capture_id,revision,sample_start,sample_count,frequency_low_hz,frequency_high_hz,
                task,taxonomy_scope_json,state,annotation_set_digest,ignore_regions_json,uncertain_regions_json,notes,created_utc)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (coverage_id,capture_id,revision,*bounds,task,json.dumps(taxonomy_scope,sort_keys=True),state,
             annotation_set_digest,json.dumps(ignore_regions,sort_keys=True),json.dumps(uncertain_regions,sort_keys=True),notes,created),
        )
        return db.execute(
            "SELECT * FROM scene_coverage_revisions WHERE coverage_id=? AND revision=?", (coverage_id,revision)
        ).fetchone()

    def create_scene_coverage(
        self, capture_id: str, *, sample_start: int, sample_count: int,
        frequency_low_hz: float | None = None, frequency_high_hz: float | None = None,
        task: str, taxonomy_scope: list[str], state: str,
        ignore_regions: list[dict[str, Any]] | None = None,
        uncertain_regions: list[dict[str, Any]] | None = None, notes: str = "",
    ) -> dict[str, Any]:
        ignore_regions = ignore_regions or []
        uncertain_regions = uncertain_regions or []
        self._validate_coverage_fields(taxonomy_scope=taxonomy_scope, task=task, state=state, notes=notes)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            capture = db.execute("SELECT sample_count,sample_rate FROM captures WHERE id=?", (capture_id,)).fetchone()
            if capture is None:
                raise KeyError(f"unknown capture: {capture_id}")
            bounds_tuple = self._validate_lineage_geometry(
                {"sample_start": sample_start, "sample_count": sample_count,
                 "frequency_low_hz": frequency_low_hz, "frequency_high_hz": frequency_high_hz},
                capture_samples=capture["sample_count"], sample_rate=capture["sample_rate"],
            )
            bounds = dict(zip(("sample_start","sample_count","frequency_low_hz","frequency_high_hz"), bounds_tuple))
            normalized_ignore = self._coverage_masks(ignore_regions, coverage_bounds=bounds,
                capture_samples=capture["sample_count"], sample_rate=capture["sample_rate"], field_name="ignore_regions")
            normalized_uncertain = self._coverage_masks(uncertain_regions, coverage_bounds=bounds,
                capture_samples=capture["sample_count"], sample_rate=capture["sample_rate"], field_name="uncertain_regions")
            digest = self._annotation_set_digest(db, capture_id, bounds)
            coverage_id = str(uuid4())
            row = self._insert_scene_coverage(db, coverage_id=coverage_id, capture_id=capture_id, revision=1,
                bounds=bounds_tuple, task=task, taxonomy_scope=taxonomy_scope, state=state,
                annotation_set_digest=digest, ignore_regions=normalized_ignore,
                uncertain_regions=normalized_uncertain, notes=notes)
            return self._coverage_record(row, digest)

    def revise_scene_coverage(
        self, capture_id: str, coverage_id: str, *, expected_revision: int,
        sample_start: int, sample_count: int,
        frequency_low_hz: float | None = None, frequency_high_hz: float | None = None,
        task: str, taxonomy_scope: list[str], state: str,
        ignore_regions: list[dict[str, Any]] | None = None,
        uncertain_regions: list[dict[str, Any]] | None = None, notes: str = "",
    ) -> dict[str, Any]:
        ignore_regions = ignore_regions or []
        uncertain_regions = uncertain_regions or []
        self._validate_coverage_fields(taxonomy_scope=taxonomy_scope, task=task, state=state, notes=notes)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            latest_row = db.execute(
                "SELECT MAX(revision) AS revision FROM scene_coverage_revisions WHERE capture_id=? AND coverage_id=?",
                (capture_id, coverage_id),
            ).fetchone()
            if latest_row is None or latest_row["revision"] is None:
                raise KeyError(f"unknown scene coverage: {coverage_id}")
            latest = int(latest_row["revision"])
            if latest != expected_revision:
                raise LabelRevisionConflict("scene coverage changed since it was opened; reopen it before editing")
            capture = db.execute("SELECT sample_count,sample_rate FROM captures WHERE id=?", (capture_id,)).fetchone()
            if capture is None:
                raise KeyError(f"unknown capture: {capture_id}")
            bounds_tuple = self._validate_lineage_geometry(
                {"sample_start": sample_start, "sample_count": sample_count,
                 "frequency_low_hz": frequency_low_hz, "frequency_high_hz": frequency_high_hz},
                capture_samples=capture["sample_count"], sample_rate=capture["sample_rate"],
            )
            bounds = dict(zip(("sample_start","sample_count","frequency_low_hz","frequency_high_hz"), bounds_tuple))
            normalized_ignore = self._coverage_masks(ignore_regions, coverage_bounds=bounds,
                capture_samples=capture["sample_count"], sample_rate=capture["sample_rate"], field_name="ignore_regions")
            normalized_uncertain = self._coverage_masks(uncertain_regions, coverage_bounds=bounds,
                capture_samples=capture["sample_count"], sample_rate=capture["sample_rate"], field_name="uncertain_regions")
            digest = self._annotation_set_digest(db, capture_id, bounds)
            row = self._insert_scene_coverage(db, coverage_id=coverage_id, capture_id=capture_id, revision=latest+1,
                bounds=bounds_tuple, task=task, taxonomy_scope=taxonomy_scope, state=state,
                annotation_set_digest=digest, ignore_regions=normalized_ignore,
                uncertain_regions=normalized_uncertain, notes=notes)
            return self._coverage_record(row, digest)

    def scene_coverages(self, capture_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            if db.execute("SELECT 1 FROM captures WHERE id=?", (capture_id,)).fetchone() is None:
                raise KeyError(f"unknown capture: {capture_id}")
            rows = db.execute(
                """SELECT * FROM scene_coverage_revisions c
                   WHERE c.capture_id=? AND c.revision=(
                       SELECT MAX(revision) FROM scene_coverage_revisions WHERE coverage_id=c.coverage_id
                   ) ORDER BY c.created_utc,c.coverage_id""", (capture_id,)
            ).fetchall()
            result = []
            for row in rows:
                bounds = {key: row[key] for key in ("sample_start","sample_count","frequency_low_hz","frequency_high_hz")}
                current_digest = self._annotation_set_digest(db, capture_id, bounds)
                result.append(self._coverage_record(row, current_digest))
        return result

    def scene_coverage_history(self, capture_id: str, coverage_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM scene_coverage_revisions WHERE capture_id=? AND coverage_id=? ORDER BY revision",
                (capture_id, coverage_id),
            ).fetchall()
            if not rows:
                if db.execute("SELECT 1 FROM captures WHERE id=?", (capture_id,)).fetchone() is None:
                    raise KeyError(f"unknown capture: {capture_id}")
                raise KeyError(f"unknown scene coverage: {coverage_id}")
            result = []
            for row in rows:
                bounds = {key: row[key] for key in ("sample_start","sample_count","frequency_low_hz","frequency_high_hz")}
                current_digest = self._annotation_set_digest(db, capture_id, bounds)
                result.append(self._coverage_record(row, current_digest))
        return result

    @staticmethod
    def _validate_label(label: SignalLabel, additional_modes: list[dict[str, Any]] | None = None) -> None:
        if label.source not in LABEL_SOURCES:
            raise ValueError(f"unknown label source: {label.source}")
        if label.review_state not in REVIEW_STATES:
            raise ValueError(f"unknown review state: {label.review_state}")
        if label.mode_id is not None:
            resolved = resolve_mode_label(label.mode_id, {
                "family": label.family, "method": label.method,
                "protocol": label.protocol, "variant": label.variant,
            }, additional_modes=additional_modes)
            if any(getattr(label, key) != resolved[key] for key in ("family", "method", "protocol", "variant")):
                raise ValueError("canonical mode labels must include the resolved taxonomy fields")

    @staticmethod
    def _insert_label(db: sqlite3.Connection, region_id: str, revision: int, label: SignalLabel) -> None:
        db.execute(
            """INSERT INTO label_revisions
               (region_id, revision, created_utc, family, method, protocol, variant, mode_id,
                source, review_state, evidence_json, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                region_id,
                revision,
                datetime.now(timezone.utc).isoformat(),
                label.family,
                label.method,
                label.protocol,
                label.variant,
                label.mode_id,
                label.source,
                label.review_state,
                json.dumps(label.evidence or {}, sort_keys=True),
                label.notes,
            ),
        )

    def revise_label(self, region_id: str, label: SignalLabel, *, expected_revision: int | None = None) -> int:
        self._validate_label(label, self.accepted_mode_definitions())
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            latest = db.execute(
                "SELECT MAX(revision) AS revision FROM label_revisions WHERE region_id = ?",
                (region_id,),
            ).fetchone()["revision"]
            if latest is None:
                raise KeyError(f"unknown region: {region_id}")
            if expected_revision is not None and latest != expected_revision:
                raise LabelRevisionConflict("label changed since it was opened; reopen the region before saving")
            revision = latest + 1
            self._insert_label(db, region_id, revision, label)
            if label.source == "human" or label.review_state == "accepted":
                identity = db.execute("SELECT signal_id FROM regions WHERE id=?", (region_id,)).fetchone()
                if identity and identity["signal_id"]:
                    peers = db.execute(
                        """SELECT r.id,l.* FROM regions r JOIN label_revisions l ON l.region_id=r.id
                           AND l.revision=(SELECT MAX(revision) FROM label_revisions WHERE region_id=r.id)
                           WHERE r.signal_id=? AND r.id<>?
                           AND NOT EXISTS(SELECT 1 FROM region_removals rr WHERE rr.region_id=r.id)
                           AND NOT EXISTS(SELECT 1 FROM region_lineage rl WHERE rl.source_region_id=r.id)""",
                        (identity["signal_id"], region_id),
                    ).fetchall()
                    for peer in peers:
                        if (peer["family"], peer["method"], peer["protocol"], peer["variant"], peer["mode_id"]) == (
                            label.family, label.method, label.protocol, label.variant, label.mode_id
                        ):
                            continue
                        self._insert_label(db, peer["id"], int(peer["revision"]) + 1, SignalLabel(
                            family=label.family, method=label.method, protocol=label.protocol,
                            variant=label.variant, mode_id=label.mode_id, source=peer["source"],
                            review_state=peer["review_state"], evidence=json.loads(peer["evidence_json"]),
                            notes=peer["notes"],
                        ))
            return revision

    def record_feedback(
        self, region_id: str, *, expected_revision: int, action: str,
        expected_geometry_revision: int, correction: dict[str, Any] | None = None, note: str = "",
    ) -> dict[str, Any]:
        if action not in {"correct", "incorrect_identity", "not_signal"}:
            raise ValueError("unsupported feedback action")
        if len(note) > 2000:
            raise ValueError("feedback note is too long")
        correction = correction or {}
        if action == "incorrect_identity" and not any(correction.get(key) for key in ("family", "method", "protocol", "variant")):
            raise ValueError("incorrect_identity feedback needs a corrected identity")
        if action != "incorrect_identity" and correction:
            raise ValueError("correction is only valid for incorrect_identity feedback")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT * FROM label_revisions WHERE region_id=? ORDER BY revision DESC LIMIT 1", (region_id,)).fetchone()
            if current is None:
                raise KeyError(f"unknown region: {region_id}")
            latest = int(current["revision"])
            if latest != expected_revision:
                raise LabelRevisionConflict("label changed since feedback was opened; reopen the region")
            geometry_revision = db.execute("SELECT MAX(revision) FROM region_geometry_revisions WHERE region_id=?", (region_id,)).fetchone()[0]
            if geometry_revision != expected_geometry_revision:
                raise LabelRevisionConflict("bounds changed since feedback was opened; reopen the region")
            if action == "incorrect_identity":
                prior_evidence = json.loads(current["evidence_json"])
                next_label = SignalLabel(
                    family=correction.get("family"), method=correction.get("method"),
                    protocol=correction.get("protocol"), variant=correction.get("variant"),
                    mode_id=correction.get("mode_id"), source="human", review_state="accepted",
                    evidence={**prior_evidence, "feedback_action": action, "corrected_from_revision": latest, **({"mode_registry_version": correction["registry_version"]} if correction.get("registry_version") else {})},
                    notes=note,
                )
            else:
                next_label = SignalLabel(
                    family=current["family"], method=current["method"], protocol=current["protocol"],
                    variant=current["variant"], mode_id=current["mode_id"], source="human",
                    review_state="accepted" if action == "correct" else "rejected",
                    evidence={**json.loads(current["evidence_json"]), "feedback_action": action, "reviewed_from_revision": latest},
                    notes=note or current["notes"],
                )
            accepted_modes = [item for item in self._mode_entries(db) if item["decision"] == "accepted"]
            self._validate_label(next_label, accepted_modes)
            resulting_revision = latest + 1
            self._insert_label(db, region_id, resulting_revision, next_label)
            cursor = db.execute(
                "INSERT INTO annotation_feedback(region_id,label_revision,geometry_revision,resulting_revision,action,correction_json,note,created_utc) VALUES(?,?,?,?,?,?,?,?)",
                (region_id, expected_revision, expected_geometry_revision, resulting_revision, action, json.dumps(correction, sort_keys=True), note, datetime.now(timezone.utc).isoformat()),
            )
            row = db.execute("SELECT * FROM annotation_feedback WHERE id=?", (cursor.lastrowid,)).fetchone()
        item = dict(row)
        item["correction"] = json.loads(item.pop("correction_json"))
        return item

    def feedback_history(self, region_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            exists = db.execute("SELECT 1 FROM regions WHERE id=?", (region_id,)).fetchone()
            if exists is None:
                raise KeyError(f"unknown region: {region_id}")
            rows = db.execute("SELECT * FROM annotation_feedback WHERE region_id=? ORDER BY id", (region_id,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["correction"] = json.loads(item.pop("correction_json"))
            result.append(item)
        return result

    @staticmethod
    def _cv_proposal_decision_record(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["detector_evidence"] = json.loads(item.pop("detector_evidence_json"))
        return item

    def record_cv_proposal_decision(
        self, capture_id: str, *, proposal_id: str, model_sha256: str,
        decision: str, sample_start: int, sample_count: int,
        frequency_low_hz: float, frequency_high_hz: float, confidence: float,
        detector_evidence: dict[str, Any], note: str = "",
    ) -> dict[str, Any]:
        if decision not in {"dismissed", "restored"}:
            raise ValueError("unsupported CV proposal decision")
        if not proposal_id or len(proposal_id) > 256:
            raise ValueError("CV proposal id must be between 1 and 256 characters")
        if not re.fullmatch(r"[0-9a-f]{64}", model_sha256):
            raise ValueError("CV proposal model hash must be a lowercase SHA-256 digest")
        if sample_start < 0 or sample_count <= 0:
            raise ValueError("CV proposal sample bounds must be non-negative and non-empty")
        if not all(math.isfinite(value) for value in (frequency_low_hz, frequency_high_hz, confidence)):
            raise ValueError("CV proposal geometry and confidence must be finite")
        if not 0 <= frequency_low_hz < frequency_high_hz <= 3000 or not 0 <= confidence <= 1:
            raise ValueError("CV proposal geometry must be ordered within 0-3000 Hz and confidence within 0-1")
        if len(note) > 2000:
            raise ValueError("CV proposal decision note is too long")
        if detector_evidence.get("model_sha256") != model_sha256:
            raise ValueError("CV proposal evidence hash does not match its decision model")
        if len(json.dumps(detector_evidence)) > 1_000_000:
            raise ValueError("CV proposal evidence is too large")

        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            capture = db.execute(
                "SELECT sample_count,sample_rate FROM captures WHERE id=?", (capture_id,),
            ).fetchone()
            if capture is None:
                raise KeyError(f"unknown capture: {capture_id}")
            if sample_start + sample_count > int(capture["sample_count"]):
                raise ValueError("CV proposal exceeds the capture sample range")
            if detector_evidence.get("sample_rate") != int(capture["sample_rate"]):
                raise ValueError("CV proposal sample rate does not match its capture")
            source_windows = [
                {
                    "sample_start": detector_evidence.get("window_sample_start"),
                    "sample_count": detector_evidence.get("window_sample_count"),
                },
                *detector_evidence.get("supporting_windows", []),
            ]
            if not any(
                isinstance(window.get("sample_start"), int)
                and isinstance(window.get("sample_count"), int)
                and window["sample_start"] <= sample_start
                and window["sample_start"] + window["sample_count"] >= sample_start + sample_count
                for window in source_windows
            ):
                raise ValueError("CV proposal bounds are outside their supporting detector windows")
            latest = db.execute(
                """SELECT * FROM cv_proposal_decisions
                   WHERE capture_id=? AND proposal_id=? AND model_sha256=?
                   ORDER BY id DESC LIMIT 1""",
                (capture_id, proposal_id, model_sha256),
            ).fetchone()
            if decision == "restored" and (latest is None or latest["decision"] != "dismissed"):
                raise ValueError("only a currently dismissed CV proposal can be restored")
            if decision == "dismissed" and latest is not None and latest["decision"] == "dismissed":
                return self._cv_proposal_decision_record(latest)
            cursor = db.execute(
                """INSERT INTO cv_proposal_decisions
                   (capture_id,proposal_id,model_sha256,decision,sample_start,sample_count,
                    frequency_low_hz,frequency_high_hz,confidence,detector_evidence_json,note,created_utc)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (capture_id, proposal_id, model_sha256, decision, sample_start, sample_count,
                 frequency_low_hz, frequency_high_hz, confidence,
                 json.dumps(detector_evidence, sort_keys=True), note.strip(), datetime.now(timezone.utc).isoformat()),
            )
            row = db.execute("SELECT * FROM cv_proposal_decisions WHERE id=?", (cursor.lastrowid,)).fetchone()
        return self._cv_proposal_decision_record(row)

    def cv_proposal_decisions(self, capture_id: str, *, model_sha256: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            if db.execute("SELECT 1 FROM captures WHERE id=?", (capture_id,)).fetchone() is None:
                raise KeyError(f"unknown capture: {capture_id}")
            rows = db.execute(
                """SELECT * FROM cv_proposal_decisions
                   WHERE capture_id=? AND model_sha256=? ORDER BY id""",
                (capture_id, model_sha256),
            ).fetchall()
        latest: dict[str, sqlite3.Row] = {}
        for row in rows:
            latest[row["proposal_id"]] = row
        dismissed = [row for row in latest.values() if row["decision"] == "dismissed"]
        dismissed.sort(key=lambda row: int(row["id"]), reverse=True)
        return [self._cv_proposal_decision_record(row) for row in dismissed]

    def export_cv_proposal_decisions_jsonl(self, output: str | Path) -> int:
        """Export the full CV proposal review-event history without treating it as labels."""
        with self._connect() as db:
            rows = db.execute(
                """SELECT d.*, c.sha256 AS capture_sha256, c.source_kind, c.sample_rate
                   FROM cv_proposal_decisions d JOIN captures c ON c.id=d.capture_id
                   ORDER BY d.created_utc,d.id"""
            ).fetchall()
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            for row in rows:
                record = dict(row)
                record["detector_evidence"] = json.loads(record.pop("detector_evidence_json"))
                record["review_feedback_only"] = True
                record["training_label"] = False
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return len(rows)

    def add_raw_demod_attempt(
        self, capture_id: str, *, mode: str, sample_start: int, sample_count: int,
        settings: dict[str, Any], result: dict[str, Any],
    ) -> dict[str, Any]:
        if mode not in {"2fsk", "ook", "mfsk", "bpsk", "qpsk"}:
            raise ValueError("unsupported raw demodulator")
        if sample_start < 0 or sample_count <= 0:
            raise ValueError("sample range must be positive")
        attempt_id = str(uuid4())
        created = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            capture = db.execute("SELECT sample_count FROM captures WHERE id=?", (capture_id,)).fetchone()
            if capture is None:
                raise KeyError(f"unknown capture: {capture_id}")
            if sample_start + sample_count > capture["sample_count"]:
                raise ValueError("demodulation sample range exceeds the parent capture")
            db.execute(
                """INSERT INTO raw_demod_attempts
                   (id,capture_id,mode,sample_start,sample_count,settings_json,result_json,created_utc)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (attempt_id, capture_id, mode, sample_start, sample_count,
                 json.dumps(settings, sort_keys=True, separators=(",", ":")),
                 json.dumps(result, sort_keys=True, separators=(",", ":")), created),
            )
        return {"id": attempt_id, "capture_id": capture_id, "mode": mode,
                "sample_start": sample_start, "sample_count": sample_count,
                "settings": settings, "result": result, "created_utc": created}

    def raw_demod_attempts(self, capture_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            if db.execute("SELECT 1 FROM captures WHERE id=?", (capture_id,)).fetchone() is None:
                raise KeyError(f"unknown capture: {capture_id}")
            rows = db.execute(
                "SELECT * FROM raw_demod_attempts WHERE capture_id=? ORDER BY created_utc DESC,id DESC",
                (capture_id,),
            ).fetchall()
        return [{**dict(row), "settings": json.loads(row["settings_json"]),
                 "result": json.loads(row["result_json"])} for row in rows]

    def add_bit_annotation(
        self, capture_id: str, *, mode: str, sample_start: int, sample_count: int,
        settings: dict[str, Any], hypothesis: str, bits: str, bit_start: int,
        bit_count: int, label: str, color: str,
    ) -> dict[str, Any]:
        if not mode or len(mode) > 40 or not hypothesis or len(hypothesis) > 80:
            raise ValueError("mode and bit hypothesis are required")
        if sample_start < 0 or sample_count <= 0 or bit_start < 0 or bit_count <= 0:
            raise ValueError("sample and bit ranges must be positive")
        if (sample_start != settings.get("sample_start") or not isinstance(settings.get("sample_count"), int)
                or sample_count > settings.get("sample_count")):
            raise ValueError("actual sample range must fit the demodulator settings")
        if not bits or len(bits) > 1_000_000 or any(bit not in "01" for bit in bits):
            raise ValueError("recovered bits must be a non-empty binary string of at most one million bits")
        if bit_start + bit_count > len(bits):
            raise ValueError("annotation range exceeds the recovered bitstream")
        if not label.strip() or len(label) > 200:
            raise ValueError("operator interpretation must contain 1 to 200 characters")
        if len(color) != 7 or color[0] != "#" or any(ch not in "0123456789abcdefABCDEF" for ch in color[1:]):
            raise ValueError("annotation color must be a six-digit hex color")
        canonical_settings = json.dumps(settings, sort_keys=True, separators=(",", ":"))
        digest_input = "\0".join((mode, canonical_settings, hypothesis, bits)).encode("utf-8")
        stream_sha256 = hashlib.sha256(digest_input).hexdigest()
        with self._connect() as db:
            capture = db.execute("SELECT sample_count FROM captures WHERE id = ?", (capture_id,)).fetchone()
            if capture is None:
                raise KeyError(f"unknown capture: {capture_id}")
            if sample_start + sample_count > capture["sample_count"]:
                raise ValueError("demodulation sample range exceeds the parent capture")
            stream = db.execute(
                "SELECT id FROM demod_bitstreams WHERE capture_id = ? AND stream_sha256 = ?",
                (capture_id, stream_sha256),
            ).fetchone()
            if stream is None:
                stream_id = str(uuid4())
                db.execute(
                    """INSERT INTO demod_bitstreams
                       (id,capture_id,stream_sha256,mode,sample_start,sample_count,settings_json,hypothesis,bits,created_utc)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (stream_id, capture_id, stream_sha256, mode, sample_start, sample_count,
                     canonical_settings, hypothesis, bits, datetime.now(timezone.utc).isoformat()),
                )
            else:
                stream_id = stream["id"]
            annotation_id = str(uuid4())
            created = datetime.now(timezone.utc).isoformat()
            db.execute(
                """INSERT INTO bit_annotations (id,bitstream_id,bit_start,bit_count,label,color,created_utc)
                   VALUES (?,?,?,?,?,?,?)""",
                (annotation_id, stream_id, bit_start, bit_count, label.strip(), color.upper(), created),
            )
        return {"id": annotation_id, "bitstream_id": stream_id, "stream_sha256": stream_sha256, "created_utc": created}

    def bit_annotations(self, capture_id: str, *, include_bits: bool = False) -> list[dict[str, Any]]:
        with self._connect() as db:
            if db.execute("SELECT 1 FROM captures WHERE id = ?", (capture_id,)).fetchone() is None:
                raise KeyError(f"unknown capture: {capture_id}")
            rows = db.execute(
                """SELECT a.id,a.bit_start,a.bit_count,a.label,a.color,a.created_utc,
                          s.id AS bitstream_id,s.stream_sha256,s.mode,s.sample_start,s.sample_count,
                          s.settings_json,s.hypothesis,s.bits
                   FROM bit_annotations a JOIN demod_bitstreams s ON s.id = a.bitstream_id
                   WHERE s.capture_id = ? ORDER BY a.created_utc,a.id""",
                (capture_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["settings"] = json.loads(item.pop("settings_json"))
            if not include_bits:
                item.pop("bits")
            result.append(item)
        return result

    def export_bit_annotations_jsonl(self, output: str | Path) -> int:
        with self._connect() as db:
            rows = db.execute(
                """SELECT a.id,a.bit_start,a.bit_count,a.label,a.color,a.created_utc,
                          s.id AS bitstream_id,s.stream_sha256,s.mode,s.sample_start,s.sample_count,
                          s.settings_json,s.hypothesis,s.bits,c.sha256 AS capture_sha256,
                          c.sample_rate,c.source_kind
                   FROM bit_annotations a JOIN demod_bitstreams s ON s.id=a.bitstream_id
                   JOIN captures c ON c.id=s.capture_id ORDER BY a.created_utc,a.id"""
            ).fetchall()
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            for row in rows:
                record = dict(row)
                record["settings"] = json.loads(record.pop("settings_json"))
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return len(rows)

    def add_protocol_stencil(
        self, *, name: str, field: str, pattern: str, protocols: list[str],
        source_capture_id: str | None = None,
    ) -> dict[str, Any]:
        name, field = name.strip(), field.strip()
        if not name or len(name) > 100 or not field or len(field) > 100:
            raise ValueError("stencil name and field must contain 1 to 100 characters")
        if len(pattern) < 4 or len(pattern) > 512 or any(bit not in "01?" for bit in pattern):
            raise ValueError("stencil pattern must contain 4 to 512 bits or wildcard characters")
        if sum(bit != "?" for bit in pattern) < 4:
            raise ValueError("stencil pattern must constrain at least four bits")
        normalized_protocols = sorted({str(item).strip() for item in protocols if str(item).strip()})
        if len(normalized_protocols) > 20 or any(len(item) > 120 for item in normalized_protocols):
            raise ValueError("stencil protocol list is too large")
        stencil_id = str(uuid4())
        created = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            if source_capture_id is not None and db.execute(
                "SELECT 1 FROM captures WHERE id = ?", (source_capture_id,)
            ).fetchone() is None:
                raise KeyError(f"unknown capture: {source_capture_id}")
            db.execute(
                """INSERT INTO protocol_stencils
                   (id,name,field,pattern,protocols_json,source_capture_id,created_utc)
                   VALUES (?,?,?,?,?,?,?)""",
                (stencil_id, name, field, pattern, json.dumps(normalized_protocols), source_capture_id, created),
            )
        return {
            "id": stencil_id, "name": name, "field": field, "pattern": pattern,
            "protocols": normalized_protocols, "source_capture_id": source_capture_id,
            "created_utc": created, "source": "operator",
        }

    def protocol_stencils(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM protocol_stencils ORDER BY name COLLATE NOCASE,id").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["protocols"] = json.loads(item.pop("protocols_json"))
            item["source"] = "operator"
            result.append(item)
        return result

    def capture(self, capture_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM captures WHERE id = ?", (capture_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown capture: {capture_id}")
        result = dict(row)
        result["source_metadata"] = json.loads(result.pop("source_metadata_json"))
        return result

    def set_live_reviewed(self, capture_id: str, reviewed: bool) -> dict[str, Any]:
        """Mark an auto-captured live sample reviewed without changing scene coverage."""
        with self._connect() as db:
            row = db.execute(
                "SELECT source_metadata_json FROM captures WHERE id = ?", (capture_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown capture: {capture_id}")
            metadata = json.loads(row["source_metadata_json"])
            if not metadata.get("auto_review_queue"):
                raise ValueError("capture is not a live review-queue sample")
            if reviewed:
                metadata["auto_review_queue_reviewed_utc"] = datetime.now(timezone.utc).isoformat()
            else:
                metadata.pop("auto_review_queue_reviewed_utc", None)
            db.execute(
                "UPDATE captures SET source_metadata_json = ? WHERE id = ?",
                (json.dumps(metadata, sort_keys=True, separators=(",", ":")), capture_id),
            )
        return self.capture(capture_id)

    def capture_by_sha256(self, digest: str) -> dict[str, Any] | None:
        """Find an exact previously imported waveform for derived-data lineage."""
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM captures WHERE sha256 = ? ORDER BY created_utc LIMIT 1",
                (digest,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["source_metadata"] = json.loads(result.pop("source_metadata_json"))
        return result

    def list_captures(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM captures ORDER BY created_utc DESC").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["source_metadata"] = json.loads(item.pop("source_metadata_json"))
            result.append(item)
        return result

    def list_library_signals(self) -> list[dict[str, Any]]:
        """List active, labeled signal regions with their source recording metadata."""
        with self._connect() as db:
            rows = db.execute(
                """SELECT c.id AS capture_id,c.created_utc,c.source_kind,c.sha256,
                          c.sample_rate,c.channels,c.sample_count,c.source_metadata_json,
                          r.id AS region_id,r.origin,r.signal_id,
                          g.revision AS geometry_revision,g.sample_start,g.sample_count AS region_sample_count,
                          g.frequency_low_hz,g.frequency_high_hz,
                          l.revision,l.family,l.method,l.protocol,l.variant,l.mode_id,l.source,l.review_state
                   FROM captures c
                   JOIN regions r ON r.capture_id=c.id
                   JOIN region_geometry_revisions g ON g.region_id=r.id AND g.revision=(
                       SELECT MAX(revision) FROM region_geometry_revisions WHERE region_id=r.id
                   )
                   JOIN label_revisions l ON l.region_id=r.id AND l.revision=(
                       SELECT MAX(revision) FROM label_revisions WHERE region_id=r.id
                   )
                   WHERE l.review_state IN ('accepted','pending')
                     AND (l.mode_id IS NOT NULL OR l.family IS NOT NULL OR l.method IS NOT NULL
                          OR l.protocol IS NOT NULL OR l.variant IS NOT NULL)
                     AND NOT EXISTS(SELECT 1 FROM region_removals rr WHERE rr.region_id=r.id)
                     AND NOT EXISTS(SELECT 1 FROM region_lineage rl WHERE rl.source_region_id=r.id)
                   ORDER BY (l.review_state='accepted') DESC,c.created_utc DESC,g.sample_start"""
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            source_metadata = json.loads(item.pop("source_metadata_json"))
            region = {
                key: item.pop(key) for key in (
                    "region_id", "origin", "signal_id", "geometry_revision", "sample_start",
                    "region_sample_count", "frequency_low_hz", "frequency_high_hz", "revision",
                    "family", "method", "protocol", "variant", "mode_id", "source", "review_state",
                )
            }
            region["id"] = region.pop("region_id")
            region["sample_count"] = region.pop("region_sample_count")
            item["source_metadata"] = source_metadata
            item["region"] = region
            result.append(item)
        return result

    def create_decoder_run(
        self, *, session_id: str, track_id: str, sample_start: int | None,
        sample_count: int | None, sample_rate: int | None,
        modulation_hypothesis: str, backend: str, modes: list[str],
        attempted_modes: list[str], current_mode: str, output_offset: int,
        status: str, note: str, capture_id: str | None = None,
        output_scope: str = "shared_live_input",
        association: str = "track_window_tentative",
        carrier_hz: float | None = None,
        auto_identify: bool | None = None,
        region_id: str | None = None,
        classifier_evidence: dict[str, Any] | None = None,
        detector_evidence: dict[str, Any] | None = None,
    ) -> str:
        if not session_id or not track_id:
            raise ValueError("decoder run requires a live session and track")
        if (sample_start is None) != (sample_count is None):
            raise ValueError("decoder run sample bounds must be provided together")
        if region_id is not None:
            with self._connect() as db:
                region = db.execute("SELECT capture_id FROM regions WHERE id = ?", (region_id,)).fetchone()
            if region is None or region["capture_id"] != capture_id:
                raise ValueError("decoder run region must belong to its saved capture")
        run_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute(
                """INSERT INTO decoder_runs
                   (id, session_id, capture_id, region_id, track_id, sample_start, sample_count, sample_rate, carrier_hz, auto_identify,
                    modulation_hypothesis, backend, modes_json, attempted_modes_json,
                    current_mode, output_offset, output_end, output_text, status, note,
                    output_scope, association, classifier_evidence_json, detector_evidence_json, created_utc, updated_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, '', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, session_id, capture_id, region_id, track_id, sample_start, sample_count, sample_rate, carrier_hz,
                 int(auto_identify),
                 modulation_hypothesis, backend, json.dumps(modes),
                 json.dumps(attempted_modes), current_mode, output_offset, status,
                 note, output_scope, association, json.dumps(classifier_evidence or {}, sort_keys=True),
                 json.dumps(detector_evidence or {}, sort_keys=True), now, now),
            )
        return run_id

    def update_decoder_run(self, run_id: str, state: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            cursor = db.execute(
                """UPDATE decoder_runs SET attempted_modes_json = ?, current_mode = ?,
                   output_offset = ?, output_end = ?, output_text = ?, status = ?, note = ?, updated_utc = ?
                   WHERE id = ?""",
                (json.dumps(state.get("attempted_modes", [])), state.get("current_mode"),
                 state.get("output_offset", 0), state.get("output_end"), str(state.get("text", ""))[-65536:],
                 state.get("status", "unknown"), state.get("note", ""), now, run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown decoder run: {run_id}")

    def associate_decoder_runs(self, session_id: str, capture_id: str) -> int:
        with self._connect() as db:
            capture = db.execute("SELECT 1 FROM captures WHERE id = ?", (capture_id,)).fetchone()
            if capture is None:
                raise KeyError(f"unknown capture: {capture_id}")
            cursor = db.execute(
                "UPDATE decoder_runs SET capture_id = ?, updated_utc = ? WHERE session_id = ? AND capture_id IS NULL",
                (capture_id, datetime.now(timezone.utc).isoformat(), session_id),
            )
            return cursor.rowcount

    def decoder_runs(self, *, capture_id: str | None = None, session_id: str | None = None) -> list[dict[str, Any]]:
        if (capture_id is None) == (session_id is None):
            raise ValueError("provide exactly one of capture_id or session_id")
        with self._connect() as db:
            if capture_id is not None:
                exists = db.execute("SELECT 1 FROM captures WHERE id = ?", (capture_id,)).fetchone()
                if exists is None:
                    raise KeyError(f"unknown capture: {capture_id}")
                rows = db.execute("SELECT * FROM decoder_runs WHERE capture_id = ? ORDER BY created_utc", (capture_id,)).fetchall()
            else:
                rows = db.execute("SELECT * FROM decoder_runs WHERE session_id = ? ORDER BY created_utc", (session_id,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            if item["auto_identify"] is not None:
                item["auto_identify"] = bool(item["auto_identify"])
            item["modes"] = json.loads(item.pop("modes_json"))
            item["attempted_modes"] = json.loads(item.pop("attempted_modes_json"))
            item["classifier_evidence"] = json.loads(item.pop("classifier_evidence_json"))
            item["detector_evidence"] = json.loads(item.pop("detector_evidence_json"))
            # Scope and association are explicit so shared live audio can never
            # be mistaken for an isolated decode of the candidate region.
            result.append(item)
        return result

    def delete_capture(self, capture_id: str) -> bool:
        """Remove a capture and its regions/labels from the local index."""
        with self._connect() as db:
            cursor = db.execute("DELETE FROM captures WHERE id = ?", (capture_id,))
        return cursor.rowcount > 0

    def delete_region(
        self, capture_id: str, region_id: str, *,
        expected_revision: int, expected_geometry_revision: int,
    ) -> None:
        """Remove an active region, retaining history when lineage references it."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                """SELECT
                       (SELECT MAX(revision) FROM label_revisions WHERE region_id=r.id) AS label_revision,
                       (SELECT MAX(revision) FROM region_geometry_revisions WHERE region_id=r.id) AS geometry_revision
                   FROM regions r WHERE r.id=? AND r.capture_id=?""",
                (region_id, capture_id),
            ).fetchone()
            if current is None:
                raise KeyError(f"region not found in capture: {region_id}")
            if db.execute("SELECT 1 FROM region_removals WHERE region_id=?", (region_id,)).fetchone():
                raise LabelRevisionConflict("this signal was already removed from the active annotations")
            if (current["label_revision"] != expected_revision
                    or current["geometry_revision"] != expected_geometry_revision):
                raise LabelRevisionConflict("region changed since it was opened; reopen it before deleting")
            lineage_source = db.execute(
                "SELECT 1 FROM region_lineage WHERE source_region_id=? LIMIT 1", (region_id,)
            ).fetchone()
            lineage_result = db.execute(
                "SELECT 1 FROM region_lineage WHERE result_region_id=? LIMIT 1", (region_id,)
            ).fetchone()
            if lineage_source is not None:
                raise ValueError("a superseded signal cannot be removed; remove its active replacement instead")
            if lineage_result is not None:
                db.execute(
                    """INSERT INTO region_removals
                       (region_id,capture_id,label_revision,geometry_revision,reason,created_utc)
                       VALUES(?,?,?,?,?,?)""",
                    (region_id, capture_id, expected_revision, expected_geometry_revision,
                     "Removed during annotation review.", datetime.now(timezone.utc).isoformat()),
                )
            else:
                db.execute("DELETE FROM regions WHERE id=? AND capture_id=?", (region_id, capture_id))

    def region_exists(self, capture_id: str, region_id: str) -> bool:
        with self._connect() as db:
            return db.execute(
                "SELECT 1 FROM regions WHERE id=? AND capture_id=?", (region_id, capture_id)
            ).fetchone() is not None

    def regions(self, capture_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            exists = db.execute("SELECT 1 FROM captures WHERE id = ?", (capture_id,)).fetchone()
            if exists is None:
                raise KeyError(f"unknown capture: {capture_id}")
            rows = db.execute(
                """SELECT r.*, g.revision AS geometry_revision,
                          g.sample_start AS current_sample_start, g.sample_count AS current_sample_count,
                          g.frequency_low_hz AS current_frequency_low_hz, g.frequency_high_hz AS current_frequency_high_hz,
                          CASE WHEN EXISTS(SELECT 1 FROM region_lineage rl WHERE rl.source_region_id=r.id)
                               THEN 'superseded' ELSE 'active' END AS lineage_state,
                          EXISTS(SELECT 1 FROM region_lineage rl
                                 WHERE rl.source_region_id=r.id OR rl.result_region_id=r.id) AS lineage_related,
                          l.revision, l.family, l.method, l.protocol, l.variant, l.mode_id,
                          l.source, l.review_state, l.evidence_json, l.notes
                   FROM regions r
                   JOIN region_geometry_revisions g ON g.region_id=r.id AND g.revision=(
                       SELECT MAX(revision) FROM region_geometry_revisions WHERE region_id=r.id
                   )
                   JOIN label_revisions l ON l.region_id = r.id
                   WHERE r.capture_id = ? AND NOT EXISTS(
                       SELECT 1 FROM region_removals rr WHERE rr.region_id=r.id
                   ) AND l.revision = (
                       SELECT MAX(revision) FROM label_revisions WHERE region_id = r.id
                   ) ORDER BY g.sample_start""",
                (capture_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["sample_start"] = item.pop("current_sample_start")
            item["sample_count"] = item.pop("current_sample_count")
            item["frequency_low_hz"] = item.pop("current_frequency_low_hz")
            item["frequency_high_hz"] = item.pop("current_frequency_high_hz")
            item["evidence"] = json.loads(item.pop("evidence_json"))
            result.append(item)
        return result

    def signal_identities(self) -> list[dict[str, Any]]:
        """List persisted signal identities that still have active region members."""
        with self._connect() as db:
            rows = db.execute(
                """SELECT s.id AS signal_id,COUNT(*) AS region_count,
                          MIN(c.created_utc) AS first_seen_utc
                   FROM signal_identities s
                   JOIN regions r ON r.signal_id=s.id
                   JOIN captures c ON c.id=r.capture_id
                   WHERE NOT EXISTS(SELECT 1 FROM region_removals rr WHERE rr.region_id=r.id)
                     AND NOT EXISTS(SELECT 1 FROM region_lineage rl WHERE rl.source_region_id=r.id)
                   GROUP BY s.id ORDER BY CAST(substr(s.id,2) AS INTEGER)"""
            ).fetchall()
        return [dict(row) for row in rows]

    def signal_identity_history(self, region_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            exists = db.execute("SELECT 1 FROM regions WHERE id=?", (region_id,)).fetchone()
            if exists is None:
                raise KeyError(f"unknown region: {region_id}")
            rows = db.execute(
                "SELECT revision,signal_id,created_utc FROM region_signal_identity_revisions WHERE region_id=? ORDER BY revision",
                (region_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def current_cv_embedding_regions(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Snapshot active accepted regions eligible to back current CV embeddings."""
        with self._connect() as db:
            rows = db.execute(
                """SELECT r.id AS region_id,r.capture_id,c.sha256 AS capture_sha256,
                          g.revision AS geometry_revision,g.sample_start,g.sample_count,
                          g.frequency_low_hz,g.frequency_high_hz,
                          l.revision AS label_revision,l.family,l.method,l.protocol,l.variant,
                          l.review_state,l.evidence_json,
                          CASE WHEN EXISTS(SELECT 1 FROM region_lineage rl WHERE rl.source_region_id=r.id)
                               OR EXISTS(SELECT 1 FROM region_removals rr WHERE rr.region_id=r.id)
                               THEN 0 ELSE 1 END AS is_active
                   FROM regions r
                   JOIN captures c ON c.id=r.capture_id
                   JOIN region_geometry_revisions g ON g.region_id=r.id AND g.revision=(
                       SELECT MAX(revision) FROM region_geometry_revisions WHERE region_id=r.id
                   )
                   JOIN label_revisions l ON l.region_id=r.id AND l.revision=(
                       SELECT MAX(revision) FROM label_revisions WHERE region_id=r.id
                   )"""
            ).fetchall()
        current: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            item = dict(row)
            evidence = json.loads(item.pop("evidence_json"))
            if (
                not item.pop("is_active")
                or item["review_state"] != "accepted"
                or evidence.get("generator") in {"sig-id-noise", "sig-id-recorded-noise"}
                or evidence.get("noise_only") is True
            ):
                continue
            current[(item["capture_id"], item["region_id"])] = item
        return current

    def label_history(self, region_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM label_revisions WHERE region_id = ? ORDER BY revision",
                (region_id,),
            ).fetchall()
        if not rows:
            with self._connect() as db:
                exists = db.execute("SELECT 1 FROM regions WHERE id = ?", (region_id,)).fetchone()
            if exists is None:
                raise KeyError(f"unknown region: {region_id}")
        result = []
        for row in rows:
            item = dict(row)
            item["evidence"] = json.loads(item.pop("evidence_json"))
            result.append(item)
        return result

    def export_jsonl(self, output: str | Path) -> int:
        """Export each region's current label with source identity and sample bounds."""
        with self._connect() as db:
            rows = db.execute(
                """SELECT r.*, g.revision AS geometry_revision,
                          g.sample_start AS current_sample_start, g.sample_count AS current_sample_count,
                          g.frequency_low_hz AS current_frequency_low_hz, g.frequency_high_hz AS current_frequency_high_hz,
                          c.sha256, c.source_kind, c.sample_rate, c.audio_path,
                          l.family, l.method, l.protocol, l.variant, l.mode_id, l.source,
                          l.review_state, l.evidence_json, l.revision
                   FROM regions r
                   JOIN region_geometry_revisions g ON g.region_id=r.id AND g.revision=(
                       SELECT MAX(revision) FROM region_geometry_revisions WHERE region_id=r.id
                   )
                   JOIN captures c ON c.id = r.capture_id
                   JOIN label_revisions l ON l.region_id = r.id
                   WHERE l.revision = (SELECT MAX(revision) FROM label_revisions WHERE region_id = r.id)
                     AND NOT EXISTS(SELECT 1 FROM region_removals rr WHERE rr.region_id=r.id)
                   ORDER BY c.created_utc, g.sample_start"""
            ).fetchall()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            for row in rows:
                record = dict(row)
                record["sample_start"] = record.pop("current_sample_start")
                record["sample_count"] = record.pop("current_sample_count")
                record["frequency_low_hz"] = record.pop("current_frequency_low_hz")
                record["frequency_high_hz"] = record.pop("current_frequency_high_hz")
                record["evidence"] = json.loads(record.pop("evidence_json"))
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return len(rows)
