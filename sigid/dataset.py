"""Versioned SQLite metadata for recordings, signal regions and label history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from .wav import WavSource


SOURCE_KINDS = {"live", "wav", "synthetic", "reference"}
LABEL_SOURCES = {"unknown", "prediction", "rsid", "decoder", "human", "synthetic", "external"}
REVIEW_STATES = {"pending", "accepted", "rejected"}


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
            if version not in (0, 1):
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
                CREATE TABLE IF NOT EXISTS regions (
                    id TEXT PRIMARY KEY,
                    capture_id TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
                    sample_start INTEGER NOT NULL CHECK(sample_start >= 0),
                    sample_count INTEGER NOT NULL CHECK(sample_count > 0),
                    frequency_low_hz REAL,
                    frequency_high_hz REAL,
                    created_utc TEXT NOT NULL,
                    CHECK((frequency_low_hz IS NULL AND frequency_high_hz IS NULL) OR
                          (frequency_low_hz IS NOT NULL AND frequency_high_hz IS NOT NULL AND
                           frequency_low_hz < frequency_high_hz))
                );
                CREATE INDEX IF NOT EXISTS regions_by_capture ON regions(capture_id, sample_start);
                CREATE INDEX IF NOT EXISTS captures_by_sha256 ON captures(sha256);
                CREATE TABLE IF NOT EXISTS label_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    region_id TEXT NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
                    revision INTEGER NOT NULL CHECK(revision > 0),
                    created_utc TEXT NOT NULL,
                    family TEXT,
                    method TEXT,
                    protocol TEXT,
                    variant TEXT,
                    source TEXT NOT NULL CHECK(source IN ('unknown','prediction','rsid','decoder','human','synthetic','external')),
                    review_state TEXT NOT NULL CHECK(review_state IN ('pending','accepted','rejected')),
                    evidence_json TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    UNIQUE(region_id, revision)
                );
                CREATE INDEX IF NOT EXISTS labels_by_region ON label_revisions(region_id, revision DESC);
                PRAGMA user_version = 1;
                """
            )

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
    ) -> str:
        if sample_start < 0 or sample_count <= 0:
            raise ValueError("region sample_start must be non-negative and sample_count positive")
        if (frequency_low_hz is None) != (frequency_high_hz is None):
            raise ValueError("both frequency bounds must be provided together")
        if frequency_low_hz is not None and frequency_low_hz >= frequency_high_hz:
            raise ValueError("frequency_low_hz must be less than frequency_high_hz")
        chosen_label = label or SignalLabel()
        self._validate_label(chosen_label)

        region_id = str(uuid4())
        with self._connect() as db:
            capture = db.execute(
                "SELECT sample_count FROM captures WHERE id = ?", (capture_id,)
            ).fetchone()
            if capture is None:
                raise KeyError(f"unknown capture: {capture_id}")
            if sample_start + sample_count > capture["sample_count"]:
                raise ValueError("region exceeds the parent capture sample range")
            db.execute(
                "INSERT INTO regions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    region_id,
                    capture_id,
                    sample_start,
                    sample_count,
                    frequency_low_hz,
                    frequency_high_hz,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._insert_label(db, region_id, 1, chosen_label)
        return region_id

    @staticmethod
    def _validate_label(label: SignalLabel) -> None:
        if label.source not in LABEL_SOURCES:
            raise ValueError(f"unknown label source: {label.source}")
        if label.review_state not in REVIEW_STATES:
            raise ValueError(f"unknown review state: {label.review_state}")

    @staticmethod
    def _insert_label(db: sqlite3.Connection, region_id: str, revision: int, label: SignalLabel) -> None:
        db.execute(
            """INSERT INTO label_revisions
               (region_id, revision, created_utc, family, method, protocol, variant,
                source, review_state, evidence_json, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                region_id,
                revision,
                datetime.now(timezone.utc).isoformat(),
                label.family,
                label.method,
                label.protocol,
                label.variant,
                label.source,
                label.review_state,
                json.dumps(label.evidence or {}, sort_keys=True),
                label.notes,
            ),
        )

    def revise_label(self, region_id: str, label: SignalLabel) -> int:
        self._validate_label(label)
        with self._connect() as db:
            latest = db.execute(
                "SELECT MAX(revision) AS revision FROM label_revisions WHERE region_id = ?",
                (region_id,),
            ).fetchone()["revision"]
            if latest is None:
                raise KeyError(f"unknown region: {region_id}")
            revision = latest + 1
            self._insert_label(db, region_id, revision, label)
            return revision

    def capture(self, capture_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM captures WHERE id = ?", (capture_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown capture: {capture_id}")
        result = dict(row)
        result["source_metadata"] = json.loads(result.pop("source_metadata_json"))
        return result

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

    def delete_capture(self, capture_id: str) -> bool:
        """Remove a capture and its regions/labels from the local index."""
        with self._connect() as db:
            cursor = db.execute("DELETE FROM captures WHERE id = ?", (capture_id,))
        return cursor.rowcount > 0

    def regions(self, capture_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            exists = db.execute("SELECT 1 FROM captures WHERE id = ?", (capture_id,)).fetchone()
            if exists is None:
                raise KeyError(f"unknown capture: {capture_id}")
            rows = db.execute(
                """SELECT r.*, l.revision, l.family, l.method, l.protocol, l.variant,
                          l.source, l.review_state, l.evidence_json, l.notes
                   FROM regions r JOIN label_revisions l ON l.region_id = r.id
                   WHERE r.capture_id = ? AND l.revision = (
                       SELECT MAX(revision) FROM label_revisions WHERE region_id = r.id
                   ) ORDER BY r.sample_start""",
                (capture_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["evidence"] = json.loads(item.pop("evidence_json"))
            result.append(item)
        return result

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
                """SELECT r.*, c.sha256, c.source_kind, c.sample_rate, c.audio_path,
                          l.family, l.method, l.protocol, l.variant, l.source,
                          l.review_state, l.evidence_json, l.revision
                   FROM regions r
                   JOIN captures c ON c.id = r.capture_id
                   JOIN label_revisions l ON l.region_id = r.id
                   WHERE l.revision = (SELECT MAX(revision) FROM label_revisions WHERE region_id = r.id)
                   ORDER BY c.created_utc, r.sample_start"""
            ).fetchall()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            for row in rows:
                record = dict(row)
                record["evidence"] = json.loads(record.pop("evidence_json"))
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return len(rows)
