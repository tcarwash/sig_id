"""Opt-in contribution and peer extent-review hub."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sqlite3
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field


class ReviewCreate(BaseModel):
    revision: int = Field(ge=1)
    decision: str
    proposed_sample_start: int | None = Field(default=None, ge=0)
    proposed_sample_count: int | None = Field(default=None, gt=0)
    proposed_frequency_low_hz: float | None = None
    proposed_frequency_high_hz: float | None = None
    note: str = Field(default="", max_length=2000)


def create_hub_app(
    data_dir: str | Path | None = None,
    *,
    upload_token: str | None = None,
    reviewer_tokens: dict[str, str] | set[str] | None = None,
    max_upload_bytes: int = 64 * 1024 * 1024,
) -> FastAPI:
    root = Path(data_dir or os.environ.get("SIG_ID_HUB_DATA_DIR", "/data"))
    root.mkdir(parents=True, exist_ok=True)
    assets = root / "assets"
    assets.mkdir(exist_ok=True)
    raw_upload_tokens = upload_token if upload_token is not None else os.environ.get("SIG_ID_HUB_UPLOAD_TOKEN", "")
    upload_secrets = {identity: secret for item in raw_upload_tokens.split(",") if item for identity, secret in [item.split("=", 1)]} if "=" in raw_upload_tokens else ({"collector": raw_upload_tokens} if raw_upload_tokens else {})
    if reviewer_tokens is None:
        reviewer_tokens = {
            key: value for item in os.environ.get("SIG_ID_HUB_REVIEWERS", "").split(",")
            if "=" in item for key, value in [item.split("=", 1)]
        }
    if isinstance(reviewer_tokens, set):
        reviewer_secrets = {token: token for token in reviewer_tokens}
    else:
        reviewer_secrets = {secret: reviewer for reviewer, secret in reviewer_tokens.items()}
    db_path = root / "hub.sqlite3"

    def connect() -> sqlite3.Connection:
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    with connect() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS submissions (
                id TEXT PRIMARY KEY, created_utc TEXT NOT NULL, sha256 TEXT NOT NULL,
                audio_path TEXT NOT NULL, byte_count INTEGER NOT NULL,
                manifest_json TEXT NOT NULL, contributor TEXT NOT NULL,
                UNIQUE(contributor, sha256)
            );
            CREATE TABLE IF NOT EXISTS annotations (
                id TEXT PRIMARY KEY, submission_id TEXT NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
                revision INTEGER NOT NULL, sample_start INTEGER NOT NULL, sample_count INTEGER NOT NULL,
                frequency_low_hz REAL, frequency_high_hz REAL, family TEXT, method TEXT,
                protocol TEXT, variant TEXT, completeness TEXT NOT NULL,
                UNIQUE(id, revision)
            );
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT, annotation_id TEXT NOT NULL,
                revision INTEGER NOT NULL, reviewer TEXT NOT NULL, decision TEXT NOT NULL,
                proposed_json TEXT NOT NULL, note TEXT NOT NULL, created_utc TEXT NOT NULL,
                UNIQUE(annotation_id, revision, reviewer),
                FOREIGN KEY(annotation_id, revision) REFERENCES annotations(id, revision)
            );
        """)

    app = FastAPI(title="Signal ID Contribution Hub", version="0.1.0")
    app.state.hub_root = root

    def bearer(authorization: str | None) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="bearer token required")
        return authorization[7:]

    def reviewer(authorization: str | None = Header(default=None)) -> str:
        token = bearer(authorization)
        for secret, identity in reviewer_secrets.items():
            if hmac.compare_digest(token, secret):
                return identity
        raise HTTPException(status_code=401, detail="invalid reviewer token")

    def collector(authorization: str | None = Header(default=None)) -> str:
        token = bearer(authorization)
        for identity, secret in upload_secrets.items():
            if hmac.compare_digest(token, secret):
                return identity
        if not upload_secrets:
            raise HTTPException(status_code=401, detail="invalid collector token")
        raise HTTPException(status_code=401, detail="invalid collector token")

    def require_reviewable(row: sqlite3.Row | None) -> sqlite3.Row:
        if row is None:
            raise HTTPException(status_code=404, detail="submission or annotation not found")
        return row

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/submissions", status_code=201)
    async def submit(audio: UploadFile = File(...), manifest: str = Form(...), _actor: str = Depends(collector)) -> dict[str, Any]:
        try:
            data = json.loads(manifest)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail="manifest must be valid JSON") from exc
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise HTTPException(status_code=422, detail="schema_version 1 is required")
        sample_rate, channels, sample_count = (data.get(key) for key in ("sample_rate", "channels", "sample_count"))
        if any(type(value) is not int or value <= 0 for value in (sample_rate, channels)) or type(sample_count) is not int or sample_count < 0:
            raise HTTPException(status_code=422, detail="invalid audio dimensions")
        chunks = bytearray()
        while chunk := await audio.read(1024 * 1024):
            if len(chunks) + len(chunk) > max_upload_bytes:
                raise HTTPException(status_code=413, detail="audio exceeds upload limit")
            chunks.extend(chunk)
        if not chunks:
            raise HTTPException(status_code=422, detail="audio is empty")
        temp_path = root / f".{uuid.uuid4().hex}.upload"
        temp_path.write_bytes(chunks)
        try:
            try:
                with wave.open(str(temp_path), "rb") as wav:
                    actual = (wav.getframerate(), wav.getnchannels(), wav.getnframes())
                    if wav.getcomptype() != "NONE":
                        raise HTTPException(status_code=422, detail="only uncompressed WAV is accepted")
            except (wave.Error, EOFError) as exc:
                raise HTTPException(status_code=422, detail="audio must be a valid PCM WAV") from exc
            if actual != (sample_rate, channels, sample_count):
                raise HTTPException(status_code=422, detail="manifest audio dimensions do not match WAV")
            annotations = data.get("annotations")
            if not isinstance(annotations, list) or not annotations:
                raise HTTPException(status_code=422, detail="at least one annotation is required")
            normalized = []
            for item in annotations:
                if not isinstance(item, dict):
                    raise HTTPException(status_code=422, detail="annotation must be an object")
                start, count = item.get("sample_start"), item.get("sample_count")
                if type(start) is not int or type(count) is not int or start < 0 or count <= 0 or start + count > sample_count:
                    raise HTTPException(status_code=422, detail="annotation sample range exceeds audio")
                low, high = item.get("frequency_low_hz"), item.get("frequency_high_hz")
                if (low is None) != (high is None) or (low is not None and (not isinstance(low, (int, float)) or not isinstance(high, (int, float)) or low >= high or low < 0 or high > sample_rate / 2)):
                    raise HTTPException(status_code=422, detail="invalid annotation frequency bounds")
                completeness = item.get("completeness", "unknown")
                if completeness not in {"complete", "left_censored", "right_censored", "both_censored", "unknown"}:
                    raise HTTPException(status_code=422, detail="invalid completeness value")
                normalized.append({**item, "completeness": completeness})
            data["annotations"] = normalized
            digest = hashlib.sha256(chunks).hexdigest()
            with connect() as db:
                prior = db.execute("SELECT * FROM submissions WHERE contributor=? AND sha256=?", (_actor, digest)).fetchone()
                if prior:
                    temp_path.unlink(missing_ok=True)
                    return submission_view(prior, db)
                submission_id = str(uuid.uuid4())
                final_path = assets / f"{digest}.wav"
                if final_path.exists():
                    temp_path.unlink(missing_ok=True)
                else:
                    temp_path.replace(final_path)
                db.execute("INSERT INTO submissions VALUES (?, ?, ?, ?, ?, ?, ?)", (submission_id, datetime.now(timezone.utc).isoformat(), digest, str(final_path), len(chunks), json.dumps(data, sort_keys=True), _actor))
                for item in normalized:
                    annotation_id = str(uuid.uuid4())
                    db.execute("INSERT INTO annotations VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (annotation_id, submission_id, item["sample_start"], item["sample_count"], item.get("frequency_low_hz"), item.get("frequency_high_hz"), item.get("family"), item.get("method"), item.get("protocol"), item.get("variant"), item["completeness"]))
            return {"id": submission_id, "sha256": digest, "status": "awaiting_review", "annotations": [{"id": row_id} for row_id in _annotation_ids(db_path, submission_id)]}
        finally:
            temp_path.unlink(missing_ok=True)

    def submission_view(row: sqlite3.Row, db: sqlite3.Connection) -> dict[str, Any]:
        annotations = db.execute("SELECT * FROM annotations WHERE submission_id=?", (row["id"],)).fetchall()
        return {"id": row["id"], "sha256": row["sha256"], "status": "awaiting_review", "annotations": [{"id": item["id"], "revision": item["revision"]} for item in annotations]}

    @app.get("/api/review/queue")
    def queue(_reviewer: str = Depends(reviewer)) -> list[dict[str, Any]]:
        with connect() as db:
            rows = db.execute("SELECT s.*, a.id AS annotation_id, a.revision, a.sample_start, a.sample_count, a.frequency_low_hz, a.frequency_high_hz, a.family, a.method, a.protocol, a.variant, a.completeness FROM submissions s JOIN annotations a ON a.submission_id=s.id ORDER BY s.created_utc").fetchall()
            result = []
            for row in rows:
                review_rows = db.execute("SELECT reviewer, decision, proposed_json, note FROM reviews WHERE annotation_id=? AND revision=? ORDER BY id", (row["annotation_id"], row["revision"])).fetchall()
                result.append({"submission_id": row["id"], "sha256": row["sha256"], "audio_url": f"/api/submissions/{row['id']}/audio", "manifest": json.loads(row["manifest_json"]), "annotation": {key: row[key] for key in ("annotation_id", "revision", "sample_start", "sample_count", "frequency_low_hz", "frequency_high_hz", "family", "method", "protocol", "variant", "completeness")}, "reviews": [{"reviewer": r["reviewer"], "decision": r["decision"], "proposed": json.loads(r["proposed_json"]), "note": r["note"]} for r in review_rows], "consensus": _consensus(review_rows)})
            return result

    @app.get("/api/submissions/{submission_id}/audio")
    def audio(submission_id: str, _reviewer: str = Depends(reviewer)) -> FileResponse:
        with connect() as db:
            row = db.execute("SELECT audio_path FROM submissions WHERE id=?", (submission_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="submission not found")
        return FileResponse(row["audio_path"], media_type="audio/wav", filename=f"{submission_id}.wav")

    @app.post("/api/submissions/{submission_id}/annotations/{annotation_id}/reviews", status_code=201)
    def create_review(submission_id: str, annotation_id: str, data: ReviewCreate, reviewer_id: str = Depends(reviewer)) -> dict[str, Any]:
        if data.decision not in {"looks_right", "adjust_bounds", "multiple_signals", "not_signal", "unsure"}:
            raise HTTPException(status_code=422, detail="unsupported review decision")
        with connect() as db:
            annotation = db.execute("SELECT a.*,s.contributor FROM annotations a JOIN submissions s ON s.id=a.submission_id WHERE a.id=? AND s.id=?", (annotation_id, submission_id)).fetchone()
            require_reviewable(annotation)
            if annotation["contributor"] == reviewer_id:
                raise HTTPException(status_code=403, detail="contributors cannot review their own submission")
            if data.revision != annotation["revision"]:
                raise HTTPException(status_code=409, detail="annotation revision is stale")
            prior_geometry = {key: annotation[key] for key in ("sample_start", "sample_count", "frequency_low_hz", "frequency_high_hz")}
            proposed = {key: getattr(data, f"proposed_{key}") for key in prior_geometry if getattr(data, f"proposed_{key}") is not None}
            geometry = prior_geometry | proposed
            if geometry["sample_start"] + geometry["sample_count"] > int(json.loads(db.execute("SELECT manifest_json FROM submissions WHERE id=?", (submission_id,)).fetchone()[0])["sample_count"]):
                raise HTTPException(status_code=422, detail="proposed bounds exceed audio")
            low, high = geometry["frequency_low_hz"], geometry["frequency_high_hz"]
            if (low is None) != (high is None) or (low is not None and (low < 0 or low >= high)):
                raise HTTPException(status_code=422, detail="invalid proposed frequency bounds")
            try:
                db.execute("INSERT INTO reviews(annotation_id,revision,reviewer,decision,proposed_json,note,created_utc) VALUES(?,?,?,?,?,?,?)", (annotation_id, data.revision, reviewer_id, data.decision, json.dumps(proposed, sort_keys=True), data.note, datetime.now(timezone.utc).isoformat()))
            except sqlite3.IntegrityError as exc:
                raise HTTPException(status_code=409, detail="reviewer already reviewed this revision") from exc
            reviews = db.execute("SELECT reviewer,decision FROM reviews WHERE annotation_id=? AND revision=?", (annotation_id, data.revision)).fetchall()
            return {"annotation_id": annotation_id, "revision": data.revision, "consensus": _consensus(reviews), "review_count": len(reviews)}

    @app.get("/review", response_class=HTMLResponse)
    def review_ui() -> str:
        return REVIEW_HTML

    return app


def _annotation_ids(db_path: Path, submission_id: str) -> list[str]:
    with sqlite3.connect(db_path) as db:
        return [row[0] for row in db.execute("SELECT id FROM annotations WHERE submission_id=?", (submission_id,))]


def _consensus(reviews: Any) -> str:
    decisions = [row["decision"] for row in reviews]
    if any(item != "looks_right" for item in decisions):
        return "disputed" if len(decisions) > 1 else "needs_attention"
    reviewers = {row["reviewer"] for row in reviews}
    return "extent_verified" if len(reviewers) >= 2 else "awaiting_second_review"


REVIEW_HTML = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Signal ID review hub</title><style>body{font:16px system-ui;max-width:960px;margin:2rem auto;padding:0 1rem;background:#10151d;color:#e8edf4}input,button{font:inherit;padding:.6rem;margin:.25rem}article{border:1px solid #394656;border-radius:12px;padding:1rem;margin:1rem 0}audio{width:100%}small{color:#9dacbd}</style><h1>Signal extent review</h1><p>Review the audio and context. Bounds approval does not assert protocol identity or that the recording contains the entire transmission.</p><label>Reviewer token <input id="token" type="password" autocomplete="off"></label><button onclick="loadQueue()">Load queue</button><main id="queue"></main><script>async function loadQueue(){let t=document.querySelector('#token').value,r=await fetch('/api/review/queue',{headers:{Authorization:'Bearer '+t}}),q=document.querySelector('#queue');if(!r.ok){q.textContent='Could not load queue ('+r.status+')';return}q.replaceChildren();for(let x of await r.json()){let a=document.createElement('article'),audio=document.createElement('audio');audio.controls=true;let asset=await fetch(x.audio_url,{headers:{Authorization:'Bearer '+t}});if(asset.ok)audio.src=URL.createObjectURL(await asset.blob());a.innerHTML='<h2>Submission '+x.submission_id+'</h2><small>SHA-256 '+x.sha256+' · annotation '+x.annotation.annotation_id+' · revision '+x.annotation.revision+' · current consensus '+x.consensus+'</small>';a.append(audio);let info=document.createElement('p');info.textContent='Samples '+x.annotation.sample_start+'–'+(x.annotation.sample_start+x.annotation.sample_count)+'; '+x.annotation.completeness+' transmission coverage';a.append(info);let sel=document.createElement('select');for(let d of ['looks_right','adjust_bounds','multiple_signals','not_signal','unsure']){let o=document.createElement('option');o.value=d;o.textContent=d.replaceAll('_',' ');sel.append(o)}let note=document.createElement('input');note.placeholder='Optional note';let b=document.createElement('button');b.textContent='Submit review';b.onclick=async()=>{let res=await fetch('/api/submissions/'+x.submission_id+'/annotations/'+x.annotation.annotation_id+'/reviews',{method:'POST',headers:{Authorization:'Bearer '+t,'Content-Type':'application/json'},body:JSON.stringify({revision:x.annotation.revision,decision:sel.value,note:note.value})});b.textContent=res.ok?'Review recorded':(await res.json()).detail};a.append(sel,note,b);q.append(a)}}</script>'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Signal ID contribution hub")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_hub_app(), host=args.host, port=args.port)
