"""Import extra receiver WAV clips for HUMAN REVIEW, never training by default.

Source SHA256 and session IDs are preserved; only explicit eligible input
devices are imported, signal-lab loopback is excluded. Captures have no
regions or coverage and therefore contribute zero supervised training
examples until the user reviews and confirms them in the workbench.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from sigid.dataset import CaptureStore


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_receiver_candidates(source_data_dir: Path, destination_data_dir: Path, *, device_match: str, backup: Path) -> dict:
    source_data_dir = source_data_dir.expanduser().resolve(strict=True)
    destination_data_dir = destination_data_dir.expanduser().resolve(strict=True)
    if source_data_dir == destination_data_dir:
        raise ValueError("cannot import source database into itself")
    source_db = source_data_dir / "dataset.sqlite3"
    dest_db = destination_data_dir / "dataset.sqlite3"
    if not source_db.is_file() or not dest_db.is_file():
        raise ValueError("source or destination capture database is missing")
    source = CaptureStore(source_db)
    destination = CaptureStore(dest_db)
    existing = {capture["sha256"] for capture in destination.list_captures()}
    items = source.list_captures()
    allowed = []
    skipped = {"non_receiver": 0, "duplicate_sha": 0, "non_window": 0}
    for capture in items:
        device = str(capture["source_metadata"].get("device_name") or "")
        if (capture["source_kind"] != "live" or device_match.casefold() not in device.casefold()
                or "sigid_signal_lab" in device.casefold()):
            skipped["non_receiver"] += 1
            continue
        if capture["sha256"] in existing:
            skipped["duplicate_sha"] += 1
            continue
        if abs(capture["sample_count"] / capture["sample_rate"] - 10) > 0.2:
            skipped["non_window"] += 1
            continue
        path = Path(capture["audio_path"])
        if not path.is_file() or _sha(path) != capture["sha256"]:
            raise ValueError("source recording has missing or mismatched waveform SHA")
        allowed.append(capture)
    backup = backup.expanduser().resolve()
    if backup.exists():
        raise FileExistsError("refusing to overwrite capture database backup")
    backup.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(dest_db) as live:
        with sqlite3.connect(backup) as snapshot:
            live.backup(snapshot)
    imported = []
    copied = []
    try:
        (destination_data_dir / "captures").mkdir(parents=True, exist_ok=True)
        for capture in allowed:
            dest_wav = destination_data_dir / "captures" / f"review-extra-{capture['sha256']}.wav"
            if dest_wav.exists():
                if _sha(dest_wav) != capture["sha256"]:
                    raise ValueError("existing destination WAV differs from imported recording")
            else:
                shutil.copy2(capture["audio_path"], dest_wav)
                copied.append(dest_wav)
            meta = dict(capture["source_metadata"])
            meta["auto_review_queue"] = True
            meta["active_learning_unverified_source"] = True
            meta["active_learning_import_policy"] = "waveform-verified-review-only-v1"
            meta["review_trigger"] = "external_receiver_review"
            meta.pop("auto_review_queue_reviewed_utc", None)
            capture_id = destination.add_wav_capture(dest_wav, source_kind="live", source_metadata=meta)
            imported.append(capture_id)
    except Exception:
        # Undo imports made in this invocation only; original recordings and
        # preexisting user reviews remain untouched.
        for capture_id in reversed(imported):
            destination.delete_capture(capture_id)
        for path in copied:
            path.unlink(missing_ok=True)
        raise
    return {
        "imported_count": len(imported), "candidate_count": len(allowed),
        "skipped": skipped, "review_only": True,
        "newly_verified_labels": 0,
        "backup_path": str(backup),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument("--destination-data-dir", type=Path, required=True)
    parser.add_argument("--device-match", default="USB Audio CODEC")
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    result = import_receiver_candidates(
        args.source_data_dir, args.destination_data_dir,
        device_match=args.device_match, backup=args.backup,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
