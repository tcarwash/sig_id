"""Generate a frozen-holdout noise exclusion manifest without altering sources.

This guards against training on signal-free audio from the same recording
session as a real-world validation/test example. Different waveform hashes
from one recording session are NOT independent data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sigid.dataset import CaptureStore


def prepare_exclusions(
    noise_data_dir: str | Path, holdout_windows: list[str | Path],
) -> dict:
    if not holdout_windows:
        raise ValueError("at least one frozen holdout windows.jsonl is required")
    noise_root = Path(noise_data_dir).expanduser().resolve(strict=True)
    store = CaptureStore(noise_root / "dataset.sqlite3")
    captures = store.list_captures()
    by_sha = {capture["sha256"]: capture for capture in captures}
    frozen_sha: set[str] = set()
    holdout_files: list[dict] = []
    for source in holdout_windows:
        path = Path(source).expanduser().resolve(strict=True)
        contents = path.read_bytes()
        if not contents:
            raise ValueError(f"holdout window list is empty: {path}")
        hits = 0
        for line in contents.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") not in {"test", "validation"}:
                continue
            sha = row.get("capture_sha256")
            if not isinstance(sha, str) or len(sha) != 64:
                raise ValueError("holdout row lacks a valid capture SHA")
            frozen_sha.add(sha)
            hits += 1
        if hits == 0:
            raise ValueError(f"supplied holdout contains no evaluation windows: {path}")
        holdout_files.append({
            "windows_jsonl": str(path),
            "sha256": hashlib.sha256(contents).hexdigest(),
            "evaluation_rows": hits,
        })
    missing = sorted(frozen_sha - set(by_sha))
    if missing:
        raise ValueError(f"{len(missing)} heldout capture hashes unavailable in supplied noise dataset")
    sessions = {
        c["source_metadata"].get("auto_capture_session_id")
        for sha in frozen_sha
        for c in [by_sha[sha]]
        if c["source_metadata"].get("auto_capture_session_id")
    }
    other_records = {
        c["sha256"] for c in captures
        if c["source_metadata"].get("auto_capture_session_id") in sessions
    }
    excluded = frozen_sha | other_records
    return {
        "schema_version": 1,
        "excluded_capture_sha256": sorted(excluded),
        "frozen_holdout_capture_sha256": sorted(frozen_sha),
        "frozen_holdout_count": len(frozen_sha),
        "frozen_capture_session_count": len(sessions),
        "excluded_same_session_capture_count": len(excluded),
        "holdout_files": holdout_files,
        "noise_data_dir": str(noise_root),
        "policy": "exclude exact frozen evaluation captures and all audio from their auto_capture_session_id",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-data-dir", type=Path, required=True)
    parser.add_argument("--holdout-windows", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = prepare_exclusions(args.noise_data_dir, args.holdout_windows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "frozen_holdout_count": data["frozen_holdout_count"],
        "frozen_capture_session_count": data["frozen_capture_session_count"],
        "excluded_same_session_capture_count": data["excluded_same_session_capture_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
