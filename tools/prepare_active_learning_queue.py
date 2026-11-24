"""Stage a frozen-test-safe, session-diverse active-learning review batch.

No annotations are created or accepted. The JSON is a worklist for the
existing SigID review UI, not a training dataset or pseudo-label manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from sigid.active_learning import rank_review_batch, protected_source_sha
from sigid.dataset import CaptureStore


def prepare(data_dir: Path, frozen_exclusions: Path, output: Path, limit: int = 100) -> dict:
    data_dir = data_dir.expanduser().resolve(strict=True)
    if not (data_dir / "dataset.sqlite3").is_file():
        raise ValueError("capture database does not exist")
    exclude_sha = protected_source_sha(frozen_exclusions)
    store = CaptureStore(data_dir / "dataset.sqlite3")
    result = rank_review_batch(
        store.list_captures(), store, excluded_sha=exclude_sha, limit=limit,
    )
    if not result["selected"]:
        raise ValueError("no eligible real receiver captures for prioritized review")
    result["frozen_exclusion_manifest_sha256"] = hashlib.sha256(
        frozen_exclusions.expanduser().resolve(strict=True).read_bytes()
    ).hexdigest()
    result["capture_index_sha256"] = hashlib.sha256(
        "|".join(sorted(row["capture_sha256"] for row in result["selected"])).encode()
    ).hexdigest()
    output = output.expanduser().resolve()
    if output.parent != data_dir:
        raise ValueError("active-learning worklist must live in the original capture-data directory")
    temp = output.with_name("." + output.name + ".tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temp.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--frozen-exclusions", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.data_dir / "active-learning-review-batch.json"
    result = prepare(args.data_dir, args.frozen_exclusions, output, args.limit)
    print(json.dumps({
        "selected_count": result["selected_count"],
        "selected_independent_sessions": result["selected_independent_sessions"],
        "selected_devices": result["selected_devices"],
        "skipped": result["skipped"],
        "review_worklist": str(output),
        "training_eligible": False,
    }, indent=2))


if __name__ == "__main__":
    main()
