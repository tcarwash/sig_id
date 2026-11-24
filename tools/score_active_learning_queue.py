"""Screen unverified receiver clips with CV and spectral tracks to rank human review.

This is a proposal-only active learning pass: no model output is accepted as
ground truth, no captures or annotations are modified, and frozen evaluation
sessions are excluded *before* any audio/model inference takes place.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from sigid.active_learning import protected_source_sha, rank_review_batch
from sigid.dataset import CaptureStore


def matched_track_count(cv: list[dict[str, Any]], tracks: list[dict[str, Any]], *,
                        frequency_slack_hz: float = 40.0) -> int:
    """Count spectral tracks overlapping a CV time span and frequency range.

    This is a weak *proposal agreement* test, deliberately not box IoU:
    narrow tone tracks may sit inside much wider modulation bounding boxes.
    """
    matched = 0
    for track in tracks:
        center = float(track["center_hz"])
        start = int(track["sample_start"])
        end = start + int(track["sample_count"])
        if any(
            max(start, int(box["sample_start"])) <
            min(end, int(box["sample_start"]) + int(box["sample_count"]))
            and float(box["frequency_low_hz"]) - frequency_slack_hz <= center <=
                float(box["frequency_high_hz"]) + frequency_slack_hz
            for box in cv
        ):
            matched += 1
    return matched


def screen_capture(capture: dict[str, Any], detector: Any) -> dict[str, Any]:
    from sigid.detection import detect_tracks
    from sigid.spectrum import waterfall

    # Both detectors are *untrusted* proposal generators. This function is
    # intentionally read-only with respect to CaptureStore.
    path = Path(capture["audio_path"])
    cv = detector.detect_capture(path)
    spectrum = waterfall(path, fft_size=4096, hop_size=512, max_rows=320)
    tracks = detect_tracks(spectrum, threshold_db=18.0)
    # This filter limits weak peak fragments for ranking only. It neither
    # validates silence nor establishes that any candidate is an RF signal.
    tracks = [
        track for track in tracks
        if track["peak_snr_db"] >= 18.0 and track["sample_count"] >= int(0.15 * capture["sample_rate"])
    ]
    return {
        "cv_count": len(cv),
        "track_count": len(tracks),
        "matched_track_count": matched_track_count(cv, tracks),
        "cv_model_sha256": detector.model_sha256,
        "proposal_only": True,
    }


def prepare_screened_batch(
    data_dir: Path, exclusions: Path, model: Path, *, limit: int = 100,
    scan_limit: int = 200, output: Path | None = None, evidence_output: Path | None = None,
) -> dict[str, Any]:
    from sigid.cv_detector import YoloSignalDetector

    data_dir = data_dir.expanduser().resolve(strict=True)
    exclusions = exclusions.expanduser().resolve(strict=True)
    model = model.expanduser().resolve(strict=True)
    output = (output or (data_dir / "active-learning-review-batch.json")).expanduser().resolve()
    if output.parent != data_dir:
        raise ValueError("worklist output must stay in the original data directory")
    if not 1 <= limit <= 10000 or not 1 <= scan_limit <= 10000:
        raise ValueError("limits must be 1..10000")
    store = CaptureStore(data_dir / "dataset.sqlite3")
    captures = store.list_captures()
    excluded = protected_source_sha(exclusions)
    # Mandatory frozen session filter, not just SHA matching.
    pool = rank_review_batch(captures, store, excluded_sha=excluded, limit=10000)
    by_id = {capture["id"]: capture for capture in captures}
    detector = YoloSignalDetector(model)
    evidence: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    to_scan = pool["selected"][:scan_limit]
    for index, row in enumerate(to_scan, 1):
        capture = by_id[row["capture_id"]]
        assert capture["sha256"] not in excluded
        try:
            evidence[capture["sha256"]] = screen_capture(capture, detector)
        except (OSError, ValueError, RuntimeError) as exc:
            # Failed scoring leaves a capture unscored; never label it noise.
            errors[capture["sha256"]] = type(exc).__name__
        if index % 25 == 0 or index == len(to_scan):
            print(f"screened {index}/{len(to_scan)} (failed {len(errors)})", flush=True)

    batch = rank_review_batch(
        captures, store, excluded_sha=excluded, limit=limit,
        proposal_evidence_by_sha=evidence,
    )
    batch["selection_policy"] = "session-round-robin-cv-vs-spectral-track-unverified-v2"
    batch["frozen_exclusion_manifest_sha256"] = hashlib.sha256(exclusions.read_bytes()).hexdigest()
    batch["capture_index_sha256"] = hashlib.sha256(
        "|".join(sorted(row["capture_sha256"] for row in batch["selected"])).encode()
    ).hexdigest()
    batch["proposal_screening"] = {
        "model_sha256": detector.model_sha256,
        "screened_count": len(evidence),
        "failed_count": len(errors),
        "scan_limit": scan_limit,
        "candidate_pool_count": pool["candidate_count"],
        "proposal_categories": dict(Counter(
            reason for row in batch["selected"] for reason in row["priority_reasons"]
            if reason in {
                "spectral-track-only-proposal", "cv-only-proposal",
                "proposals-disagree-on-location", "cross-detector-proposals",
                "possible-negative-control-unverified",
            }
        )),
        "proposals_are_verified_labels": False,
    }
    assert batch["training_eligible"] is False
    assert all(row["capture_sha256"] not in excluded and
               row["label_trust"] == "unverified" for row in batch["selected"])
    # Preserve detailed evidence privately; neither WAVs nor predictions are
    # committed to the repository.
    if evidence_output is not None:
        p = evidence_output.expanduser().resolve()
        if p == output:
            raise ValueError("evidence output must differ from the review worklist")
        p.parent.mkdir(parents=True, exist_ok=True)
        temp = p.with_name("." + p.name + ".tmp")
        temp.write_text(json.dumps({
            "model_sha256": detector.model_sha256,
            "unverified_by_waveform_sha256": evidence,
            "scoring_errors_by_waveform_sha256": errors,
            "label_trust": "unverified",
        }, indent=2, sort_keys=True) + "\n")
        temp.replace(p)
    temp = output.with_name("." + output.name + ".tmp")
    temp.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n")
    temp.replace(output)
    return {
        "selected_count": batch["selected_count"],
        "selected_independent_sessions": batch["selected_independent_sessions"],
        "proposal_screening": batch["proposal_screening"],
        "training_eligible": False,
        "worklist": str(output),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--frozen-exclusions", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True,
                   help="verified metadata-backed baseline CV model; recommendations only")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--scan-limit", type=int, default=200)
    p.add_argument("--output", type=Path)
    p.add_argument("--evidence-output", type=Path)
    args = p.parse_args()
    print(json.dumps(prepare_screened_batch(
        args.data_dir, args.frozen_exclusions, args.model, limit=args.limit,
        scan_limit=args.scan_limit, output=args.output, evidence_output=args.evidence_output,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
