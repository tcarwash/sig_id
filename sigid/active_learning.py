"""Session-diverse annotation candidate ranking; does not create training labels.

Only operator-accepted and complete non-stale detection coverage can form
training examples. These ranks and any model proposals are advisory and are
never used as target labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
from typing import Any


def protected_source_sha(path: str | Path) -> set[str]:
    """Load a frozen holdout exclusion manifest without modifying it."""
    manifest = json.loads(Path(path).expanduser().resolve(strict=True).read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported frozen source exclusion manifest")
    excluded = manifest.get("excluded_capture_sha256")
    frozen = manifest.get("frozen_holdout_capture_sha256")
    if (not isinstance(excluded, list) or not isinstance(frozen, list)
            or not frozen or not set(frozen).issubset(excluded)
            or any(not isinstance(digest, str) or len(digest) != 64
                   or any(char not in "0123456789abcdef" for char in digest)
                   for digest in excluded)):
        raise ValueError("invalid or incomplete frozen source exclusion manifest")
    return set(excluded)


def _session(capture: dict[str, Any]) -> str:
    meta = capture["source_metadata"]
    return str(meta.get("auto_capture_session_id") or capture.get("parent_capture_id") or capture["sha256"])


def _priority(capture: dict[str, Any], regions: list[dict[str, Any]]) -> tuple[int, list[str]]:
    meta = capture["source_metadata"]
    score = 0
    reasons = []
    if not regions:
        score += 5
        reasons.append("unannotated")
    else:
        if any(item.get("review_state") == "pending" for item in regions):
            score += 5
            reasons.append("unresolved-region")
        if any(item.get("review_state") == "accepted" and not item.get("mode_id") for item in regions):
            score += 3
            reasons.append("confirmed-signal-unknown-mode")
    summary = meta.get("detection_summary") or {}
    if isinstance(summary, dict):
        count = summary.get("cv_detection_count")
        if isinstance(count, int) and count > 0:
            score += 3
            reasons.append("model-suggested-presence")
        methods = summary.get("classifier_methods") or []
        if isinstance(methods, list) and len(set(str(x) for x in methods)) > 1:
            score += 2
            reasons.append("mode-disagreement")
    trigger = meta.get("review_trigger")
    if trigger == "cv_signal":
        score += 2
        reasons.append("candidate-signal")
    if trigger == "periodic_noise":
        score += 2
        reasons.append("negative-control-candidate")
    if trigger == "continuous_recording":
        score += 1
        reasons.append("continuous-receiver")
    # Stable jitter avoids choosing identical temporal offsets from one session,
    # without depending on global random state or mutable server time.
    score += int(hashlib.sha256(capture["sha256"].encode()).hexdigest()[:2], 16) % 3
    return score, reasons


def rank_review_batch(
    captures: list[dict[str, Any]], store: Any, *,
    excluded_sha: set[str], limit: int = 100,
    include_reviewed: bool = False,
) -> dict[str, Any]:
    """Deterministically balance candidates by independent source session.

    Does not infer a confirmed negative from silence, or a confirmed mode from
    model suggestions. Protected frozen sources and their whole sessions never
    enter the batch, even if some same-session captures have different hashes.
    """
    if limit < 1 or limit > 10000:
        raise ValueError("review limit must be between 1 and 10000")
    protected_sessions = {_session(item) for item in captures if item["sha256"] in excluded_sha}
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    seen_sha = set()
    for capture in captures:
        meta = capture["source_metadata"]
        session = _session(capture)
        if session in protected_sessions or capture["sha256"] in excluded_sha:
            skipped["frozen-session"] += 1
            continue
        if capture["sha256"] in seen_sha:
            skipped["duplicate-source-sha"] += 1
            continue
        if capture["source_kind"] not in ("live", "wav") or meta.get("exclude_training") is True:
            skipped["non-receiver-or-excluded"] += 1
            continue
        if meta.get("auto_review_queue") is not True:
            skipped["not-in-live-review-queue"] += 1
            continue
        device = str(meta.get("device_name") or "")
        if "sigid_signal_lab" in device.casefold() or "signal_lab" in str(meta.get("generator", "")).casefold():
            skipped["signal-lab-loopback"] += 1
            continue
        sr = int(capture["sample_rate"])
        n = int(capture["sample_count"])
        if sr <= 0 or n < sr * 8 or n > sr * 12:
            skipped["non-10s-window"] += 1
            continue
        seen_sha.add(capture["sha256"])
        regions = store.regions(capture["id"])
        if not include_reviewed:
            reviews = store.scene_coverages(capture["id"])
            # Only a non-stale COMPLETE full-window detection review counts
            # as fully reviewed, not a click to mark reviewed.
            done = any(review["task"] == "detection" and review["state"] == "complete"
                       and not review["is_stale"] and review["sample_start"] <= 0
                       and review["sample_start"] + review["sample_count"] >= n
                       and (review["frequency_low_hz"] is None
                            or (review["frequency_low_hz"] <= 0
                                and review["frequency_high_hz"] >= 3000))
                       for review in reviews)
            if done:
                skipped["completed-detection-review"] += 1
                continue
            if meta.get("auto_review_queue_reviewed_utc"):
                skipped["marked-reviewed-without-coverage"] += 1
                continue
        score, reasons = _priority(capture, regions)
        sessions[session].append({
            "capture_id": capture["id"],
            "capture_sha256": capture["sha256"],
            "source_session_sha256": hashlib.sha256(session.encode()).hexdigest(),
            "device": device or "unknown",
            "sample_count": n, "sample_rate": sr,
            "priority_score": score,
            "priority_reasons": reasons,
            "annotation_status": "unannotated" if not regions else "requires-review",
            "label_trust": "unverified",
        })
    for items in sessions.values():
        items.sort(key=lambda item: (-item["priority_score"], item["capture_sha256"]))
    # Round robin sessions, then promote receiver diversity within each round.
    ordered = sorted(sessions, key=lambda sess: (
        -max(row["priority_score"] for row in sessions[sess]), hashlib.sha256(sess.encode()).hexdigest()))
    pool = {session: deque(sessions[session]) for session in ordered}
    output = []
    while len(output) < limit and any(pool.values()):
        used_devices: set[str] = set()
        for session in sorted(ordered, key=lambda key: (
                pool[key][0]["device"] in used_devices if pool[key] else True,
                -pool[key][0]["priority_score"] if pool[key] else 0,
                hashlib.sha256(key.encode()).hexdigest())):
            if not pool[session]:
                continue
            item = pool[session].popleft()
            output.append(item)
            used_devices.add(item["device"])
            if len(output) >= limit:
                break
    return {
        "schema_version": 1,
        "selection_policy": "session-round-robin-model-proposals-untrusted-v1",
        "review_limit": limit,
        "candidate_count": sum(len(items) for items in sessions.values()),
        "selected_count": len(output),
        "selected_independent_sessions": len({item["source_session_sha256"] for item in output}),
        "selected_devices": dict(Counter(item["device"] for item in output)),
        "skipped": dict(skipped),
        "frozen_session_count": len(protected_sessions),
        "selected": output,
        "training_eligible": False,
        "note": "Review suggestions only; never use these rows as training truth before human-accepted fresh coverage.",
    }
