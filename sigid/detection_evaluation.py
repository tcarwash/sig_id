"""Reproducible candidate recall and false-alarm evaluation on synthetic data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .detection import detect_tracks
from .dataset import CaptureStore
from .scenes import GENERATOR as SCENE_GENERATOR
from .spectrum import waterfall


NOISE_GENERATOR = "sig-id-noise"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _maximum_temporal_matches(
    candidates: list[dict[str, Any]],
    regions: list[dict[str, Any]],
    *,
    minimum_truth_coverage: float,
    minimum_candidate_coverage: float,
) -> list[tuple[int, int, float]]:
    adjacency: list[list[tuple[int, float]]] = []
    for candidate in candidates:
        candidate_start = int(candidate["sample_start"])
        candidate_end = candidate_start + int(candidate["sample_count"])
        candidate_duration = max(1, candidate_end - candidate_start)
        edges = []
        for region_index, region in enumerate(regions):
            truth_start = int(region["sample_start"])
            truth_end = truth_start + int(region["sample_count"])
            overlap = max(0, min(candidate_end, truth_end) - max(candidate_start, truth_start))
            truth_coverage = overlap / max(1, truth_end - truth_start)
            candidate_coverage = overlap / candidate_duration
            if truth_coverage >= minimum_truth_coverage and candidate_coverage >= minimum_candidate_coverage:
                edges.append((region_index, min(truth_coverage, candidate_coverage)))
        adjacency.append(sorted(edges, key=lambda item: item[1], reverse=True))

    matched_by_truth: dict[int, int] = {}

    def augment(candidate_index: int, visited: set[int]) -> bool:
        for region_index, _ in adjacency[candidate_index]:
            if region_index in visited:
                continue
            visited.add(region_index)
            if region_index not in matched_by_truth or augment(matched_by_truth[region_index], visited):
                matched_by_truth[region_index] = candidate_index
                return True
        return False

    for candidate_index in range(len(candidates)):
        augment(candidate_index, set())
    matches = []
    for region_index, candidate_index in matched_by_truth.items():
        coverage = next(score for index, score in adjacency[candidate_index] if index == region_index)
        matches.append((candidate_index, region_index, coverage))
    return sorted(matches)


def evaluate_synthetic_detector(
    store: CaptureStore,
    *,
    capture_ids: list[str] | None = None,
    fft_size: int = 512,
    hop_size: int = 128,
    max_rows: int = 4096,
    threshold_db: float = 18.0,
    minimum_truth_coverage: float = 0.30,
    minimum_candidate_coverage: float = 0.20,
) -> dict[str, Any]:
    """Score temporal component recall, candidate precision, and noise false alarms.

    Only imported deterministic mixed scenes and noise-only negatives enter the
    default benchmark. Source-region labels remain the recall denominator; one
    broad candidate cannot satisfy multiple simultaneous components.
    """
    if fft_size < 32 or fft_size & (fft_size - 1) or not 0 < hop_size <= fft_size or max_rows < 1:
        raise ValueError("invalid waterfall evaluation dimensions")
    if threshold_db <= 0 or not 0 < minimum_truth_coverage <= 1 or not 0 < minimum_candidate_coverage <= 1:
        raise ValueError("invalid detector thresholds or temporal matching coverage")
    captures = {capture["id"]: capture for capture in store.list_captures()}
    if capture_ids is None:
        selected = [
            capture for capture in captures.values()
            if capture["source_kind"] == "synthetic"
            and capture["source_metadata"].get("generator") in {SCENE_GENERATOR, NOISE_GENERATOR}
        ]
    else:
        if len(set(capture_ids)) != len(capture_ids):
            raise ValueError("capture_ids must not contain duplicates")
        try:
            selected = [captures[capture_id] for capture_id in capture_ids]
        except KeyError as exc:
            raise ValueError(f"unknown capture ID in evaluation set: {exc.args[0]}") from exc
    if not selected:
        raise ValueError("no synthetic mixed-scene or noise-only captures are available to evaluate")

    scene_count = noise_count = component_count = detected_component_count = 0
    total_candidates = false_candidates = 0
    total_duration_seconds = noise_duration_seconds = 0.0
    noise_false_candidates = 0
    capture_results = []
    for capture in selected:
        metadata = capture["source_metadata"]
        generator = metadata.get("generator")
        if capture["source_kind"] != "synthetic" or generator not in {SCENE_GENERATOR, NOISE_GENERATOR}:
            raise ValueError(f"capture {capture['id']} is not a supported synthetic evaluation example")
        path = Path(capture["audio_path"])
        if _sha256(path) != capture["sha256"]:
            raise ValueError(f"evaluation capture hash changed: {capture['id']}")
        duration = int(capture["sample_count"]) / int(capture["sample_rate"])
        total_duration_seconds += duration
        regions = store.regions(capture["id"])
        if generator == SCENE_GENERATOR:
            scene_count += 1
            component_ids = metadata.get("component_capture_ids")
            if not isinstance(component_ids, list) or len(regions) != len(component_ids):
                raise ValueError(f"mixed-scene capture {capture['id']} has inconsistent source regions")
            truth_regions = [region for region in regions if region["family"] is not None or region["method"] is not None]
            for region in truth_regions:
                if region["source"] != "synthetic" or region["review_state"] != "accepted":
                    raise ValueError(f"scene region {region['id']} is not accepted synthetic truth")
                evidence = region["evidence"]
                component_id = evidence.get("component_capture_id")
                if component_id not in component_ids:
                    raise ValueError(f"scene region {region['id']} is not tied to a listed component")
                source = store.capture(component_id)
                source_path = Path(source["audio_path"])
                if source["sha256"] != evidence.get("component_sha256") or _sha256(source_path) != source["sha256"]:
                    raise ValueError(f"scene region {region['id']} source capture hash changed")
                source_region = next(
                    (item for item in store.regions(component_id) if item["id"] == evidence.get("source_region_id")),
                    None,
                )
                if source_region is None or any(
                    region[field] != source_region[field]
                    for field in ("family", "method", "protocol", "variant")
                ):
                    raise ValueError(f"scene region {region['id']} disagrees with its source label")
        else:
            noise_count += 1
            noise_duration_seconds += duration
            if len(regions) != 1 or any(
                regions[0][field] is not None for field in ("family", "method", "protocol", "variant")
            ) or regions[0]["source"] != "synthetic" or regions[0]["review_state"] != "accepted":
                raise ValueError(f"noise capture {capture['id']} has contradictory labels")
            truth_regions = []

        component_count += len(truth_regions)
        candidates = []
        if capture["sample_count"] >= fft_size:
            spectrum = waterfall(path, fft_size=fft_size, hop_size=hop_size, max_rows=max_rows)
            candidates = detect_tracks(spectrum, threshold_db=threshold_db)
        matches = _maximum_temporal_matches(
            candidates,
            truth_regions,
            minimum_truth_coverage=minimum_truth_coverage,
            minimum_candidate_coverage=minimum_candidate_coverage,
        )
        detected_component_count += len(matches)
        total_candidates += len(candidates)
        false_in_capture = len(candidates) - len(matches)
        false_candidates += false_in_capture
        if generator == NOISE_GENERATOR:
            noise_false_candidates += false_in_capture
        capture_results.append({
            "capture_id": capture["id"],
            "sha256": capture["sha256"],
            "kind": "mixed_scene" if generator == SCENE_GENERATOR else "noise_only",
            "duration_seconds": duration,
            "ground_truth_components": len(truth_regions),
            "detected_components": len(matches),
            "missed_components": len(truth_regions) - len(matches),
            "candidate_count": len(candidates),
            "false_candidates": false_in_capture,
            "matches": [
                {
                    "candidate_id": candidates[candidate_index]["id"],
                    "region_id": truth_regions[region_index]["id"],
                    "temporal_coverage": coverage,
                }
                for candidate_index, region_index, coverage in matches
            ],
        })

    total_hours = total_duration_seconds / 3600.0
    noise_hours = noise_duration_seconds / 3600.0
    return {
        "schema_version": 1,
        "generator": "sig-id-evaluate-synthetic-detector",
        "detector": "interpretable spectral track grouping",
        "parameters": {
            "fft_size": fft_size,
            "hop_size": hop_size,
            "max_rows": max_rows,
            "threshold_db": threshold_db,
            "minimum_truth_coverage": minimum_truth_coverage,
            "minimum_candidate_coverage": minimum_candidate_coverage,
        },
        "captures": len(selected),
        "mixed_scenes": scene_count,
        "noise_only_captures": noise_count,
        "ground_truth_components": component_count,
        "detected_components": detected_component_count,
        "component_recall": detected_component_count / component_count if component_count else None,
        "candidate_count": total_candidates,
        "matched_candidates": detected_component_count,
        "false_candidates": false_candidates,
        "candidate_precision": detected_component_count / total_candidates if total_candidates else None,
        "false_candidates_per_hour": false_candidates / total_hours if total_hours else None,
        "noise_only_false_candidates": noise_false_candidates,
        "noise_only_hours": noise_hours,
        "noise_only_false_alarms_per_hour": noise_false_candidates / noise_hours if noise_hours else None,
        "capture_results": capture_results,
        "scope": "synthetic mixed-scene component timing and synthetic noise-only negatives; no real-radio performance claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate spectral candidate detection on imported synthetic scenes")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("SIG_ID_DATA_DIR", "~/.local/share/sig-id")).expanduser())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fft-size", type=int, default=512)
    parser.add_argument("--hop-size", type=int, default=128)
    parser.add_argument("--threshold-db", type=float, default=18.0)
    args = parser.parse_args()
    try:
        result = evaluate_synthetic_detector(
            CaptureStore(args.data_dir / "dataset.sqlite3"),
            fft_size=args.fft_size,
            hop_size=args.hop_size,
            threshold_db=args.threshold_db,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "captures", "mixed_scenes", "ground_truth_components", "component_recall",
        "candidate_precision", "false_candidates_per_hour", "noise_only_false_alarms_per_hour",
    )}, indent=2))


if __name__ == "__main__":
    main()
