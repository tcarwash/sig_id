"""Offline crowded-audio benchmark with optional hand-reviewed reference bands.

Example:
    sig-id-benchmark-crowded recording.wav --cv-json cv-proposals.json \
        --reference-json reviewed-bands.json --output run.json

Without reference annotations this is a measurement report, NOT an accuracy
evaluation. The original WAV and manual labels are not uploaded by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import wave

import numpy as np

from .frequency_refinement import MAX_REFINEMENT_SECONDS, refine_frequency_region


def _boxes(value: Any) -> list[dict[str, Any]]:
    boxes = value.get("detections", value.get("regions", [])) if isinstance(value, dict) else value
    if not isinstance(boxes, list) or not all(isinstance(box, dict) for box in boxes):
        raise ValueError("expected a list or {detections: [...]} of signal boxes")
    return boxes


def _bounds(box: dict[str, Any], rate: int) -> tuple[int, int, float, float]:
    start = int(box["sample_start"]) if "sample_start" in box else round(float(box["start_seconds"]) * rate)
    count = int(box["sample_count"]) if "sample_count" in box else round(
        (float(box["end_seconds"]) - float(box["start_seconds"])) * rate
    )
    low = float(box["frequency_low_hz"])
    high = float(box["frequency_high_hz"])
    if start < 0 or count <= 0 or not np.isfinite((low, high)).all() or low < 0 or high <= low:
        raise ValueError("invalid benchmark box geometry")
    return start, count, low, high


def _audio_excerpt(wav: wave.Wave_read, start: int, count: int) -> np.ndarray:
    rate = wav.getframerate()
    bounded = min(count, round(MAX_REFINEMENT_SECONDS * rate))
    offset = start + max(0, (count - bounded) // 2)
    available = max(0, min(wav.getnframes() - offset, bounded))
    if available <= 0:
        return np.zeros(0, dtype=np.float64)
    wav.setpos(offset)
    frames = np.frombuffer(wav.readframes(available), dtype="<i2")
    return frames.reshape(-1, wav.getnchannels()).astype(np.float64).mean(axis=1) / 32768.0


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _matches_reference(
    proposals: list[dict[str, Any]],
    references: list[dict[str, Any]],
    rate: int,
) -> dict[str, Any]:
    # Match one-to-one with both time overlap and frequency IoU. Using only
    # frequency would incorrectly equate different FT8 slots on the same tone.
    ranked: list[tuple[float, int, int]] = []
    for i, candidate in enumerate(proposals):
        a, n, lo, hi = _bounds(candidate, rate)
        for j, reference in enumerate(references):
            b, m, rlo, rhi = _bounds(reference, rate)
            time_overlap = _overlap((a, a + n), (b, b + m))
            freq_intersection = _overlap((lo, hi), (rlo, rhi))
            freq_union = hi - lo + rhi - rlo - freq_intersection
            if time_overlap / min(n, m) >= 0.5 and freq_union > 0:
                score = freq_intersection / freq_union
                if score >= 0.25:
                    ranked.append((score, i, j))
    used_proposals: set[int] = set()
    used_references: set[int] = set()
    for _, i, j in sorted(ranked, reverse=True):
        if i not in used_proposals and j not in used_references:
            used_proposals.add(i)
            used_references.add(j)
    tp = len(used_proposals)
    return {
        "matched": tp,
        "unmatched_proposals": len(proposals) - tp,
        "missed_references": len(references) - tp,
        "precision": tp / len(proposals) if proposals else None,
        "recall": tp / len(references) if references else None,
        "matching_rule": ">=0.5 of shorter time interval, >=0.25 frequency IoU; one-to-one",
        "scope": "Only supplied CV regions and reference annotations; not end-to-end CV recall.",
    }


def evaluate_crowded_wav(
    wav_path: str | Path,
    cv_boxes: list[dict[str, Any]] | dict[str, Any],
    *,
    references: list[dict[str, Any]] | dict[str, Any] | None = None,
    max_regions: int = 32,
) -> dict[str, Any]:
    """Run a deterministic, read-only benchmark on a local PCM16 WAV."""
    source = Path(wav_path)
    boxes = _boxes(cv_boxes)
    if max_regions < 1 or len(boxes) > max_regions:
        raise ValueError(f"the benchmark allows 1–{max_regions} boxes")
    if references is not None:
        reviewed = _boxes(references)
    else:
        reviewed = None

    sha = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    with wave.open(str(source), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("benchmark requires uncompressed 16-bit PCM WAV")
        rate = wav.getframerate()
        count = wav.getnframes()
        results: list[dict[str, Any]] = []
        proposed: list[dict[str, Any]] = []
        original: list[dict[str, Any]] = []
        for index, box in enumerate(boxes):
            start, length, low, high = _bounds(box, rate)
            if start + length > count:
                raise ValueError(f"box {index} exceeds WAV sample range")
            audio = _audio_excerpt(wav, start, length)
            refinement = refine_frequency_region(audio, rate, low, high)
            parent = {
                "sample_start": start, "sample_count": length,
                "frequency_low_hz": low, "frequency_high_hz": high,
            }
            original.append(parent)
            proposals = [
                {**parent, **{"frequency_low_hz": sub["frequency_low_hz"], "frequency_high_hz": sub["frequency_high_hz"]}}
                for sub in refinement["subbands"]
            ] if refinement["split_suggested"] else [parent]
            proposed.extend(proposals)
            results.append({
                "box_index": index, "original": parent,
                "refinement": refinement,
                "effective_boxes": proposals,
            })
    return {
        "benchmark_version": "crowded-audio-v1",
        "source_name": source.name,
        "wav_sha256": sha.hexdigest(),
        "sample_rate": rate,
        "duration_seconds": count / rate,
        "reference_provided": reviewed is not None,
        "regions_checked": len(boxes),
        "split_suggestions": sum(row["refinement"]["split_suggested"] for row in results),
        "proposed_boxes": len(proposed),
        "results": results,
        "metrics": (
            {
                "original": _matches_reference(original, reviewed, rate),
                "refined": _matches_reference(proposed, reviewed, rate),
            }
            if reviewed is not None else None
        ),
        "caveat": "No protocol is verified. Metrics are meaningful only for independently reviewed references.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", help="Local uncompressed PCM16 WAV")
    parser.add_argument("--cv-json", required=True, help="CV result with detections[] or array of bounding regions")
    parser.add_argument("--reference-json", help="Optional independently reviewed reference boxes")
    parser.add_argument("--output", help="Write JSON metrics here (default: standard output)")
    args = parser.parse_args()
    cv = json.loads(Path(args.cv_json).read_text(encoding="utf-8"))
    references = json.loads(Path(args.reference_json).read_text(encoding="utf-8")) if args.reference_json else None
    result = evaluate_crowded_wav(args.wav, cv, references=references)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")
        print(f"Saved benchmark to {args.output}")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
