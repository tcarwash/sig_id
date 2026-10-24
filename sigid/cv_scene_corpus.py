"""Create deterministic burst-in-noise scenes for optional CV detector training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .cv_detector import MAX_FREQUENCY_HZ, WINDOW_SECONDS
from .dataset import CaptureStore
from .scenes import create_mixed_scene, import_mixed_scene
from .synthetic import generate_noise_wav
from .synthetic_dataset import import_synthetic_wav


SIGNAL_GENERATORS = {
    "sig-id-2fsk", "sig-id-mfsk", "sig-id-bpsk", "sig-id-qpsk",
    "sig-id-ook", "sig-id-ask", "sig-id-ppm",
}
SCHEMA_VERSION = 1
MAX_NOISE_ONLY_EXAMPLES = 5000


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _derived_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _scene_groups(captures: list[dict[str, Any]], seed: int, overlap_rate: float) -> list[list[dict[str, Any]]]:
    by_rate: dict[int, list[dict[str, Any]]] = {}
    for capture in captures:
        by_rate.setdefault(int(capture["sample_rate"]), []).append(capture)
    groups = []
    for sample_rate, sources in sorted(by_rate.items()):
        rng = np.random.default_rng(_derived_seed(seed, f"sample-rate:{sample_rate}"))
        ordered = sorted(sources, key=lambda item: item["sha256"])
        order = rng.permutation(len(ordered)).tolist()
        pending = [ordered[index] for index in order]
        while pending:
            first = pending.pop()
            if pending and rng.random() < overlap_rate:
                groups.append([first, pending.pop()])
            else:
                groups.append([first])
    return groups


def _signal_recipe(
    capture: dict[str, Any], region: dict[str, Any], *, rng: np.random.Generator,
    scene_sample_count: int, sample_rate: int, scene_start_sample: int | None = None,
    requested_sample_count: int | None = None,
) -> dict[str, Any]:
    if requested_sample_count is None:
        max_duration = min(4.0, capture["sample_count"] / sample_rate)
        min_duration = min(0.75, max_duration)
        duration = float(rng.uniform(min_duration, max_duration)) if max_duration > min_duration else max_duration
        count = max(4096, min(capture["sample_count"], int(round(duration * sample_rate))))
    else:
        count = requested_sample_count
    if count > capture["sample_count"]:
        count = capture["sample_count"]
    source_start = int(rng.integers(0, capture["sample_count"] - count + 1))
    maximum_start = scene_sample_count - count
    if scene_start_sample is None:
        scene_start_sample = int(rng.integers(0, maximum_start + 1))

    low, high = region["frequency_low_hz"], region["frequency_high_hz"]
    offsets = [offset for offset in (-200.0, -100.0, 0.0, 100.0, 200.0)
               if low + offset >= 0 and high + offset <= sample_rate / 2]
    offset = float(rng.choice(offsets or [0.0]))
    return {
        "capture_id": capture["id"],
        "source_start_sample": source_start,
        "sample_count": count,
        "scene_start_sample": scene_start_sample,
        "gain": float(rng.uniform(0.3, 0.85)),
        "frequency_offset_hz": offset,
        "drift_hz_per_second": float(rng.choice([-3.0, -1.5, 0.0, 1.5, 3.0])),
    }


def _overlapping_start(
    first: dict[str, Any], second_count: int, *, scene_sample_count: int,
    rng: np.random.Generator,
) -> int:
    first_start = first["scene_start_sample"]
    first_end = first_start + first["sample_count"]
    lower = max(0, first_start - second_count + 1)
    upper = min(scene_sample_count - second_count, first_end - 1)
    if lower > upper:
        return int(rng.integers(0, scene_sample_count - second_count + 1))
    return int(rng.integers(lower, upper + 1))


def _ensure_synthetic_review(store: CaptureStore, capture: dict[str, Any], note: str) -> None:
    reviews = store.scene_coverages(capture["id"])
    if any(
        review["task"] == "detection" and review["state"] == "complete"
        and not review["is_stale"] and review["sample_start"] == 0
        and review["sample_count"] == capture["sample_count"]
        and review["frequency_low_hz"] in (None, 0.0)
        and review["frequency_high_hz"] in (None, MAX_FREQUENCY_HZ)
        and not review["ignore_regions"] and not review["uncertain_regions"]
        for review in reviews
    ):
        return
    store.create_scene_coverage(
        capture["id"], sample_start=0, sample_count=capture["sample_count"],
        frequency_low_hz=0.0, frequency_high_hz=MAX_FREQUENCY_HZ,
        task="detection", taxonomy_scope=["signal"], state="complete", notes=note,
    )


def build_cv_synthetic_scenes(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 1,
    overlap_rate: float = 0.25,
    noise_only_examples: int = 0,
) -> dict[str, Any]:
    """Generate reviewed, source-held-out-ready signal bursts over known noise.

    Inputs must be accepted, manifest-backed synthetic modulation captures.
    Each source is used in at most one scene, and each scene gets its own
    independently seeded 10-second noise capture so lineage groups stay distinct.
    Optional noise-only captures add independent reviewed negative windows.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if not isinstance(overlap_rate, (int, float)) or isinstance(overlap_rate, bool) or not 0 <= overlap_rate <= 1:
        raise ValueError("overlap_rate must be between 0 and 1")
    if (
        not isinstance(noise_only_examples, int) or isinstance(noise_only_examples, bool)
        or not 0 <= noise_only_examples <= MAX_NOISE_ONLY_EXAMPLES
    ):
        raise ValueError(f"noise_only_examples must be between 0 and {MAX_NOISE_ONLY_EXAMPLES}")
    root = Path(data_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    store = CaptureStore(root / "dataset.sqlite3")
    index_path = output / "cv-scenes.json"
    existing_index = None
    if index_path.exists():
        existing_index = json.loads(index_path.read_text(encoding="utf-8"))
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")

    captures = store.list_captures()
    signals = []
    seen_source_hashes: set[str] = set()
    for capture in captures:
        metadata = capture["source_metadata"]
        if (
            capture["source_kind"] != "synthetic"
            or metadata.get("generator") not in SIGNAL_GENERATORS
            or capture["parent_capture_id"] is not None
            or capture["sample_rate"] / 2 < MAX_FREQUENCY_HZ
            or capture["sha256"] in seen_source_hashes
        ):
            continue
        regions = store.regions(capture["id"])
        if len(regions) != 1:
            continue
        region = regions[0]
        if (
            region["sample_start"] != 0 or region["sample_count"] != capture["sample_count"]
            or region["review_state"] != "accepted" or region["source"] != "synthetic"
            or region["frequency_low_hz"] is None or region["frequency_high_hz"] is None
        ):
            continue
        signals.append((capture, region))
        seen_source_hashes.add(capture["sha256"])
    if len(signals) < 3:
        raise ValueError("CV scene generation requires at least three accepted synthetic signal sources with frequency bounds")
    source_pool_sha256 = hashlib.sha256("|".join(sorted(seen_source_hashes)).encode()).hexdigest()
    config = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "overlap_rate": float(overlap_rate),
        "window_seconds": WINDOW_SECONDS,
        "signal_generators": sorted(SIGNAL_GENERATORS),
        "source_pool_sha256": source_pool_sha256,
    }
    if noise_only_examples:
        config["noise_only_examples"] = noise_only_examples
    if existing_index is not None:
        if existing_index.get("config") != config:
            raise ValueError("existing scene index uses a different recipe or source pool; choose another output directory")
        index = existing_index
    else:
        output.mkdir(parents=True, exist_ok=True)
        index = {"config": config, "records": []}
        _atomic_json(index_path, index)

    for capture, _ in signals:
        _ensure_synthetic_review(
            store, capture,
            "Controlled synthetic modulation capture; validated generator manifest provides complete signal truth.",
        )

    for capture in captures:
        if (
            capture["source_kind"] == "synthetic"
            and capture["sample_rate"] / 2 >= MAX_FREQUENCY_HZ
            and capture["source_metadata"].get("generator") == "sig-id-noise"
        ):
            _ensure_synthetic_review(
                store, capture,
                "Controlled noise-only capture; validated generator manifest provides complete no-signal truth.",
            )

    noise_only_records = list(index.get("noise_only_records", []))
    if noise_only_examples:
        records_by_index = {int(record["index"]): record for record in noise_only_records}
        for item_index in range(noise_only_examples):
            noise_seed = _derived_seed(seed, f"noise-only:{item_index}")
            waveform = output / "noise-only" / f"{item_index:05d}.wav"
            manifest_path = waveform.with_suffix(waveform.suffix + ".json")
            if not waveform.is_file() or not manifest_path.is_file():
                generate_noise_wav(
                    waveform, seed=noise_seed, sample_rate=12_000,
                    sample_count=12_000 * WINDOW_SECONDS, amplitude=0.35,
                )
            digest = hashlib.sha256(waveform.read_bytes()).hexdigest()
            manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            previous_record = records_by_index.get(item_index)
            if previous_record and (
                previous_record.get("sha256") != digest
                or previous_record.get("manifest_sha256") != manifest_digest
            ):
                raise ValueError(f"noise-only source {item_index} no longer matches its generated index record")
            noise_capture = store.capture_by_sha256(digest)
            if noise_capture is None:
                noise_capture = import_synthetic_wav(
                    waveform, manifest_path, store=store, capture_dir=root / "captures",
                )["capture"]
            if (
                noise_capture["source_kind"] != "synthetic"
                or noise_capture["source_metadata"].get("generator") != "sig-id-noise"
                or noise_capture["sha256"] != digest
                or int(noise_capture["sample_rate"]) != 12_000
                or int(noise_capture["sample_count"]) != 12_000 * WINDOW_SECONDS
            ):
                raise ValueError("noise-only corpus source hash is registered with incompatible provenance")
            _ensure_synthetic_review(
                store, noise_capture,
                "Independent controlled noise-only window; validated generator manifest provides complete no-signal truth.",
            )
            records_by_index[item_index] = {
                "index": item_index,
                "capture_id": noise_capture["id"],
                "sha256": digest,
                "manifest_sha256": manifest_digest,
                "seed": noise_seed,
                "sample_rate": 12_000,
                "duration_seconds": WINDOW_SECONDS,
            }
            index["noise_only_records"] = [records_by_index[key] for key in sorted(records_by_index)]
            _atomic_json(index_path, index)
        noise_only_records = index["noise_only_records"]

    output_records = list(index.get("records", []))
    processed = {
        digest
        for record in output_records
        for digest in record.get("source_sha256", [])
    }
    scenes_created = 0
    for sources in _scene_groups([capture for capture, _ in signals], seed, float(overlap_rate)):
        source_regions = {capture["id"]: region for capture, region in signals if capture["id"] in {item["id"] for item in sources}}
        source_hashes = sorted(item["sha256"] for item in sources)
        if all(digest in processed for digest in source_hashes):
            continue
        sample_rate = int(sources[0]["sample_rate"])
        scene_sample_count = sample_rate * WINDOW_SECONDS
        recipe_key = hashlib.sha256(f"{seed}:{'|'.join(source_hashes)}".encode()).hexdigest()
        rng = np.random.default_rng(_derived_seed(seed, recipe_key))
        signal_recipes: list[dict[str, Any]] = []
        first = _signal_recipe(
            sources[0], source_regions[sources[0]["id"]], rng=rng,
            scene_sample_count=scene_sample_count, sample_rate=sample_rate,
        )
        signal_recipes.append(first)
        if len(sources) > 1:
            second_region = source_regions[sources[1]["id"]]
            second_duration = min(4.0, sources[1]["sample_count"] / sample_rate)
            second_min_duration = min(0.75, second_duration)
            second_count = max(4096, min(
                sources[1]["sample_count"],
                int(round(float(rng.uniform(second_min_duration, second_duration)) * sample_rate)),
            ))
            second_start = _overlapping_start(
                first, second_count, scene_sample_count=scene_sample_count, rng=rng,
            )
            # Reuse the chosen interval while still drawing an independent source offset and impairment recipe.
            second = _signal_recipe(
                sources[1], second_region, rng=rng, scene_sample_count=scene_sample_count,
                sample_rate=sample_rate, scene_start_sample=second_start,
                requested_sample_count=second_count,
            )
            signal_recipes.append(second)

        noise_seed = _derived_seed(seed, f"noise:{recipe_key}")
        noise_waveform = output / "noise" / f"{recipe_key}.wav"
        noise_waveform.parent.mkdir(parents=True, exist_ok=True)
        noise_manifest = noise_waveform.with_suffix(noise_waveform.suffix + ".json")
        if not noise_waveform.is_file() or not noise_manifest.is_file():
            generate_noise_wav(
                noise_waveform, seed=noise_seed, sample_rate=sample_rate,
                sample_count=scene_sample_count, amplitude=0.35,
            )
        noise_capture = store.capture_by_sha256(hashlib.sha256(noise_waveform.read_bytes()).hexdigest())
        if noise_capture is None:
            noise_capture = import_synthetic_wav(
                noise_waveform, noise_manifest, store=store, capture_dir=root / "captures",
            )["capture"]
        _ensure_synthetic_review(
            store, noise_capture,
            "Generated noise background; validated generator manifest provides complete no-signal truth.",
        )
        scene_recipe = [*signal_recipes, {
            "capture_id": noise_capture["id"],
            "sample_count": scene_sample_count,
            "scene_start_sample": 0,
            "gain": float(rng.uniform(0.25, 0.55)),
        }]
        scene_waveform = output / "scenes" / f"{recipe_key}.wav"
        scene_waveform.parent.mkdir(parents=True, exist_ok=True)
        scene_manifest = create_mixed_scene(
            scene_waveform, scene_recipe, store=store, sample_count=scene_sample_count,
        )
        scene_capture = store.capture_by_sha256(scene_manifest["waveform_sha256"])
        if scene_capture is None:
            scene_capture = import_mixed_scene(
                scene_waveform, scene_waveform.with_suffix(scene_waveform.suffix + ".json"),
                store=store, capture_dir=root / "captures",
            )["capture"]
        _ensure_synthetic_review(
            store, scene_capture,
            "Deterministic synthetic scene; validated component manifests provide complete signal and noise truth.",
        )
        record = {
            "source_capture_ids": [item["id"] for item in sources],
            "source_sha256": source_hashes,
            "noise_capture_id": noise_capture["id"],
            "noise_seed": noise_seed,
            "scene_capture_id": scene_capture["id"],
            "scene_sha256": scene_capture["sha256"],
            "component_count": len(signal_recipes) + 1,
            "scene_seconds": WINDOW_SECONDS,
        }
        output_records.append(record)
        processed.update(source_hashes)
        scenes_created += 1
        index["records"] = output_records
        _atomic_json(index_path, index)

    return {
        "index": str(index_path),
        "source_count": len(signals),
        "scene_count": len(output_records),
        "scenes_created_this_run": scenes_created,
        "noise_only_windows": len(output_records),
        "additional_noise_only_windows": len(noise_only_records),
        "lineage_split_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("~/.local/share/sig-id").expanduser())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overlap-rate", type=float, default=0.25)
    parser.add_argument(
        "--noise-only-examples", type=int, default=0,
        help=f"add this many independent reviewed 10-second uniform-noise captures (max {MAX_NOISE_ONLY_EXAMPLES})",
    )
    args = parser.parse_args()
    try:
        report = build_cv_synthetic_scenes(
            args.data_dir, args.output_dir, seed=args.seed, overlap_rate=args.overlap_rate,
            noise_only_examples=args.noise_only_examples,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
