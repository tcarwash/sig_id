"""Train data-driven family and per-family mode classifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np

from .classifier import (
    CLASSIFIER_WINDOW_SECONDS, DEFAULT_FEATURE_VERSION, FEATURE_COUNT_BY_VERSION,
    ROI_FEATURE_VERSIONS, fit_softmax, predict_probabilities,
)
from .cv_dataset import _review_covers_window
from .cv_detector import WINDOW_SECONDS
from .dataset import CaptureStore
from .modes import MODE_REGISTRY
from .ood_corpus import load_ood_corpus
from .splits import FAMILY_BY_MODE, METHOD_BY_MODE, classifier_family_for_mode, register_mode_definitions
from .training import _calibrate_threshold, _features_for, _mixed_scene_features, _ood_features_for, _validated_rows


def _write_weights(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as output:
        np.savez_compressed(output, **arrays)
        temp = Path(output.name)
    temp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        temp = Path(output.name)
    temp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reviewed_real_rows(
    data_dirs: list[str | Path], *, known_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect accepted signal crops only from fresh, complete 10-second reviews."""
    from .splits import FAMILY_BY_MODE, METHOD_BY_MODE, register_mode_definitions

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    per_capture: dict[str, list[str]] = {}
    skipped = {"excluded": 0, "unreviewed_windows": 0, "unlabeled_regions": 0}
    window_count_by_capture: dict[str, int] = {}
    window_length_cache: dict[int, int] = {}
    for data_dir in dict.fromkeys(Path(item).expanduser().resolve() for item in data_dirs):
        store = CaptureStore(data_dir / "dataset.sqlite3")
        register_mode_definitions(store.accepted_mode_definitions())
        for capture in store.list_captures():
            if capture["source_kind"] == "synthetic" or capture["source_metadata"].get("exclude_training") is True:
                skipped["excluded"] += 1
                continue
            digest = capture["sha256"]
            if digest in known_hashes:
                continue
            sample_rate = int(capture["sample_rate"])
            window_length = window_length_cache.setdefault(sample_rate, sample_rate * WINDOW_SECONDS)
            regions = store.regions(capture["id"])
            reviews = store.scene_coverages(capture["id"])
            active_labeled = [
                region for region in regions
                if region["lineage_state"] == "active"
                and region["review_state"] == "accepted"
                and isinstance(region.get("mode_id"), str)
                and region["mode_id"].casefold() in METHOD_BY_MODE
                and region.get("family") == FAMILY_BY_MODE.get(region["mode_id"].casefold())
                and region.get("method") == METHOD_BY_MODE.get(region["mode_id"].casefold())
            ]
            if not active_labeled:
                continue
            group_token = (
                capture["source_metadata"].get("auto_capture_session_id")
                or capture.get("parent_capture_id")
                or digest
            )
            group = "reviewed-" + hashlib.sha256(str(group_token).encode()).hexdigest()
            per_capture[capture["id"]] = []
            starts = range(0, int(capture["sample_count"]), window_length)
            for window_start in starts:
                window_samples = min(window_length, int(capture["sample_count"]) - window_start)
                if window_samples != window_length or not any(
                    _review_covers_window(review, window_start, window_samples) for review in reviews
                ):
                    skipped["unreviewed_windows"] += 1
                    continue
                window_count_by_capture[capture["id"]] = window_count_by_capture.get(capture["id"], 0) + 1
                window_end = window_start + window_samples
                for region in active_labeled:
                    region_start = int(region["sample_start"])
                    region_end = region_start + int(region["sample_count"])
                    start, end = max(window_start, region_start), min(window_end, region_end)
                    if end <= start:
                        continue
                    mode_id = region["mode_id"].casefold()
                    key = (digest, start, end - start, mode_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({
                        "capture_id": capture["id"], "audio_path": capture["audio_path"],
                        "sha256": digest, "sample_rate": sample_rate,
                        "sample_start": start, "sample_count": end - start,
                        "frequency_low_hz": region["frequency_low_hz"],
                        "frequency_high_hz": region["frequency_high_hz"],
                        "lineage_group": group, "mode": mode_id,
                        "kind": "fully_reviewed_real_10s_window", "split": "train",
                        "source_kind": capture["source_kind"],
                        "region_id": region["id"], "window_start": window_start,
                    })
                    per_capture[capture["id"]].append(mode_id)
            if not per_capture[capture["id"]]:
                per_capture.pop(capture["id"], None)
    summary = {
        "examples": len(rows),
        "capture_count": len(per_capture),
        "lineage_group_count": len({row["lineage_group"] for row in rows}),
        "classes": sorted({row["mode"] for row in rows}),
        "examples_by_mode": {mode: sum(row["mode"] == mode for row in rows) for mode in sorted({row["mode"] for row in rows})},
        "complete_review_windows_by_capture": window_count_by_capture,
        "skipped": skipped,
    }
    return rows, summary


def _fldigi_synthetic_rows(
    data_dirs: list[str | Path], *, known_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load canonical native Fldigi samples and reserve deterministic holdouts per mode."""
    rows: list[dict[str, Any]] = []
    candidates_by_mode: dict[str, list[dict[str, Any]]] = {}
    seen_hashes = set(known_hashes)
    skipped = 0
    for data_dir in dict.fromkeys(Path(item).expanduser().resolve() for item in data_dirs):
        store = CaptureStore(data_dir / "dataset.sqlite3")
        register_mode_definitions(store.accepted_mode_definitions())
        for capture in store.list_captures():
            metadata = capture["source_metadata"]
            if capture["source_kind"] != "synthetic" or metadata.get("generator") != "sig-id-fldigi":
                continue
            digest = str(capture["sha256"])
            if digest in seen_hashes:
                skipped += 1
                continue
            regions = store.regions(capture["id"])
            if len(regions) != 1:
                skipped += 1
                continue
            region = regions[0]
            mode_id = region.get("mode_id")
            if (
                region["lineage_state"] != "active" or region["review_state"] != "accepted"
                or not isinstance(mode_id, str) or mode_id.casefold() not in METHOD_BY_MODE
                or region["family"] != FAMILY_BY_MODE.get(mode_id.casefold())
                or region["method"] != METHOD_BY_MODE.get(mode_id.casefold())
                or region["sample_start"] != 0 or region["sample_count"] != capture["sample_count"]
            ):
                skipped += 1
                continue
            waveform = Path(capture["audio_path"])
            actual_digest = hashlib.sha256(waveform.read_bytes()).hexdigest()
            if actual_digest != digest:
                raise ValueError(f"Fldigi synthetic capture {capture['id']} changed after import")
            seen_hashes.add(digest)
            candidates_by_mode.setdefault(mode_id.casefold(), []).append({
                "capture_id": capture["id"], "audio_path": str(waveform),
                "sha256": digest, "sample_rate": int(capture["sample_rate"]),
                "sample_start": 0, "sample_count": int(capture["sample_count"]),
                "frequency_low_hz": region["frequency_low_hz"],
                "frequency_high_hz": region["frequency_high_hz"],
                "lineage_group": f"fldigi-{mode_id.casefold()}-{digest}",
                "mode": mode_id.casefold(), "kind": "native_fldigi_transmission",
                "source_kind": "synthetic", "region_id": region["id"],
            })

    split_counts: dict[str, dict[str, int]] = {}
    for mode_id, items in sorted(candidates_by_mode.items()):
        ordered = sorted(items, key=lambda item: hashlib.sha256(
            f"{mode_id}:{item['sha256']}".encode()
        ).hexdigest())
        # Keep a train example for each represented mode. Hold out one source
        # for validation and test only when at least three independent examples
        # exist, avoiding unsupported validation claims for one-off captures.
        assignments = ["train"] * len(ordered)
        if len(ordered) >= 3:
            assignments[0], assignments[1] = "validation", "test"
        elif len(ordered) == 2:
            assignments[0] = "validation"
        for item, split in zip(ordered, assignments):
            item["split"] = split
            rows.append(item)
        split_counts[mode_id] = {split: assignments.count(split) for split in ("train", "validation", "test")}
    return rows, {
        "examples": len(rows), "mode_count": len(candidates_by_mode),
        "examples_by_mode_and_split": split_counts, "skipped": skipped,
    }


def train_hierarchical(
    split_path: str | Path,
    data_dir: str | Path,
    output_manifest: str | Path,
    *,
    ood_corpus: str | Path | None = None,
    reviewed_data_dirs: list[str | Path] | None = None,
    fldigi_data_dirs: list[str | Path] | None = None,
    initial_model_path: str | Path | None = None,
    epochs: int = 1200,
    feature_version: str = DEFAULT_FEATURE_VERSION,
    classifier_window_seconds: float = CLASSIFIER_WINDOW_SECONDS,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    split_path = Path(split_path).expanduser().resolve(strict=True)
    data_dir = Path(data_dir).expanduser().resolve()
    manifest_path = Path(output_manifest).expanduser().resolve()
    if feature_version not in FEATURE_COUNT_BY_VERSION or epochs < 1:
        raise ValueError("invalid feature version or training epoch count")
    store = CaptureStore(data_dir / "dataset.sqlite3")
    register_mode_definitions(store.accepted_mode_definitions())
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    rows = _validated_rows(split_path, store)
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    known_hashes = {row["sha256"] for row in rows}
    real_rows, real_summary = _reviewed_real_rows(reviewed_data_dirs or [], known_hashes=known_hashes)
    fldigi_rows, fldigi_summary = _fldigi_synthetic_rows(fldigi_data_dirs or [], known_hashes=known_hashes)
    train_rows.extend(real_rows)
    train_rows.extend(row for row in fldigi_rows if row["split"] == "train")
    validation_rows.extend(row for row in fldigi_rows if row["split"] == "validation")
    train_x, train_labels, _ = _features_for(train_rows, feature_version, classifier_window_seconds)
    validation_x, validation_labels, _ = _features_for(validation_rows, feature_version, classifier_window_seconds)
    scene_features = {
        split: _mixed_scene_features(split_manifest, store, split, feature_version, classifier_window_seconds)
        for split in ("train", "validation")
    }
    for split, (scene_x, scene_labels, _) in scene_features.items():
        if not len(scene_labels):
            continue
        if split == "train":
            train_x = np.concatenate((train_x, scene_x), axis=0)
            train_labels = np.concatenate((train_labels, scene_labels), axis=0)
        else:
            validation_x = np.concatenate((validation_x, scene_x), axis=0)
            validation_labels = np.concatenate((validation_labels, scene_labels), axis=0)
    ood_rows = load_ood_corpus(ood_corpus) if ood_corpus else []
    ood_validation = [row for row in ood_rows if row["split"] == "validation"]
    if ood_corpus and len(ood_validation) < 6:
        raise ValueError("OOD corpus needs at least six validation sources for calibration")
    ood_x = _ood_features_for(ood_validation, feature_version, classifier_window_seconds) if ood_validation else np.empty((0, FEATURE_COUNT_BY_VERSION[feature_version]), dtype=np.float32)
    mode_ids = sorted(set(train_labels.tolist()) - {"noise"})
    unknown_modes = set(mode_ids) - set(METHOD_BY_MODE) - set(FAMILY_BY_MODE)
    if unknown_modes:
        raise ValueError(f"training labels lack family/method definitions: {sorted(unknown_modes)}")
    family_by_mode = {mode: classifier_family_for_mode(mode) for mode in mode_ids}
    families = sorted(set(family_by_mode.values()))
    if not families:
        raise ValueError("no labeled signal modes occur in the training split")
    initial_metadata = None
    initial_components = None
    if initial_model_path is not None:
        from .classifier import load_model
        initial_metadata, initial_components = load_model(initial_model_path)
        if initial_metadata.get("feature_version") != feature_version:
            raise ValueError("cannot warm-start classifier with a different feature version")

    def warm_start(parameters: dict[str, np.ndarray] | None, classes: list[str]) -> dict[str, np.ndarray] | None:
        if parameters is None:
            return None
        return {**parameters, "classes": classes}

    family_for_row = np.asarray([family_by_mode.get(str(mode), "") for mode in train_labels], dtype=object)
    signal_mask = train_labels != "noise"
    total_epoch_units = epochs * (1 + len(families))
    completed_epoch_units = 0

    def report_epoch(phase: str, epoch: int, phase_epochs: int) -> None:
        nonlocal completed_epoch_units
        completed_epoch_units += 1
        if progress_callback:
            fraction = min(1.0, completed_epoch_units / max(1, total_epoch_units))
            progress_callback(fraction, f"{phase} · epoch {epoch}/{phase_epochs}")

    initial_family = warm_start(
        initial_components["family"], initial_metadata.get("families", [])
    ) if initial_components else None
    family_parameters = fit_softmax(
        train_x[signal_mask], family_for_row[signal_mask], families, epochs=epochs,
        initial_parameters=initial_family,
        progress_callback=lambda epoch, count: report_epoch("Family classifier", epoch, count),
    )
    expert_parameters: dict[str, dict[str, np.ndarray]] = {}
    classes_by_family: dict[str, list[str]] = {}
    mode_thresholds: dict[str, float] = {}
    calibrations: dict[str, Any] = {}
    # Validation unknowns include other known families, noise, and optional OOD samples.
    val_family_labels = np.asarray([family_by_mode.get(str(mode), "") for mode in validation_labels], dtype=object)
    val_is_known = val_family_labels != ""
    family_val_probs = predict_probabilities(validation_x, family_parameters)
    family_val_indices = np.asarray([families.index(x) if x else -1 for x in val_family_labels], dtype=np.int64)
    family_cal_x = np.concatenate((validation_x, ood_x), axis=0) if len(ood_x) else validation_x
    family_cal_y = np.concatenate((family_val_indices, np.full(len(ood_x), -1, dtype=np.int64))) if len(ood_x) else family_val_indices
    family_cal_p = predict_probabilities(family_cal_x, family_parameters)
    family_calibration = _calibrate_threshold(family_cal_p, family_cal_y)
    family_thresholds = {family: float(family_calibration["threshold"]) for family in families}

    for family in families:
        modes = sorted(mode for mode in mode_ids if family_by_mode[mode] == family)
        classes_by_family[family] = modes
        family_train_mask = np.asarray([family_by_mode.get(str(label)) == family for label in train_labels])
        initial_expert = None
        if initial_components and family in initial_components["experts"]:
            prior_modes = initial_metadata.get("mode_classes_by_family", {}).get(family, [])
            initial_expert = warm_start(initial_components["experts"][family], prior_modes)
        expert = fit_softmax(
            train_x[family_train_mask], train_labels[family_train_mask], modes,
            epochs=epochs, initial_parameters=initial_expert,
            progress_callback=lambda epoch, count, family=family: report_epoch(
                f"{family} mode classifier", epoch, count,
            ),
        )
        expert_parameters[family] = expert
        if len(modes) == 1:
            mode_thresholds[family] = 0.0
            calibrations[family] = {"status": "single_mode_family", "threshold": 0.0}
            continue
        val_mode_indices = np.asarray([modes.index(str(label)) if str(label) in modes else -1 for label in validation_labels], dtype=np.int64)
        expert_cal_x = np.concatenate((validation_x, ood_x), axis=0) if len(ood_x) else validation_x
        expert_cal_y = np.concatenate((val_mode_indices, np.full(len(ood_x), -1, dtype=np.int64))) if len(ood_x) else val_mode_indices
        calibration = _calibrate_threshold(predict_probabilities(expert_cal_x, expert), expert_cal_y)
        mode_thresholds[family] = float(calibration["threshold"])
        calibrations[family] = calibration

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    family_file = manifest_path.stem + ".family.npz"
    family_hash = _write_weights(manifest_path.parent / family_file, family_parameters)
    experts = {}
    for family, parameters in expert_parameters.items():
        filename = manifest_path.stem + ".expert-" + str(len(experts)) + ".npz"
        digest = _write_weights(manifest_path.parent / filename, parameters)
        experts[family] = {"file": filename, "sha256": digest}
    reviewed_manifest = {
        "schema_version": 1,
        "scope": "train-only crops from fresh, complete, fully annotated 10-second windows",
        "examples": [
            {key: row[key] for key in (
                "capture_id", "sha256", "region_id", "mode", "source_kind",
                "sample_rate", "sample_start", "sample_count", "window_start", "lineage_group",
            )}
            for row in real_rows
        ],
    }
    reviewed_manifest_path = manifest_path.with_name("reviewed-training-examples.json")
    reviewed_manifest_hash = _write_json(reviewed_manifest_path, reviewed_manifest)
    metadata = {
        "schema_version": 1, "model_type": "hierarchical_family_experts",
        "feature_version": feature_version, "feature_count": FEATURE_COUNT_BY_VERSION[feature_version],
        "feature_frequency_scope": "roi_bandpass_expanded" if feature_version in ROI_FEATURE_VERSIONS else "full_audio_passband",
        "classifier_window_seconds": classifier_window_seconds,
        "sample_rate_hz": int(train_rows[0]["sample_rate"]),
        "families": families, "classes": mode_ids,
        "mode_classes_by_family": classes_by_family,
        "family_by_method": {mode: FAMILY_BY_MODE[mode] for mode in mode_ids},
        "classifier_family_by_method": family_by_mode,
        "mode_details": {
            mode: {
                "display_name": MODE_REGISTRY.get(mode, {}).get("display_name", mode),
                "method": METHOD_BY_MODE.get(mode),
                "protocol": MODE_REGISTRY.get(mode, {}).get("protocol"),
                "variant": MODE_REGISTRY.get(mode, {}).get("variant"),
            }
            for mode in mode_ids
        },
        "method_names": {mode: METHOD_BY_MODE[mode] for mode in mode_ids},
        "family_model": {"file": family_file, "sha256": family_hash}, "experts": experts,
        "family_thresholds": family_thresholds, "mode_thresholds": mode_thresholds,
        "family_calibration": family_calibration, "mode_calibration": calibrations,
        "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "split_manifest_path": str(split_path),
        "classifier_data_dir": str(data_dir),
        "ood_corpus_path": str(Path(ood_corpus).expanduser().resolve()) if ood_corpus else None,
        "ood_sources_fit_on": 0, "ood_validation_sources": len(ood_validation),
        "registry_is_training_taxonomy": False,
        "training_classes_from_training_sources_only": True,
        "reviewed_real_training": real_summary,
        "warm_started_from_model_sha256": initial_metadata.get("model_sha256") if initial_metadata else None,
        "native_fldigi_training": fldigi_summary,
        "mixed_scene_training": scene_features["train"][2],
        "mixed_scene_validation": scene_features["validation"][2],
        "reviewed_training_manifest": reviewed_manifest_path.name,
        "reviewed_training_manifest_sha256": reviewed_manifest_hash,
        "abstention_threshold": min(mode_thresholds.values()),
        "family_abstention_threshold": min(family_thresholds.values()),
        "model_sha256": hashlib.sha256((family_hash + "".join(item["sha256"] for item in experts.values())).encode()).hexdigest(),
        "unknown_rejection_validated": False,
    }
    _write_json(manifest_path, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a family classifier and one mode expert per represented family")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--ood-corpus", type=Path)
    parser.add_argument("--reviewed-data-dir", type=Path, action="append", default=[],
                        help="application dataset containing fresh complete 10-second reviews; may be repeated")
    parser.add_argument("--fldigi-data-dir", type=Path, action="append", default=[],
                        help="dataset containing imported native Fldigi synthetic samples; may be repeated")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--feature-version", choices=tuple(FEATURE_COUNT_BY_VERSION), default=DEFAULT_FEATURE_VERSION)
    parser.add_argument("--classifier-window-seconds", type=float, default=CLASSIFIER_WINDOW_SECONDS)
    args = parser.parse_args()
    try:
        result = train_hierarchical(
            args.split, args.data_dir, args.output_manifest, ood_corpus=args.ood_corpus,
            reviewed_data_dirs=args.reviewed_data_dir, fldigi_data_dirs=args.fldigi_data_dir,
            epochs=args.epochs, feature_version=args.feature_version,
            classifier_window_seconds=args.classifier_window_seconds,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "model_manifest": str(args.output_manifest), "families": result["families"],
        "classes": result["classes"], "reviewed_real_training": result["reviewed_real_training"],
    }, indent=2))


if __name__ == "__main__":
    main()
