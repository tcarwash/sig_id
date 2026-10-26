"""Deterministic lineage-grouped train/validation/test manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

from .corpus import MODES
from .dataset import CaptureStore
from .modes import MODE_REGISTRY


METHOD_BY_MODE = {
    "2fsk": "binary frequency shift keying",
    "4fsk": "4-ary frequency shift keying",
    "8fsk": "8-ary frequency shift keying",
    "16fsk": "16-ary frequency shift keying",
    "bpsk": "binary phase-shift keying",
    "qpsk": "quadrature phase-shift keying",
}
FAMILY_BY_MODE = {
    "2fsk": "FSK",
    "4fsk": "MFSK",
    "8fsk": "MFSK",
    "16fsk": "MFSK",
    "bpsk": "PSK",
    "qpsk": "PSK",
}

# Generated baseline classes above remain stable. Add recognized IDs from the
# dataset mode catalog so the classifier can learn reviewed classes as they
# enter its train split without changing the synthetic generator's six modes.
for _mode_id, _definition in MODE_REGISTRY.items():
    _family = _definition.get("family")
    _method = _definition.get("method")
    if isinstance(_family, str) and _family and isinstance(_method, str) and _method:
        METHOD_BY_MODE.setdefault(_mode_id, _method)
        FAMILY_BY_MODE.setdefault(_mode_id, _family)


def families_for_modes(modes: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Families present for the supplied classifier mode IDs, in stable order."""
    present = {FAMILY_BY_MODE[mode] for mode in modes if mode in FAMILY_BY_MODE}
    preferred = ["FSK", "MFSK", "PSK"]
    return tuple([family for family in preferred if family in present]
                 + sorted(present - set(preferred)))


def classifier_family_for_mode(mode_id: str) -> str:
    """Return the modulation-family stage used by the hierarchical classifier.

    Catalog ``family`` values can describe a protocol community (for example,
    weak-signal digital) rather than the waveform's modulation family. Group
    recognizable FSK and PSK methods by modulation, keeping other catalog
    families as their own first-stage category.
    """
    method = str(METHOD_BY_MODE.get(mode_id, "")).casefold().replace(" ", "")
    catalog_family = FAMILY_BY_MODE.get(mode_id, "")
    if "gfsk" in method or "frequencyshiftkeying" in method or "fsk" in method:
        if any(token in method for token in ("binary", "2-fsk", "2fsk")):
            return "FSK"
        return "MFSK"
    if "psk" in method or "phase-shiftkeying" in method or "phaseshiftkeying" in method:
        return "PSK"
    return catalog_family


def register_mode_definitions(definitions: list[dict[str, Any]]) -> None:
    """Make accepted dataset mode IDs available to classifier train/inference."""
    for definition in definitions:
        mode_id = definition.get("id")
        family = definition.get("family")
        method = definition.get("method")
        if not all(isinstance(value, str) and value.strip() for value in (mode_id, family, method)):
            continue
        key = mode_id.strip().casefold()
        if key in METHOD_BY_MODE and (
            METHOD_BY_MODE[key] != method.strip() or FAMILY_BY_MODE[key] != family.strip()
        ):
            raise ValueError(f"dataset mode {mode_id!r} conflicts with the classifier's existing taxonomy")
        METHOD_BY_MODE[key] = method.strip()
        FAMILY_BY_MODE[key] = family.strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
        temporary = Path(file.name)
    temporary.replace(path)


def _split_counts(group_count: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    if group_count < 3:
        raise ValueError("each split stratum needs at least three independent source groups")
    train_fraction, validation_fraction, _ = fractions
    train = max(1, round(group_count * train_fraction))
    validation = max(1, round(group_count * validation_fraction))
    while train + validation >= group_count:
        if train > validation and train > 1:
            train -= 1
        elif validation > 1:
            validation -= 1
        else:
            train -= 1
    return train, validation, group_count - train - validation


def _scene_aware_assignments(
    group_modes: dict[str, str],
    group_hashes: dict[str, str],
    scenes: list[dict[str, Any]],
    capture_groups: dict[str, str],
    *,
    seed: int,
    assignment_version: int,
    fractions: tuple[float, float, float],
) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Keep every set of source lineages used by a scene in one split."""
    modes = (*MODES, *sorted(set(group_modes.values()) - set(MODES)))
    parent = {group: group for group in group_modes}

    def find(group: str) -> str:
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if group_hashes[left_root] <= group_hashes[right_root]:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    dependencies = []
    exclusions = []
    for scene in scenes:
        metadata = scene["source_metadata"]
        component_ids = metadata.get("component_capture_ids")
        if not isinstance(component_ids, list) or len(component_ids) < 2:
            exclusions.append({"capture_id": scene["id"], "reason": "scene has no valid component list"})
            continue
        signal_component_ids = metadata.get("signal_component_capture_ids", component_ids)
        if (
            not isinstance(signal_component_ids, list) or not signal_component_ids
            or any(component_id not in component_ids for component_id in signal_component_ids)
        ):
            exclusions.append({"capture_id": scene["id"], "reason": "scene has no valid signal component list"})
            continue
        if any(component_id not in capture_groups for component_id in signal_component_ids):
            exclusions.append({
                "capture_id": scene["id"],
                "reason": "one or more signal components are absent from the source corpus",
                "missing_component_capture_ids": [item for item in signal_component_ids if item not in capture_groups],
            })
            continue
        groups = sorted({capture_groups[component_id] for component_id in signal_component_ids})
        for group in groups[1:]:
            union(groups[0], group)
        dependencies.append({
            "capture_id": scene["id"],
            "sha256": scene["sha256"],
            "component_capture_ids": component_ids,
            "lineage_groups": groups,
        })

    clusters: dict[str, set[str]] = {}
    for group in group_modes:
        clusters.setdefault(find(group), set()).add(group)
    cluster_modes = {
        root: {group_modes[group] for group in groups}
        for root, groups in clusters.items()
    }
    independent_cluster_counts = {
        mode: sum(mode in modes for modes in cluster_modes.values())
        for mode in (*modes, "noise")
    }
    for mode, independent_clusters in independent_cluster_counts.items():
        if independent_clusters < 3:
            raise ValueError(
                f"mode {mode} needs at least three independent source groups after mixed-scene lineage joins"
            )

    split_names = ("train", "validation", "test")
    targets: dict[str, tuple[int, int, int]] = {}
    for mode in (*modes, "noise"):
        count = sum(mode in modes for modes in cluster_modes.values())
        targets[mode] = _split_counts(count, fractions)

    rng = random.Random(seed)
    ordered = list(clusters)
    rng.shuffle(ordered)
    seed_priority = {root: rng.random() for root in ordered} if assignment_version == 2 else {}
    rarity = {
        root: sum(1 / sum(mode in modes for modes in cluster_modes.values()) for mode in cluster_modes[root])
        for root in ordered
    }
    if assignment_version == 1:
        ordered.sort(key=lambda root: (
            -len(cluster_modes[root]),
            -rarity[root],
            group_hashes[min(clusters[root], key=lambda g: group_hashes[g])],
        ))
    else:
        ordered.sort(key=lambda root: (
            -len(cluster_modes[root]),
            -rarity[root],
            seed_priority[root],
            group_hashes[min(clusters[root], key=lambda g: group_hashes[g])],
        ))
    scene_cluster_roots = {
        find(group)
        for dependency in dependencies
        for group in dependency["lineage_groups"]
    }
    assigned_counts = {split: {mode: 0 for mode in (*modes, "noise")} for split in split_names}
    assignment: dict[str, str] = {}
    held_out_scene_assigned = False
    for root in ordered:
        modes = cluster_modes[root]
        scores = []
        for split_index, split in enumerate(split_names):
            score = 0.0
            for mode in modes:
                current = assigned_counts[split][mode]
                target = targets[mode][split_index]
                weight = 1 / max(1, target)
                score += (((current + 1 - target) ** 2) - ((current - target) ** 2)) * weight
            score += assigned_counts[split]["noise"] * 0.001
            if root in scene_cluster_roots and not held_out_scene_assigned and split != "test":
                score += 1_000_000
            scores.append((score + rng.random() * 1e-6, split_index))
        _, chosen_index = min(scores)
        chosen = split_names[chosen_index]
        for group in clusters[root]:
            assignment[group] = chosen
        for mode in modes:
            assigned_counts[chosen][mode] += 1
        if root in scene_cluster_roots and chosen == "test":
            held_out_scene_assigned = True

    for split in split_names:
        present = {mode for mode, count in assigned_counts[split].items() if count}
        if not set((*modes, "noise")).issubset(present):
            raise ValueError(f"scene-aware split assignment cannot place every method and noise in {split}")
    for scene in dependencies:
        component_splits = {assignment[group] for group in scene["lineage_groups"]}
        if len(component_splits) != 1:
            raise AssertionError("mixed-scene source lineages crossed split boundaries")
        scene["split"] = next(iter(component_splits))
    return assignment, dependencies, exclusions, independent_cluster_counts


def create_split_manifest(
    corpus_path: str | Path,
    store: CaptureStore,
    *,
    seed: int = 20261021,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    assignment_version: int = 1,
) -> dict[str, Any]:
    """Validate a corpus against the dataset and split whole source lineages."""
    corpus_path = Path(corpus_path).expanduser().resolve(strict=True)
    raw = corpus_path.read_bytes()
    corpus = json.loads(raw)
    if not isinstance(corpus, dict) or corpus.get("config", {}).get("modes") != list(MODES):
        raise ValueError("corpus index must contain all supported modulation methods")
    records = corpus.get("records")
    if not isinstance(records, list) or corpus.get("record_count") != len(records):
        raise ValueError("corpus record count does not match its index")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if assignment_version not in (1, 2):
        raise ValueError("assignment_version must be 1 or 2")
    if len(fractions) != 3 or any(not 0 < fraction < 1 for fraction in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must be three positive values summing to 1")

    unique_captures: set[str] = set()
    group_modes: dict[str, str] = {}
    group_hashes: dict[str, str] = {}
    validated_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("corpus records must be objects")
        capture_id = record.get("capture_id")
        mode = record.get("mode")
        group = record.get("lineage_group")
        if not isinstance(capture_id, str) or not isinstance(group, str) or mode not in (*METHOD_BY_MODE, "noise"):
            raise ValueError("corpus record is missing capture, mode or lineage group")
        if capture_id in unique_captures:
            raise ValueError(f"capture {capture_id} appears more than once in the corpus")
        unique_captures.add(capture_id)
        if group in group_modes and group_modes[group] != mode:
            raise ValueError("one lineage group cannot contain multiple modulation methods")
        group_modes[group] = mode

        try:
            capture = store.capture(capture_id)
        except KeyError as exc:
            raise ValueError(f"corpus capture {capture_id} is missing from the dataset") from exc
        if capture["source_kind"] != "synthetic" or capture["sha256"] != record.get("sha256"):
            raise ValueError(f"corpus capture {capture_id} does not match its source hash")
        if capture["id"] == group:
            if capture["parent_capture_id"] is not None:
                raise ValueError("lineage group root must be a parentless capture")
            group_hashes[group] = capture["sha256"]
        elif capture["parent_capture_id"] != group:
            raise ValueError(f"capture {capture_id} is not a direct child of its declared lineage group")
        if capture["parent_capture_id"] != record.get("parent_capture_id"):
            raise ValueError(f"corpus parent reference disagrees for capture {capture_id}")

        regions = store.regions(capture_id)
        if len(regions) != 1:
            raise ValueError(f"capture {capture_id} must have one accepted full-capture label")
        label = regions[0]
        if mode == "noise":
            valid_label = (
                label["sample_start"] == 0
                and label["sample_count"] == capture["sample_count"]
                and label["source"] == "synthetic"
                and label["review_state"] == "accepted"
                and label["family"] is None
                and label["method"] is None
                and label["protocol"] is None
                and label["variant"] is None
                and record.get("kind") == "noise"
            )
        else:
            valid_label = label["sample_start"] == 0 and (
                label["sample_count"] == capture["sample_count"]
                and label["source"] == "synthetic"
                and label["review_state"] == "accepted"
                and label["family"] == FAMILY_BY_MODE[mode]
                and label["protocol"] is None
                and label["variant"] is None
                and label["method"] == METHOD_BY_MODE[mode]
            )
        if not valid_label:
            raise ValueError(f"capture {capture_id} has a missing or contradictory synthetic method label")
        validated_records.append({
            "key": record.get("key"),
            "capture_id": capture_id,
            "sha256": capture["sha256"],
            "parent_capture_id": capture["parent_capture_id"],
            "lineage_group": group,
            "mode": mode,
            "method": label["method"],
            "kind": record.get("kind"),
            "snr_db": record.get("snr_db"),
        })

    root_records = {
        record["lineage_group"]
        for record in validated_records
        if record["capture_id"] == record["lineage_group"] and record["kind"] in {"clean", "noise"}
    }
    if root_records != set(group_modes):
        raise ValueError("every lineage group must include its indexed clean root capture")
    if any(record["kind"] not in {"clean", "awgn", "noise"} for record in validated_records):
        raise ValueError("corpus contains an unsupported generated record kind")

    modes = (*MODES, *sorted(set(group_modes.values()) - set(MODES)))
    grouped: dict[str, list[str]] = {mode: [] for mode in (*modes, "noise")}
    for group, mode in group_modes.items():
        grouped[mode].append(group)
    for mode in (*modes, "noise"):
        if len(grouped[mode]) < 3:
            raise ValueError(f"mode {mode} needs at least three independent source groups for three-way evaluation")

    capture_groups = {record["capture_id"]: record["lineage_group"] for record in validated_records}
    mixed_scenes = [
        capture for capture in store.list_captures()
        if capture["source_kind"] == "synthetic"
        and capture["source_metadata"].get("generator") == "sig-id-mixed-scene"
    ]
    split_for_group, scene_dependencies, scene_exclusions, independent_cluster_counts = _scene_aware_assignments(
        group_modes,
        group_hashes,
        mixed_scenes,
        capture_groups,
        seed=seed,
        assignment_version=assignment_version,
        fractions=fractions,
    )
    split_names = ("train", "validation", "test")

    for record in validated_records:
        record["split"] = split_for_group[record["lineage_group"]]
    group_sets = {split: set() for split in split_names}
    capture_sets = {split: set() for split in split_names}
    for record in validated_records:
        group_sets[record["split"]].add(record["lineage_group"])
        capture_sets[record["split"]].add(record["capture_id"])
    if any(group_sets[left] & group_sets[right] for i, left in enumerate(split_names) for right in split_names[i + 1 :]):
        raise AssertionError("lineage leakage detected while building split manifest")
    if any(capture_sets[left] & capture_sets[right] for i, left in enumerate(split_names) for right in split_names[i + 1 :]):
        raise AssertionError("capture leakage detected while building split manifest")

    counts = {
        split: {
            "captures": sum(record["split"] == split for record in validated_records),
            "source_groups": len(group_sets[split]),
            "by_mode": {
                mode: len({record["lineage_group"] for record in validated_records if record["split"] == split and record["mode"] == mode})
                for mode in (*modes, "noise")
            },
        }
        for split in split_names
    }
    return {
        "schema_version": 1,
        "generator": "sig-id-split-corpus",
        **({"assignment_version": assignment_version} if assignment_version != 1 else {}),
        "seed": seed,
        "fractions": {"train": fractions[0], "validation": fractions[1], "test": fractions[2]},
        "source_corpus": corpus_path.name,
        "source_corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "group_rule": "source lineages connected by mixed-scene constituents share one split",
        "mixed_scene_dependencies": scene_dependencies,
        "excluded_mixed_scenes": scene_exclusions,
        "independent_source_clusters_by_mode": independent_cluster_counts,
        "counts": counts,
        "records": validated_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic lineage-safe corpus split manifest")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("SIG_ID_DATA_DIR", "~/.local/share/sig-id")).expanduser())
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--assignment-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--validation", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    args = parser.parse_args()
    try:
        result = create_split_manifest(
            args.corpus,
            CaptureStore(args.data_dir / "dataset.sqlite3"),
            seed=args.seed,
            fractions=(args.train, args.validation, args.test),
            assignment_version=args.assignment_version,
        )
        _atomic_json(args.output, result)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(args.output), "counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
