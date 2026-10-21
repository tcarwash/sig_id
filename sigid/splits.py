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


def create_split_manifest(
    corpus_path: str | Path,
    store: CaptureStore,
    *,
    seed: int = 20261021,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
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
        if not isinstance(capture_id, str) or not isinstance(group, str) or mode not in (*MODES, "noise"):
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

    grouped: dict[str, list[str]] = {mode: [] for mode in (*MODES, "noise")}
    for group, mode in group_modes.items():
        grouped[mode].append(group)
    for mode in (*MODES, "noise"):
        if len(grouped[mode]) < 3:
            raise ValueError(f"mode {mode} needs at least three independent source groups for three-way evaluation")

    split_for_group: dict[str, str] = {}
    split_names = ("train", "validation", "test")
    for mode_index, mode in enumerate((*MODES, "noise")):
        groups = sorted(grouped[mode], key=lambda group: group_hashes[group])
        rng = random.Random(seed + mode_index * 1009)
        rng.shuffle(groups)
        n_train, n_validation, _ = _split_counts(len(groups), fractions)
        for split, items in (
            ("train", groups[:n_train]),
            ("validation", groups[n_train : n_train + n_validation]),
            ("test", groups[n_train + n_validation :]),
        ):
            split_for_group.update({group: split for group in items})

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
                for mode in (*MODES, "noise")
            },
        }
        for split in split_names
    }
    return {
        "schema_version": 1,
        "generator": "sig-id-split-corpus",
        "seed": seed,
        "fractions": {"train": fractions[0], "validation": fractions[1], "test": fractions[2]},
        "source_corpus": corpus_path.name,
        "source_corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "group_rule": "all captures sharing a clean synthetic source capture ID remain in one split",
        "counts": counts,
        "records": validated_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic lineage-safe corpus split manifest")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("SIG_ID_DATA_DIR", "~/.local/share/sig-id")).expanduser())
    parser.add_argument("--seed", type=int, default=20261021)
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
        )
        _atomic_json(args.output, result)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(args.output), "counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
