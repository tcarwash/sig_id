"""Build a compact Laya few-shot reference set from train-split synthetic captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from sigid.classifier import FEATURE_VERSION, extract_features
from sigid.wav import WavSource


MODES = ("2fsk", "4fsk", "8fsk", "16fsk", "bpsk", "qpsk")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wav_path(corpus_dir: Path, record: dict[str, Any]) -> Path:
    mode, index, kind = record["key"].split("/")
    if mode == "noise":
        return corpus_dir / "noise" / f"{index}.wav"
    if kind == "clean":
        return corpus_dir / "sources" / mode / f"{index}.wav"
    raise ValueError("reference generation only accepts clean roots and noise-only roots")


def _small_parameters(manifest_path: Path, mode: str) -> dict[str, Any]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    parameters = data.get("parameters", {})
    if mode.endswith("fsk"):
        return {
            "tone_count": parameters.get("tone_count"),
            "symbol_rate": parameters.get("symbol_rate"),
            "tone_spacing_hz": parameters.get("tone_spacing_hz"),
            "phase_continuous": parameters.get("phase_continuous"),
        }
    if mode in {"bpsk", "qpsk"}:
        phase_map = parameters.get("phase_mapping", {})
        return {
            "phase_state_count": len(phase_map),
            "symbol_rate": parameters.get("symbol_rate"),
            "phase_continuous_carrier": parameters.get("phase_continuous_carrier"),
        }
    return {"negative_kind": parameters.get("negative_kind", "noise_only")}


def _features(path: Path, sample_rate: int) -> list[float]:
    values = np.asarray([sample for block in WavSource(path).blocks() for sample in block.samples], dtype=np.float64)
    return [round(float(value), 6) for value in extract_features(values, sample_rate, feature_version=FEATURE_VERSION)]


def build(corpus_dir: Path, split_path: Path, output: Path, examples_per_method: int, noise_examples: int) -> dict[str, Any]:
    corpus_path = corpus_dir / "corpus.json"
    corpus_bytes = corpus_path.read_bytes()
    corpus = json.loads(corpus_bytes)
    split_bytes = split_path.read_bytes()
    split = json.loads(split_bytes)
    corpus_digest = hashlib.sha256(corpus_bytes).hexdigest()
    if split.get("generator") != "sig-id-split-corpus" or split.get("source_corpus_sha256") != corpus_digest:
        raise ValueError("split manifest does not match this corpus")
    records = split.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("split manifest has no records")
    group_splits: dict[str, str] = {}
    for record in records:
        group, split_name = record.get("lineage_group"), record.get("split")
        if not isinstance(group, str) or split_name not in {"train", "validation", "test"}:
            raise ValueError("split manifest contains an invalid record")
        if group in group_splits and group_splits[group] != split_name:
            raise ValueError(f"lineage group {group} crosses split boundaries")
        group_splits[group] = split_name
    corpus_by_id = {record["capture_id"]: record for record in corpus.get("records", [])}
    train = [record for record in records if record.get("split") == "train"]
    selected: list[dict[str, Any]] = []
    for mode in MODES:
        candidates = sorted(
            (record for record in train if record.get("mode") == mode and record.get("kind") == "clean"),
            key=lambda record: record["key"],
        )
        groups: set[str] = set()
        for record in candidates:
            if record["lineage_group"] in groups:
                continue
            selected.append(record)
            groups.add(record["lineage_group"])
            if len(groups) == examples_per_method:
                break
        if len(groups) != examples_per_method:
            raise ValueError(f"train split has fewer than {examples_per_method} independent clean {mode} sources")
    noises = sorted((record for record in train if record.get("mode") == "noise"), key=lambda record: record["key"])
    if len(noises) < noise_examples:
        raise ValueError("train split has too few noise-only examples")
    selected.extend(noises[:noise_examples])

    examples = []
    for record in selected:
        source = corpus_by_id.get(record["capture_id"])
        if not source or source.get("sha256") != record.get("sha256") or source.get("key") != record.get("key"):
            raise ValueError("split record is missing or inconsistent in the corpus manifest")
        path = _wav_path(corpus_dir, record)
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"waveform hash mismatch for {record['key']}")
        sidecar = path.with_suffix(path.suffix + ".json")
        samples = WavSource(path)
        is_noise = record["mode"] == "noise"
        examples.append({
            "method": "unknown" if is_noise else record["mode"],
            "example_kind": "noise_only_negative" if is_noise else "modulated_signal",
            "family": None if is_noise else json.loads(sidecar.read_text(encoding="utf-8"))["labels"]["modulation_family"],
            "source_kind": "synthetic_train_reference",
            "source_split": "train",
            "sample_rate_hz": samples.metadata()["sample_rate"],
            "parameters": _small_parameters(sidecar, record["mode"]),
            "features": _features(path, samples.metadata()["sample_rate"]),
        })
    bundle = {
        "schema_version": 1,
        "feature_version": FEATURE_VERSION,
        "source_corpus_sha256": corpus_digest,
        "source_split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "source_split_seed": split.get("seed"),
        "source_split": "train",
        "examples": examples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True, help="directory containing corpus.json, sources/, and noise/")
    parser.add_argument("--split", type=Path, required=True, help="lineage-safe split manifest for this corpus")
    parser.add_argument("--output", type=Path, default=Path("sigid/laya_reference_examples.json"))
    parser.add_argument("--examples-per-method", type=int, default=2)
    parser.add_argument("--noise-examples", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.examples_per_method <= 8 or not 0 <= args.noise_examples <= 8:
        parser.error("reference counts must be between 1 and 8")
    bundle = build(args.corpus_dir, args.split, args.output, args.examples_per_method, args.noise_examples)
    print(json.dumps({"output": str(args.output), "examples": len(bundle["examples"]), "train_only": True}, indent=2))


if __name__ == "__main__":
    main()
