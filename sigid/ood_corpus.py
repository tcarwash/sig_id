"""Build a deterministic, model-external corpus of synthetic OOK signals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .synthetic import generate_ook_wav
from .synthetic_dataset import _validate_manifest


MAX_EXAMPLES = 1000


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        temporary = Path(output.name)
    temporary.replace(path)


def _seed(base_seed: int, index: int) -> int:
    digest = hashlib.sha256(f"sig-id-ook-ood:{base_seed}:{index}".encode()).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**32)


def build_ood_corpus(
    output_dir: str | Path,
    *,
    examples: int = 30,
    seed: int = 20261021,
) -> dict[str, Any]:
    """Generate independent OOK sources with varied baud, carrier, phase and payload.

    The output is separate from the six-class training-corpus index, so the existing
    trainer cannot accidentally consume these examples as known classes.
    """
    if not isinstance(examples, int) or isinstance(examples, bool) or not 1 <= examples <= MAX_EXAMPLES:
        raise ValueError(f"examples must be between 1 and {MAX_EXAMPLES}")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "ood-corpus.json"
    config = {
        "schema_version": 1,
        "generator": "sig-id-build-ook-ood-corpus",
        "generator_version": "0.1.0",
        "modulation_family": "ASK",
        "modulation_method": "on-off keying (OOK)",
        "examples": examples,
        "seed": seed,
        "sample_rate": 12_000,
        "symbols_per_source": 256,
        "split_strategy": "3-of-5-validation-2-of-5-test-by-independent-source-index",
    }
    if index_path.exists():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if existing.get("config") != config:
            raise ValueError("existing OOK OOD index uses a different configuration; choose another output directory")
        records = {record["index"]: record for record in existing.get("records", [])}
    else:
        records = {}

    for index in range(examples):
        source_seed = _seed(seed, index)
        rng = np.random.default_rng(source_seed)
        bits = "".join(str(int(bit)) for bit in rng.integers(0, 2, size=config["symbols_per_source"]))
        symbol_rate = int(rng.choice([50, 75, 100, 150, 200]))
        carrier_hz = float(rng.choice([600, 900, 1200, 1500, 1800, 2200, 2600]))
        waveform = output_dir / "waveforms" / f"ook-{index:05d}.wav"
        sidecar = waveform.with_suffix(waveform.suffix + ".json")
        if not waveform.exists() or not sidecar.exists():
            generate_ook_wav(
                waveform,
                bits,
                sample_rate=config["sample_rate"],
                symbol_rate=symbol_rate,
                carrier_hz=carrier_hz,
                amplitude=float(rng.uniform(0.2, 0.8)),
                initial_phase_radians=float(rng.uniform(-np.pi, np.pi)),
            )
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        _validate_manifest(manifest, waveform)
        record = {
            "index": index,
            "source_seed": source_seed,
            "waveform": str(waveform.relative_to(output_dir)),
            "waveform_sha256": hashlib.sha256(waveform.read_bytes()).hexdigest(),
            "manifest": str(sidecar.relative_to(output_dir)),
            "manifest_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            "parent_index": None,
            "independent_source": True,
            "split": "validation" if index % 5 in {0, 1, 2} else "test",
            "symbol_rate": symbol_rate,
            "carrier_hz": carrier_hz,
        }
        records[index] = record
        _atomic_json(index_path, {"config": config, "record_count": len(records), "records": [records[key] for key in sorted(records)]})
    result = {"config": config, "record_count": len(records), "records": [records[key] for key in sorted(records)]}
    _atomic_json(index_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a held-out synthetic OOK/ASK benchmark corpus")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    try:
        result = build_ood_corpus(args.output_dir, examples=args.examples, seed=args.seed)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"index": str(args.output_dir / "ood-corpus.json"), "record_count": result["record_count"]}, indent=2))


def load_ood_corpus(index_path: str | Path) -> list[dict[str, Any]]:
    """Verify the OOD corpus index, independent sources, hashes and exact WAV recipes."""
    index_path = Path(index_path).expanduser().resolve(strict=True)
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid OOD corpus index: {exc}") from exc
    config = index.get("config") if isinstance(index, dict) else None
    indexed_records = index.get("records") if isinstance(index, dict) else None
    if (
        not isinstance(config, dict)
        or config.get("generator") != "sig-id-build-ook-ood-corpus"
        or config.get("generator_version") != "0.1.0"
        or not isinstance(config.get("examples"), int)
        or isinstance(config.get("examples"), bool)
        or config.get("examples", 0) < 1
        or not isinstance(indexed_records, list)
        or index.get("record_count") != len(indexed_records)
        or config.get("examples") != len(indexed_records)
    ):
        raise ValueError("unsupported or inconsistent OOD corpus index")
    root = index_path.parent.resolve()
    records = indexed_records
    seen_indices: set[int] = set()
    seen_hashes: set[str] = set()
    verified = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("OOD corpus records must be objects")
        source_index = record.get("index")
        if (
            not isinstance(source_index, int)
            or isinstance(source_index, bool)
            or source_index in seen_indices
            or record.get("independent_source") is not True
            or record.get("parent_index") is not None
            or record.get("split") not in {"validation", "test"}
            or record.get("split") != ("validation" if source_index % 5 in {0, 1, 2} else "test")
            or source_index >= config["examples"]
        ):
            raise ValueError("OOD corpus must contain uniquely indexed independent validation/test sources")
        seen_indices.add(source_index)
        waveform_name = record.get("waveform")
        manifest_name = record.get("manifest")
        if not isinstance(waveform_name, str) or not isinstance(manifest_name, str):
            raise ValueError("OOD corpus waveform and manifest paths must be strings")
        waveform = (root / waveform_name).resolve(strict=True)
        sidecar = (root / manifest_name).resolve(strict=True)
        if not waveform.is_relative_to(root) or not sidecar.is_relative_to(root):
            raise ValueError("OOD corpus paths must remain inside the indexed directory")
        waveform_digest = hashlib.sha256(waveform.read_bytes()).hexdigest()
        sidecar_digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
        if (
            waveform_digest != record.get("waveform_sha256")
            or sidecar_digest != record.get("manifest_sha256")
            or waveform_digest in seen_hashes
        ):
            raise ValueError("OOD corpus hashes are missing, duplicated or inconsistent")
        seen_hashes.add(waveform_digest)
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        _validate_manifest(manifest, waveform)
        if manifest.get("generator") != "sig-id-ook" or manifest.get("labels", {}).get("modulation_family") != "ASK":
            raise ValueError("OOD corpus contains a signal outside the declared synthetic OOK family")
        verified.append({
            **record,
            "waveform_path": str(waveform),
            "manifest_path": str(sidecar),
            "sample_rate": manifest["sample_rate"],
        })
    if seen_indices != set(range(config["examples"])):
        raise ValueError("OOD corpus indices must cover the configured independent sources exactly once")
    return verified


if __name__ == "__main__":
    main()
