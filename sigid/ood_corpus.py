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


if __name__ == "__main__":
    main()
