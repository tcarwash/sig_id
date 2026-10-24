"""Build an optional synthetic source corpus for CV detector training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .dataset import CaptureStore
from .synthetic import generate_ask_wav, generate_ook_wav, generate_ppm_wav
from .synthetic_dataset import import_synthetic_wav


TYPES = ("ook", "ask", "ppm")
MAX_EXAMPLES_PER_TYPE = 100


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _source_seed(seed: int, signal_type: str, index: int) -> int:
    token = f"{seed}:{signal_type}:{index}".encode()
    offset = int.from_bytes(hashlib.sha256(token).digest()[:4], "little")
    return (seed + offset) % (2**32)


def _generate_source(path: Path, signal_type: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sample_rate = 12_000
    symbol_rate = int(rng.choice([50, 100, 200]))
    bits = "".join(str(int(value)) for value in rng.integers(0, 2, size=512))
    common = {
        "sample_rate": sample_rate,
        "symbol_rate": symbol_rate,
        "carrier_hz": float(rng.choice([900, 1200, 1500, 1800, 2100])),
        "amplitude": float(rng.uniform(0.35, 0.7)),
        "initial_phase_radians": float(rng.uniform(-np.pi, np.pi)),
    }
    if signal_type == "ook":
        return generate_ook_wav(path, bits, **common)
    if signal_type == "ask":
        symbols = [int(value) for value in rng.integers(0, 4, size=len(bits) // 2)]
        return generate_ask_wav(path, symbols, level_count=4, **common)
    if signal_type == "ppm":
        return generate_ppm_wav(
            path, bits, pulse_fraction=float(rng.choice([0.1, 0.2, 0.3])), **common,
        )
    raise ValueError(f"unsupported CV signal type: {signal_type}")


def _existing_record(store: CaptureStore, waveform: Path) -> dict[str, Any] | None:
    digest = hashlib.sha256(waveform.read_bytes()).hexdigest()
    capture = store.capture_by_sha256(digest)
    if capture is None:
        return None
    if capture["source_kind"] != "synthetic":
        raise ValueError("CV source waveform hash collides with a non-synthetic capture")
    return {"capture_id": capture["id"], "sha256": digest}


def build_cv_signal_corpus(
    output_dir: str | Path,
    data_dir: str | Path,
    *,
    examples_per_type: int = 20,
    seed: int = 20261023,
) -> dict[str, Any]:
    """Generate/import deterministic OOK, 4-ASK and PPM signal sources.

    This CV-only corpus adds visual signal diversity without changing the
    application's known-modulation classifier classes. Each clean source is
    imported with its validated generator manifest and accepted signal region.
    """
    if (
        not isinstance(examples_per_type, int) or isinstance(examples_per_type, bool)
        or not 1 <= examples_per_type <= MAX_EXAMPLES_PER_TYPE
    ):
        raise ValueError(f"examples_per_type must be between 1 and {MAX_EXAMPLES_PER_TYPE}")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")

    output = Path(output_dir).expanduser().resolve()
    data = Path(data_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    store = CaptureStore(data / "dataset.sqlite3")
    index_path = output / "cv-signals.json"
    config = {
        "schema_version": 1,
        "generator": "sig-id-build-cv-signal-corpus",
        "generator_version": "0.1.0",
        "seed": seed,
        "examples_per_type": examples_per_type,
        "types": list(TYPES),
    }
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("config") != config:
            raise ValueError("existing CV signal index uses a different configuration; choose another output directory")
    else:
        if any(output.iterdir()):
            raise FileExistsError(f"output directory must be empty: {output}")
        index = {"config": config, "records": []}
        _atomic_json(index_path, index)

    records = {record["key"]: record for record in index.get("records", [])}
    stale_keys = []
    for key, record in records.items():
        try:
            capture = store.capture(record["capture_id"])
        except (KeyError, TypeError):
            stale_keys.append(key)
            continue
        if capture["sha256"] != record.get("sha256"):
            raise ValueError(f"CV signal index hash disagrees with capture {record['capture_id']}")
    for key in stale_keys:
        del records[key]

    for signal_type in TYPES:
        for item_index in range(examples_per_type):
            key = f"{signal_type}/{item_index:04d}"
            waveform = output / "sources" / signal_type / f"{item_index:04d}.wav"
            manifest = waveform.with_suffix(waveform.suffix + ".json")
            if not waveform.is_file() or not manifest.is_file():
                _generate_source(waveform, signal_type, _source_seed(seed, signal_type, item_index))
            if key in records:
                continue
            existing = _existing_record(store, waveform)
            if existing is None:
                imported = import_synthetic_wav(
                    waveform, manifest, store=store, capture_dir=data / "captures",
                )
                existing = {
                    "capture_id": imported["capture"]["id"],
                    "sha256": imported["capture"]["sha256"],
                }
            records[key] = {"key": key, "type": signal_type, **existing}
            index["records"] = list(records.values())
            _atomic_json(index_path, index)

    result = {
        "index": str(index_path),
        "record_count": len(records),
        "examples_per_type": examples_per_type,
        "types": list(TYPES),
        "classifier_classes_unchanged": True,
    }
    index["record_count"] = len(records)
    _atomic_json(index_path, index)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="generated WAVs and source index")
    parser.add_argument("--data-dir", type=Path, default=Path("~/.local/share/sig-id").expanduser())
    parser.add_argument("--examples-per-type", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    try:
        report = build_cv_signal_corpus(
            args.output_dir, args.data_dir,
            examples_per_type=args.examples_per_type, seed=args.seed,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
