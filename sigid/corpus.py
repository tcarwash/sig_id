"""Small, deterministic synthetic corpus builder with resumable local imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .augmentation import generate_awgn_augmentation
from .dataset import CaptureStore
from .synthetic import generate_2fsk_wav, generate_bpsk_wav, generate_mfsk_wav, generate_qpsk_wav
from .synthetic_dataset import import_synthetic_wav


MODES = ("2fsk", "4fsk", "8fsk", "16fsk", "bpsk", "qpsk")
MAX_EXAMPLES_PER_MODE = 100
MAX_SNR_LEVELS = 8
MAX_CORPUS_RECORDS = 5000


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
        temporary = Path(file.name)
    temporary.replace(path)


def _source_seed(seed: int, mode: str, example_index: int) -> int:
    token = f"{seed}:{mode}:{example_index}".encode()
    offset = int.from_bytes(hashlib.sha256(token).digest()[:4], "little")
    return (seed + offset) % (2**32)


def _generate_source(path: Path, mode: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sample_rate = 12_000
    symbol_rate = int(rng.choice([50, 100, 200]))
    if mode == "2fsk":
        bits = "".join(str(int(value)) for value in rng.integers(0, 2, size=256))
        low = int(rng.choice([600, 800, 1000]))
        return generate_2fsk_wav(
            path,
            bits,
            sample_rate=sample_rate,
            symbol_rate=symbol_rate,
            tone_low_hz=low,
            tone_high_hz=low + 300,
            amplitude=float(rng.uniform(0.3, 0.7)),
        )
    if mode in {"4fsk", "8fsk", "16fsk"}:
        tone_count = int(mode[:-3])
        symbols = [int(value) for value in rng.integers(0, tone_count, size=256)]
        spacing = 100.0
        base = float(rng.choice([400, 600, 800]))
        return generate_mfsk_wav(
            path,
            symbols,
            tone_count=tone_count,
            sample_rate=sample_rate,
            symbol_rate=symbol_rate,
            base_tone_hz=base,
            tone_spacing_hz=spacing,
            amplitude=float(rng.uniform(0.3, 0.7)),
        )
    bits = "".join(str(int(value)) for value in rng.integers(0, 2, size=512 if mode == "qpsk" else 256))
    common = {
        "sample_rate": sample_rate,
        "symbol_rate": symbol_rate,
        "carrier_hz": float(rng.choice([1200, 1500, 1800, 2100])),
        "amplitude": float(rng.uniform(0.3, 0.7)),
        "initial_phase_radians": float(rng.uniform(-np.pi, np.pi)),
    }
    return generate_bpsk_wav(path, bits, **common) if mode == "bpsk" else generate_qpsk_wav(path, bits, **common)


def _existing_record(store: CaptureStore, waveform: Path, record_kind: str) -> dict[str, Any] | None:
    digest = hashlib.sha256(waveform.read_bytes()).hexdigest()
    capture = store.capture_by_sha256(digest)
    if capture is None:
        return None
    if capture["source_kind"] != "synthetic":
        raise ValueError("corpus waveform hash collides with a non-synthetic capture")
    return {
        "capture_id": capture["id"],
        "sha256": digest,
        "parent_capture_id": capture["parent_capture_id"],
        "kind": record_kind,
    }


def build_corpus(
    output_dir: str | Path,
    data_dir: str | Path,
    *,
    examples_per_mode: int = 1,
    snr_levels_db: list[float] | tuple[float, ...] = (0.0, 10.0, 20.0),
    seed: int = 20261021,
) -> dict[str, Any]:
    """Generate/import six modulation methods and deterministic AWGN variants.

    The JSONL-like index is rewritten atomically after each imported item so a
    stopped batch can resume with the same configuration without duplicate rows.
    The source capture ID is the lineage group key for later split generation.
    """
    if not isinstance(examples_per_mode, int) or isinstance(examples_per_mode, bool) or not 1 <= examples_per_mode <= MAX_EXAMPLES_PER_MODE:
        raise ValueError(f"examples_per_mode must be between 1 and {MAX_EXAMPLES_PER_MODE}")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    snr_values = [float(value) for value in snr_levels_db]
    if len(snr_values) > MAX_SNR_LEVELS or len(set(snr_values)) != len(snr_values):
        raise ValueError(f"provide at most {MAX_SNR_LEVELS} distinct SNR levels")
    if any(not np.isfinite(value) or not -10 <= value <= 80 for value in snr_values):
        raise ValueError("all SNR levels must be between -10 and 80 dB")
    record_limit = len(MODES) * examples_per_mode * (1 + len(snr_values))
    if record_limit > MAX_CORPUS_RECORDS:
        raise ValueError("requested corpus exceeds the configured record limit")

    output_dir = Path(output_dir).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_dir = data_dir / "captures"
    store = CaptureStore(data_dir / "dataset.sqlite3")
    index_path = output_dir / "corpus.json"
    config = {
        "schema_version": 1,
        "generator": "sig-id-build-corpus",
        "generator_version": "0.1.0",
        "seed": seed,
        "examples_per_mode": examples_per_mode,
        "modes": list(MODES),
        "snr_levels_db": snr_values,
        "max_records": record_limit,
    }
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("config") != config:
            raise ValueError("existing corpus index uses a different configuration; choose another output directory")
    else:
        index = {"config": config, "records": []}
        _atomic_json(index_path, index)
    records_by_key = {record["key"]: record for record in index.get("records", [])}
    stale_keys = []
    for key, record in records_by_key.items():
        try:
            capture = store.capture(record["capture_id"])
        except (KeyError, TypeError):
            stale_keys.append(key)
            continue
        if capture["sha256"] != record.get("sha256"):
            raise ValueError(f"corpus index hash disagrees with capture {record['capture_id']}")
        expected_parent = record.get("parent_capture_id")
        if capture["parent_capture_id"] != expected_parent:
            raise ValueError(f"corpus index lineage disagrees with capture {record['capture_id']}")
    for key in stale_keys:
        del records_by_key[key]
    if stale_keys:
        index["records"] = list(records_by_key.values())
        _atomic_json(index_path, index)

    for mode_index, mode in enumerate(MODES):
        for example_index in range(examples_per_mode):
            source_key = f"{mode}/{example_index:04d}/clean"
            source_path = output_dir / "sources" / mode / f"{example_index:04d}.wav"
            source_manifest_path = source_path.with_suffix(source_path.suffix + ".json")
            if not source_path.exists() or not source_manifest_path.exists():
                _generate_source(source_path, mode, _source_seed(seed, mode, example_index))
            clean_record = records_by_key.get(source_key)
            if clean_record is None:
                clean_record = _existing_record(store, source_path, "clean")
                if clean_record is None:
                    imported = import_synthetic_wav(
                        source_path,
                        source_manifest_path,
                        store=store,
                        capture_dir=capture_dir,
                    )
                    clean_record = {
                        "capture_id": imported["capture"]["id"],
                        "sha256": imported["capture"]["sha256"],
                        "parent_capture_id": None,
                        "kind": "clean",
                    }
                clean_record = {
                    "key": source_key,
                    "mode": mode,
                    "lineage_group": clean_record["capture_id"],
                    **clean_record,
                }
                records_by_key[source_key] = clean_record
                index["records"] = list(records_by_key.values())
                _atomic_json(index_path, index)

            for snr_index, snr_db in enumerate(snr_values):
                child_key = f"{mode}/{example_index:04d}/snr-{snr_db:g}"
                child_path = output_dir / "awgn" / mode / f"{example_index:04d}-snr-{snr_db:g}.wav"
                child_manifest_path = child_path.with_suffix(child_path.suffix + ".json")
                if not child_path.exists() or not child_manifest_path.exists():
                    child_seed = _source_seed(seed ^ 0xA5A5A5A5, mode, example_index * MAX_SNR_LEVELS + snr_index)
                    generate_awgn_augmentation(
                        source_path,
                        source_manifest_path,
                        child_path,
                        snr_db=snr_db,
                        seed=child_seed,
                    )
                if child_key in records_by_key:
                    continue
                child_record = _existing_record(store, child_path, "awgn")
                if child_record is None:
                    imported = import_synthetic_wav(
                        child_path,
                        child_manifest_path,
                        store=store,
                        capture_dir=capture_dir,
                    )
                    child_record = {
                        "capture_id": imported["capture"]["id"],
                        "sha256": imported["capture"]["sha256"],
                        "parent_capture_id": imported["capture"]["parent_capture_id"],
                        "kind": "awgn",
                    }
                child_record = {
                    "key": child_key,
                    "mode": mode,
                    "snr_db": snr_db,
                    "lineage_group": clean_record["capture_id"],
                    **child_record,
                }
                records_by_key[child_key] = child_record
                index["records"] = list(records_by_key.values())
                _atomic_json(index_path, index)

    index["records"] = list(records_by_key.values())
    index["record_count"] = len(index["records"])
    _atomic_json(index_path, index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and import a deterministic modulation corpus")
    parser.add_argument("--output-dir", type=Path, required=True, help="generated WAVs and corpus index")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("SIG_ID_DATA_DIR", "~/.local/share/sig-id")).expanduser())
    parser.add_argument("--examples-per-mode", type=int, default=1)
    parser.add_argument("--snr-db", type=float, nargs="*", default=[0.0, 10.0, 20.0])
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    try:
        index = build_corpus(
            args.output_dir,
            args.data_dir,
            examples_per_mode=args.examples_per_mode,
            snr_levels_db=args.snr_db,
            seed=args.seed,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"index": str(args.output_dir / "corpus.json"), "record_count": index["record_count"]}, indent=2))


if __name__ == "__main__":
    main()
