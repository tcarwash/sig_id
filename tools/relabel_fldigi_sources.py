"""Migrate accepted Fldigi source WAVs to measured-band v0.2.0 manifests.

This is a CREATE-ONLY operation on a separate target dataset. It never
changes old source audio, manifests, labels or training/validation/test sets.
New target captures are checked by the ordinary manifest validator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sigid.dataset import CaptureStore
from sigid.synthetic_dataset import _validate_manifest, import_synthetic_wav
from sigid.synthetic_quality import measure_occupied_band


def migrate_fldigi_sources(
    source_data_dir: str | Path, target_data_dir: str | Path, *,
    limit: int | None = None,
) -> dict:
    source_root = Path(source_data_dir).expanduser().resolve()
    target_root = Path(target_data_dir).expanduser().resolve()
    if source_root == target_root:
        raise ValueError("migration target must be a different dataset root")
    if not (source_root / "dataset.sqlite3").is_file():
        raise FileNotFoundError("source dataset does not exist")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    source_store = CaptureStore(source_root / "dataset.sqlite3")
    target_root.mkdir(parents=True, exist_ok=True)
    target_store = CaptureStore(target_root / "dataset.sqlite3")
    sources = sorted((
        item for item in source_store.list_captures()
        if item["source_kind"] == "synthetic"
        and item.get("parent_capture_id") is None
        and item["source_metadata"].get("generator") == "sig-id-fldigi"
    ), key=lambda item: item["sha256"])
    if limit is not None:
        sources = sources[:limit]

    imported = already = 0
    widths_before: list[float] = []
    widths_after: list[float] = []
    source_ids: list[str] = []
    skipped: list[dict] = []
    for source in sources:
        sha = source["sha256"]
        original_wav = Path(source["audio_path"])
        original_sidecar = original_wav.with_suffix(original_wav.suffix + ".json")
        try:
            manifest = json.loads(original_sidecar.read_text(encoding="utf-8"))
            _validate_manifest(manifest, original_wav)
            if manifest["generator"] != "sig-id-fldigi":
                raise ValueError("source manifest does not describe a Fldigi capture")
            if hashlib.sha256(original_wav.read_bytes()).hexdigest() != sha:
                raise ValueError("source waveform differs from its indexed digest")
            regions = source_store.regions(source["id"])
            if len(regions) != 1 or regions[0]["source"] != "synthetic" or regions[0]["review_state"] != "accepted":
                raise ValueError("source does not have one accepted synthetic label")
            measured = measure_occupied_band(original_wav)
            previous = float(regions[0]["frequency_high_hz"]) - float(regions[0]["frequency_low_hz"])
            new_width = measured["frequency_high_hz"] - measured["frequency_low_hz"]
            if new_width <= 0 or new_width >= 3000:
                raise ValueError("source occupied-band estimate is not usable")
            existing = target_store.capture_by_sha256(sha)
            if existing is not None:
                if existing["source_metadata"].get("generator_version") != "0.2.0":
                    raise ValueError("target SHA already exists with incompatible source-manifest version")
                already += 1
            else:
                updated = {**manifest, "generator_version": "0.2.0"}
                updated["parameters"] = {
                    **manifest["parameters"], "band_measurement": measured,
                    "frequency_low_hz": measured["frequency_low_hz"],
                    "frequency_high_hz": measured["frequency_high_hz"],
                }
                updated["source_manifest_sha256"] = hashlib.sha256(original_sidecar.read_bytes()).hexdigest()
                sidecar = target_root / "source-manifests" / f"{sha}.wav.json"
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                sidecar.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                imported_capture = import_synthetic_wav(
                    original_wav, sidecar, store=target_store, capture_dir=target_root / "captures",
                )["capture"]
                if imported_capture["sha256"] != sha:
                    raise AssertionError("re-imported waveform digest changed")
                imported += 1
            source_ids.append(sha)
            widths_before.append(previous)
            widths_after.append(new_width)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            skipped.append({"source_sha256": sha, "reason": str(exc)})
    return {
        "source_count": len(sources),
        "imported_count": imported,
        "already_imported_count": already,
        "skipped_count": len(skipped),
        "skipped": skipped[:25],
        "source_hashes": source_ids,
        "mean_original_bandwidth_hz": round(sum(widths_before) / max(1, len(widths_before)), 2),
        "mean_measured_bandwidth_hz": round(sum(widths_after) / max(1, len(widths_after)), 2),
        "source_data_dir": str(source_root),
        "new_data_dir": str(target_root),
        "source_modification": False,
        "note": "Historical waveforms retained byte-for-byte. Updated labels are spectrally estimated, not independently verified payload decodes.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument("--target-data-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = migrate_fldigi_sources(args.source_data_dir, args.target_data_dir, limit=args.limit)
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(data, encoding="utf-8")
    print(data, end="")


if __name__ == "__main__":
    main()
