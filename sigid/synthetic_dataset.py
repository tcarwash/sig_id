"""Validate generated WAV manifests and register their labeled data in the store."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from uuid import uuid4

from .dataset import CaptureStore, SignalLabel
from .wav import WavSource


MAX_MANIFEST_BYTES = 32 * 1024 * 1024


def _read_manifest(path: Path) -> tuple[dict[str, Any], str]:
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("synthetic manifest exceeds the 32 MiB limit")
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid synthetic manifest JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("synthetic manifest must contain a JSON object")
    return manifest, hashlib.sha256(raw).hexdigest()


def _validate_manifest(manifest: dict[str, Any], wav_path: Path) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("source_kind") != "synthetic":
        raise ValueError("unsupported or non-synthetic manifest")
    if manifest.get("deterministic") is not True:
        raise ValueError("only deterministic synthetic sources can be imported as generated ground truth")
    generator = manifest.get("generator")
    labels = manifest.get("labels")
    parameters = manifest.get("parameters")
    if not isinstance(labels, dict) or not isinstance(parameters, dict):
        raise ValueError("synthetic manifest must include labels and parameters objects")
    if labels.get("protocol") is not None or labels.get("variant") is not None:
        raise ValueError("current generic modulation generators cannot claim a protocol or variant")

    if generator == "sig-id-2fsk":
        if manifest.get("generator_version") != "0.1.0":
            raise ValueError("unsupported 2-FSK generator version")
        expected_family = "FSK"
        expected_method = "binary frequency shift keying"
        bits = manifest.get("reference_bits")
        if not isinstance(bits, str) or not bits or any(bit not in "01" for bit in bits):
            raise ValueError("2-FSK manifest must contain exact reference bits")
        sample_rate = manifest.get("sample_rate")
        symbol_rate = parameters.get("symbol_rate")
        if (
            parameters.get("tone_count") != 2
            or not isinstance(sample_rate, int)
            or not isinstance(symbol_rate, int)
            or symbol_rate <= 0
            or sample_rate % symbol_rate
            or len(bits) * (sample_rate // symbol_rate) != manifest.get("sample_count")
        ):
            raise ValueError("2-FSK reference bits do not match the waveform dimensions")
    elif generator == "sig-id-mfsk":
        if manifest.get("generator_version") != "0.2.0":
            raise ValueError("unsupported MFSK generator version")
        tone_count = parameters.get("tone_count")
        if not isinstance(tone_count, int) or isinstance(tone_count, bool) or not 2 <= tone_count <= 64:
            raise ValueError("MFSK manifest has an invalid tone count")
        expected_family = "FSK" if tone_count == 2 else "MFSK"
        expected_method = "binary frequency shift keying" if tone_count == 2 else f"{tone_count}-ary frequency shift keying"
        symbols = manifest.get("reference_symbols")
        if not isinstance(symbols, list) or not symbols:
            raise ValueError("MFSK manifest must contain exact reference symbols")
        if any(not isinstance(symbol, int) or isinstance(symbol, bool) or not 0 <= symbol < tone_count for symbol in symbols):
            raise ValueError("MFSK reference symbols must be valid tone indices")
        sample_rate = manifest.get("sample_rate")
        symbol_rate = parameters.get("symbol_rate")
        if not isinstance(sample_rate, int) or not isinstance(symbol_rate, int) or symbol_rate <= 0 or sample_rate % symbol_rate:
            raise ValueError("MFSK manifest has invalid timing parameters")
        samples_per_symbol = sample_rate // symbol_rate
        if manifest.get("sample_count") != len(symbols) * samples_per_symbol:
            raise ValueError("MFSK reference symbols do not match the waveform dimensions")
        bits_per_symbol = parameters.get("bits_per_symbol")
        expected_bits_per_symbol = tone_count.bit_length() - 1 if tone_count & (tone_count - 1) == 0 else None
        expected_bit_order = "most-significant bit first" if expected_bits_per_symbol is not None else None
        if (
            bits_per_symbol != expected_bits_per_symbol
            or isinstance(bits_per_symbol, bool)
            or parameters.get("bit_order") != expected_bit_order
        ):
            raise ValueError("MFSK bit mapping metadata does not match the tone count")
        expected_bits = None
        if expected_bits_per_symbol is not None:
            expected_bits = "".join(f"{symbol:0{bits_per_symbol}b}" for symbol in symbols)
        if manifest.get("reference_bits") != expected_bits:
            raise ValueError("MFSK reference bits do not match the declared symbol mapping")
    elif generator == "sig-id-bpsk":
        if manifest.get("generator_version") != "0.1.0":
            raise ValueError("unsupported BPSK generator version")
        expected_family = "PSK"
        expected_method = "binary phase-shift keying"
        bits = manifest.get("reference_bits")
        if not isinstance(bits, str) or not bits or any(bit not in "01" for bit in bits):
            raise ValueError("BPSK manifest must contain exact reference bits")
        sample_rate = manifest.get("sample_rate")
        symbol_rate = parameters.get("symbol_rate")
        if (
            not isinstance(sample_rate, int)
            or isinstance(sample_rate, bool)
            or not isinstance(symbol_rate, int)
            or isinstance(symbol_rate, bool)
            or symbol_rate <= 0
            or sample_rate % symbol_rate
            or len(bits) * (sample_rate // symbol_rate) != manifest.get("sample_count")
        ):
            raise ValueError("BPSK reference bits do not match the waveform dimensions")
    elif generator == "sig-id-qpsk":
        if manifest.get("generator_version") != "0.1.0":
            raise ValueError("unsupported QPSK generator version")
        expected_family = "PSK"
        expected_method = "quadrature phase-shift keying"
        bits = manifest.get("reference_bits")
        if not isinstance(bits, str) or not bits or len(bits) % 2 or any(bit not in "01" for bit in bits):
            raise ValueError("QPSK manifest must contain an even-length exact reference bit sequence")
        sample_rate = manifest.get("sample_rate")
        symbol_rate = parameters.get("symbol_rate")
        if (
            not isinstance(sample_rate, int)
            or isinstance(sample_rate, bool)
            or not isinstance(symbol_rate, int)
            or isinstance(symbol_rate, bool)
            or symbol_rate <= 0
            or sample_rate % symbol_rate
            or len(bits) // 2 * (sample_rate // symbol_rate) != manifest.get("sample_count")
        ):
            raise ValueError("QPSK reference bits do not match the waveform dimensions")
        if manifest.get("reference_symbols") != [bits[index : index + 2] for index in range(0, len(bits), 2)]:
            raise ValueError("QPSK symbols do not match the reference bit mapping")
    else:
        raise ValueError(f"unsupported synthetic generator: {generator!r}")

    if labels.get("modulation_family") != expected_family or labels.get("modulation_method") != expected_method:
        raise ValueError("synthetic modulation labels do not match the generator parameters")
    audio = WavSource(wav_path).metadata()
    if (
        manifest.get("sample_rate") != audio["sample_rate"]
        or manifest.get("sample_count") != audio["sample_count"]
        or manifest.get("channels") != audio["channels"]
        or manifest.get("sample_width_bytes") != audio["sample_width_bytes"]
    ):
        raise ValueError("synthetic manifest audio dimensions do not match the WAV")

    # Recreate from the reference symbols with the shipped deterministic
    # generator. This prevents a hand-edited sidecar from turning arbitrary
    # audio into accepted synthetic ground truth.
    from .synthetic import generate_2fsk_wav, generate_bpsk_wav, generate_mfsk_wav, generate_qpsk_wav

    try:
        with tempfile.TemporaryDirectory(prefix="sig-id-validate-") as temporary:
            regenerated = Path(temporary) / "check.wav"
            if generator == "sig-id-2fsk":
                generate_2fsk_wav(
                    regenerated,
                    manifest["reference_bits"],
                    sample_rate=manifest["sample_rate"],
                    symbol_rate=parameters["symbol_rate"],
                    tone_low_hz=parameters["tone_low_hz"],
                    tone_high_hz=parameters["tone_high_hz"],
                    amplitude=parameters["amplitude"],
                )
            elif generator == "sig-id-mfsk":
                generate_mfsk_wav(
                    regenerated,
                    manifest["reference_symbols"],
                    tone_count=parameters["tone_count"],
                    sample_rate=manifest["sample_rate"],
                    symbol_rate=parameters["symbol_rate"],
                    base_tone_hz=parameters["base_tone_hz"],
                    tone_spacing_hz=parameters["tone_spacing_hz"],
                    amplitude=parameters["amplitude"],
                )
            elif generator == "sig-id-bpsk":
                generate_bpsk_wav(
                    regenerated,
                    manifest["reference_bits"],
                    sample_rate=manifest["sample_rate"],
                    symbol_rate=parameters["symbol_rate"],
                    carrier_hz=parameters["carrier_hz"],
                    amplitude=parameters["amplitude"],
                    initial_phase_radians=parameters["initial_phase_radians"],
                )
            else:
                generate_qpsk_wav(
                    regenerated,
                    manifest["reference_bits"],
                    sample_rate=manifest["sample_rate"],
                    symbol_rate=parameters["symbol_rate"],
                    carrier_hz=parameters["carrier_hz"],
                    amplitude=parameters["amplitude"],
                    initial_phase_radians=parameters["initial_phase_radians"],
                )
            if regenerated.read_bytes() != wav_path.read_bytes():
                raise ValueError("WAV samples do not match the synthetic generator manifest")
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"invalid synthetic generator parameters: {exc}") from exc


def import_synthetic_wav(
    wav_path: str | Path,
    manifest_path: str | Path,
    *,
    store: CaptureStore,
    capture_dir: str | Path,
) -> dict[str, Any]:
    """Copy and register a generated WAV plus its exact sidecar manifest.

    Imports create one full-capture region with the manifest's modulation labels.
    No protocol label is inferred. The original trace remains in the sidecar file
    and is linked by its SHA-256 digest in capture metadata and label evidence.
    """
    source_audio = Path(wav_path).resolve(strict=True)
    source_manifest = Path(manifest_path).resolve(strict=True)
    manifest, manifest_digest = _read_manifest(source_manifest)
    try:
        _validate_manifest(manifest, source_audio)
    except ValueError:
        raise
    except (TypeError, KeyError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid synthetic manifest values: {exc}") from exc

    capture_dir = Path(capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)
    import_id = str(uuid4())
    destination_audio = capture_dir / f"synthetic-{import_id}.wav"
    destination_manifest = destination_audio.with_suffix(destination_audio.suffix + ".json")
    capture_id: str | None = None
    try:
        shutil.copyfile(source_audio, destination_audio)
        shutil.copyfile(source_manifest, destination_manifest)
        metadata = {
            "generator": manifest["generator"],
            "generator_version": manifest.get("generator_version"),
            "manifest_file": destination_manifest.name,
            "manifest_sha256": manifest_digest,
            "modulation_labels": manifest["labels"],
            "parameters": manifest["parameters"],
        }
        capture_id = store.add_wav_capture(
            destination_audio,
            source_kind="synthetic",
            source_metadata=metadata,
        )
        region_id = store.add_region(
            capture_id,
            sample_start=0,
            sample_count=int(manifest["sample_count"]),
            label=SignalLabel(
                family=manifest["labels"]["modulation_family"],
                method=manifest["labels"]["modulation_method"],
                protocol=None,
                variant=None,
                source="synthetic",
                review_state="accepted",
                evidence={
                    "generator": manifest["generator"],
                    "manifest_file": destination_manifest.name,
                    "manifest_sha256": manifest_digest,
                },
                notes="Controlled synthetic modulation source; protocol is intentionally unknown.",
            ),
        )
    except Exception:
        if capture_id is not None:
            store.delete_capture(capture_id)
        destination_audio.unlink(missing_ok=True)
        destination_manifest.unlink(missing_ok=True)
        raise

    return {
        "capture": store.capture(capture_id),
        "region": next(region for region in store.regions(capture_id) if region["id"] == region_id),
        "manifest_path": str(destination_manifest),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Import a generated WAV and sidecar into the local signal dataset")
    parser.add_argument("waveform", type=Path)
    parser.add_argument("--manifest", type=Path, help="sidecar path (defaults to WAVEFORM.wav.json)")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("SIG_ID_DATA_DIR", "~/.local/share/sig-id")).expanduser(),
    )
    args = parser.parse_args()
    manifest_path = args.manifest or args.waveform.with_suffix(args.waveform.suffix + ".json")
    try:
        result = import_synthetic_wav(
            args.waveform,
            manifest_path,
            store=CaptureStore(args.data_dir / "dataset.sqlite3"),
            capture_dir=args.data_dir / "captures",
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"capture": result["capture"], "region": result["region"]}, indent=2))
