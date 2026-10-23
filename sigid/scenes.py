"""Deterministic mixtures of accepted synthetic captures for scene evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import wave
from typing import Any
from uuid import uuid4

import numpy as np

from .dataset import CaptureStore, SignalLabel


MAX_COMPONENTS = 8
MAX_SCENE_SECONDS = 10
GENERATOR = "sig-id-mixed-scene"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_mono_pcm16(path: Path, start_sample: int = 0, sample_count: int | None = None) -> tuple[int, int, np.ndarray]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getcomptype() != "NONE":
            raise ValueError("scene components must be uncompressed mono 16-bit PCM WAVs")
        rate = source.getframerate()
        total_frames = source.getnframes()
        count = total_frames - start_sample if sample_count is None else sample_count
        if start_sample < 0 or count <= 0 or start_sample + count > total_frames:
            raise ValueError("scene component sample bounds are invalid")
        if count > rate * MAX_SCENE_SECONDS:
            raise ValueError(f"scene component exceeds the {MAX_SCENE_SECONDS}-second limit")
        source.setpos(start_sample)
        pcm = source.readframes(count)
    return rate, total_frames, np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0


def _source_component(store: CaptureStore, specification: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], int, int, float, int, np.ndarray, float, float, float | None, float, int]:
    if not isinstance(specification, dict):
        raise ValueError("scene components must be objects")
    capture_id = specification.get("capture_id")
    if not isinstance(capture_id, str) or not capture_id:
        raise ValueError("each component requires a capture_id")
    capture = store.capture(capture_id)
    if capture["source_kind"] != "synthetic":
        raise ValueError("mixed-scene components must be accepted synthetic captures")
    regions = store.regions(capture_id)
    if len(regions) != 1:
        raise ValueError("each component capture must have exactly one accepted full-capture label")
    region = regions[0]
    if (
        region["sample_start"] != 0 or region["sample_count"] != capture["sample_count"]
        or region["source"] != "synthetic" or region["review_state"] != "accepted"
    ):
        raise ValueError("each component must have an accepted synthetic full-capture label")
    path = Path(capture["audio_path"])
    if _sha256(path) != capture["sha256"]:
        raise ValueError(f"component capture hash changed: {capture_id}")
    source_start = specification.get("source_start_sample", 0)
    if not isinstance(source_start, int) or isinstance(source_start, bool) or source_start < 0:
        raise ValueError("source_start_sample must be a non-negative integer")
    sample_count = specification.get("sample_count")
    scene_start = specification.get("scene_start_sample", 0)
    if sample_count is None:
        rate, total_frames, _ = _read_mono_pcm16(path, source_start, 1)
        sample_count = total_frames - source_start
    if any(not isinstance(value, int) or isinstance(value, bool) for value in (source_start, sample_count, scene_start)):
        raise ValueError("scene sample positions and lengths must be integers")
    if sample_count <= 0 or scene_start < 0:
        raise ValueError("scene component sample bounds are invalid")
    rate, total_frames, audio = _read_mono_pcm16(path, source_start, sample_count)
    if total_frames != capture["sample_count"] or rate != capture["sample_rate"]:
        raise ValueError("scene component audio dimensions disagree with the dataset")
    gain = specification.get("gain", 1.0)
    if not isinstance(gain, (int, float)) or isinstance(gain, bool) or not np.isfinite(gain) or not 0 < gain <= 4:
        raise ValueError("component gain must be finite and between 0 and 4")
    offset = specification.get("frequency_offset_hz", 0.0)
    drift = specification.get("drift_hz_per_second", 0.0)
    clock_error = specification.get("clock_error_ppm", 0.0)
    if not isinstance(clock_error, (int, float)) or isinstance(clock_error, bool) or not np.isfinite(clock_error) or not -5000 <= clock_error <= 5000:
        raise ValueError("component clock_error_ppm must be finite and between -5000 and 5000")
    clock_scale = 1.0 + float(clock_error) * 1e-6
    rendered_count = max(1, int(round(sample_count / clock_scale)))
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) for value in (offset, drift)):
        raise ValueError("component frequency offset and drift must be finite numbers")
    maximum_shift = abs(float(offset)) + abs(float(drift)) * rendered_count / rate
    if maximum_shift > rate / 4:
        raise ValueError("component frequency offset and drift exceed the supported quarter-rate limit")
    lowpass_hz = specification.get("lowpass_hz")
    if lowpass_hz is not None and (
        not isinstance(lowpass_hz, (int, float)) or isinstance(lowpass_hz, bool)
        or not np.isfinite(lowpass_hz) or not 100 <= lowpass_hz < rate / 2
    ):
        raise ValueError("component lowpass_hz must be finite and between 100 Hz and Nyquist")
    return (capture, region, source_start, sample_count, float(gain), scene_start, audio,
            float(offset), float(drift), None if lowpass_hz is None else float(lowpass_hz), float(clock_error), rendered_count)


def _frequency_shift(audio: np.ndarray, sample_rate: int, offset_hz: float, drift_hz_per_second: float) -> np.ndarray:
    if offset_hz == 0 and drift_hz_per_second == 0:
        return audio
    count = audio.size
    spectrum = np.fft.fft(audio)
    analytic_mask = np.zeros(count, dtype=np.float64)
    analytic_mask[0] = 1.0
    if count % 2 == 0:
        analytic_mask[1 : count // 2] = 2.0
        analytic_mask[count // 2] = 1.0
    else:
        analytic_mask[1 : (count + 1) // 2] = 2.0
    analytic = np.fft.ifft(spectrum * analytic_mask)
    time_s = np.arange(count, dtype=np.float64) / sample_rate
    phase = 2.0 * np.pi * (offset_hz * time_s + 0.5 * drift_hz_per_second * time_s**2)
    return np.real(analytic * np.exp(1j * phase))


def _lowpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    frequencies = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)
    transition_hz = max(1.0, cutoff_hz * 0.05)
    response = np.ones_like(frequencies)
    response[frequencies >= cutoff_hz + transition_hz] = 0.0
    transition = (frequencies > cutoff_hz - transition_hz) & (frequencies < cutoff_hz + transition_hz)
    response[transition] = 0.5 * (1.0 + np.cos(np.pi * (frequencies[transition] - cutoff_hz + transition_hz) / (2.0 * transition_hz)))
    return np.fft.irfft(np.fft.rfft(audio) * response, n=audio.size)


def _clock_resample(audio: np.ndarray, clock_error_ppm: float, rendered_count: int) -> np.ndarray:
    if clock_error_ppm == 0 and rendered_count == audio.size:
        return audio
    scale = 1.0 + clock_error_ppm * 1e-6
    source_positions = np.arange(audio.size, dtype=np.float64)
    output_positions = np.arange(rendered_count, dtype=np.float64) * scale
    return np.interp(output_positions, source_positions, audio, left=0.0, right=0.0)


def _scene_frequency_bounds(
    region: dict[str, Any], *, sample_rate: int, offset_hz: float,
    drift_hz_per_second: float, sample_count: int,
) -> tuple[float | None, float | None]:
    """Bound the aliased real-signal frequencies after scene offset and drift."""
    low, high = region["frequency_low_hz"], region["frequency_high_hz"]
    if low is None or high is None:
        return None, None
    duration = sample_count / sample_rate
    end_offset = offset_hz + drift_hz_per_second * duration
    shifted_low = float(low) + min(offset_hz, end_offset)
    shifted_high = float(high) + max(offset_hz, end_offset)
    half_rate = sample_rate / 2.0
    candidates = [shifted_low, shifted_high]
    first_boundary = int(np.ceil(shifted_low / half_rate))
    last_boundary = int(np.floor(shifted_high / half_rate))
    candidates.extend(index * half_rate for index in range(first_boundary, last_boundary + 1))
    folded = [
        min(value % sample_rate, sample_rate - (value % sample_rate))
        for value in candidates
    ]
    result_low, result_high = min(folded), max(folded)
    if result_high <= result_low:
        return None, None
    return float(result_low), float(result_high)


def _render(components: list[tuple[dict[str, Any], dict[str, Any], int, int, float, int, np.ndarray, float, float, float | None, float, int]], sample_count: int, sample_rate: int) -> bytes:
    mixed = np.zeros(sample_count, dtype=np.float64)
    for _, _, source_start, source_count, gain, scene_start, audio, offset, drift, lowpass_hz, clock_error, rendered_count in components:
        available = min(rendered_count, sample_count - scene_start)
        if available <= 0:
            continue
        resampled = _clock_resample(audio[:source_count], clock_error, rendered_count)[:available]
        shifted = _frequency_shift(resampled, sample_rate, offset, drift)
        if lowpass_hz is not None:
            shifted = _lowpass(shifted, sample_rate, lowpass_hz)
        mixed[scene_start : scene_start + available] += shifted * gain
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    normalization = max(1.0, peak)
    pcm = np.rint(np.clip(mixed / normalization, -1.0, 1.0) * 32767).astype("<i2")
    return pcm.tobytes()


def create_mixed_scene(
    output: str | Path,
    components: list[dict[str, Any]],
    *,
    store: CaptureStore,
    sample_count: int | None = None,
) -> dict[str, Any]:
    """Mix labeled source captures and preserve exact constituent sample labels."""
    if not isinstance(components, list) or not 2 <= len(components) <= MAX_COMPONENTS:
        raise ValueError(f"a scene must contain between 2 and {MAX_COMPONENTS} components")
    resolved = [_source_component(store, component) for component in components]
    rates = {item[0]["sample_rate"] for item in resolved}
    if len(rates) != 1:
        raise ValueError("mixed-scene components must have the same sample rate")
    sample_rate = int(next(iter(rates)))
    if not 4000 <= sample_rate <= 192000:
        raise ValueError("scene sample rate must be between 4000 and 192000 Hz")
    required_samples = max(item[5] + item[11] for item in resolved)
    count = required_samples if sample_count is None else sample_count
    if not isinstance(count, int) or isinstance(count, bool) or count < required_samples:
        raise ValueError("scene sample_count must cover every component")
    if count > sample_rate * MAX_SCENE_SECONDS:
        raise ValueError(f"scene exceeds the {MAX_SCENE_SECONDS}-second limit")

    output = Path(output)
    output_resolved = output.resolve()
    if any(output_resolved == Path(item[0]["audio_path"]).resolve() for item in resolved):
        raise ValueError("scene output cannot overwrite one of its source captures")
    output.parent.mkdir(parents=True, exist_ok=True)
    pcm = _render(resolved, count, sample_rate)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)

    manifest_components = []
    for capture, region, source_start, component_count, gain, scene_start, _, offset, drift, lowpass_hz, clock_error, rendered_count in resolved:
        manifest_components.append({
            "capture_id": capture["id"],
            "source_sha256": capture["sha256"],
            "source_start_sample": source_start,
            "sample_count": component_count,
            "scene_start_sample": scene_start,
            "gain": gain,
            "frequency_offset_hz": offset,
            "drift_hz_per_second": drift,
            "lowpass_hz": lowpass_hz,
            "clock_error_ppm": clock_error,
            "rendered_sample_count": rendered_count,
            "frequency_low_hz": region["frequency_low_hz"],
            "frequency_high_hz": region["frequency_high_hz"],
            "labels": {
                "modulation_family": region["family"],
                "modulation_method": region["method"],
                "protocol": region["protocol"],
                "variant": region["variant"],
            },
            "region_id": region["id"],
        })
    manifest = {
        "schema_version": 1,
        "source_kind": "synthetic",
        "generator": GENERATOR,
        "generator_version": "0.1.0",
        "deterministic": True,
        "waveform": output.name,
        "waveform_sha256": _sha256(output),
        "sample_rate": sample_rate,
        "sample_count": count,
        "sample_width_bytes": 2,
        "channels": 1,
        "labels": {"modulation_family": None, "modulation_method": None, "protocol": None, "variant": None},
        "parameters": {"component_count": len(resolved), "peak_normalized": True},
        "components": manifest_components,
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def import_mixed_scene(
    waveform: str | Path,
    manifest_path: str | Path,
    *,
    store: CaptureStore,
    capture_dir: str | Path,
) -> dict[str, Any]:
    """Verify a scene against its source captures, then import constituent labels."""
    audio_path = Path(waveform).resolve(strict=True)
    sidecar = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict) or manifest.get("schema_version") != 1
        or manifest.get("generator") != GENERATOR or manifest.get("generator_version") != "0.1.0"
        or manifest.get("deterministic") is not True
        or manifest.get("waveform_sha256") != _sha256(audio_path)
        or manifest.get("labels") != {"modulation_family": None, "modulation_method": None, "protocol": None, "variant": None}
    ):
        raise ValueError("scene manifest does not match a supported deterministic waveform")
    components = manifest.get("components")
    if not isinstance(components, list) or not 2 <= len(components) <= MAX_COMPONENTS:
        raise ValueError("scene manifest has an invalid component list")
    resolved = []
    sample_rate = manifest.get("sample_rate")
    sample_count = manifest.get("sample_count")
    if (
        not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or not 4000 <= sample_rate <= 192000
        or not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count <= 0
        or sample_count > sample_rate * MAX_SCENE_SECONDS
        or manifest.get("parameters") != {"component_count": len(components), "peak_normalized": True}
    ):
        raise ValueError("scene manifest has invalid audio dimensions")
    for component in components:
        capture, region, src, count, gain, start, source_audio, offset, drift, lowpass_hz, clock_error, rendered_count = _source_component(store, component)
        if start + rendered_count > sample_count:
            raise ValueError("scene manifest sample_count truncates a component")
        if capture["sha256"] != component.get("source_sha256"):
            raise ValueError("scene source capture hash no longer matches the manifest")
        source_labels = {
            "modulation_family": region["family"],
            "modulation_method": region["method"],
            "protocol": region["protocol"],
            "variant": region["variant"],
        }
        if (
            component.get("region_id") != region["id"]
            or component.get("labels") != source_labels
            or component.get("frequency_low_hz") != region["frequency_low_hz"]
            or component.get("frequency_high_hz") != region["frequency_high_hz"]
        ):
            raise ValueError("scene component labels disagree with the accepted source region")
        if capture["sample_rate"] != sample_rate:
            raise ValueError("scene component sample rate does not match its manifest")
        if component.get("frequency_offset_hz", 0.0) != offset or component.get("drift_hz_per_second", 0.0) != drift:
            raise ValueError("scene frequency impairment disagrees with its manifest")
        if component.get("lowpass_hz") != lowpass_hz:
            raise ValueError("scene filtering impairment disagrees with its manifest")
        if component.get("clock_error_ppm", 0.0) != clock_error or component.get("rendered_sample_count", count) != rendered_count:
            raise ValueError("scene clock impairment disagrees with its manifest")
        resolved.append((capture, region, src, count, gain, start, source_audio, offset, drift, lowpass_hz, clock_error, rendered_count))
    expected_pcm = _render(resolved, sample_count, sample_rate)
    with wave.open(str(audio_path), "rb") as wav:
        if (
            wav.getnchannels() != 1 or wav.getsampwidth() != 2
            or wav.getframerate() != sample_rate or wav.getnframes() != sample_count
            or wav.readframes(sample_count) != expected_pcm
        ):
            raise ValueError("scene samples do not match the declared source captures and recipe")

    digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    capture_dir = Path(capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)
    import_id = str(uuid4())
    stored_audio = capture_dir / f"synthetic-scene-{import_id}.wav"
    stored_manifest = stored_audio.with_suffix(stored_audio.suffix + ".json")
    capture_id = None
    try:
        shutil.copyfile(audio_path, stored_audio)
        shutil.copyfile(sidecar, stored_manifest)
        capture_id = store.add_wav_capture(
            stored_audio,
            source_kind="synthetic",
            source_metadata={
                "generator": GENERATOR,
                "generator_version": "0.1.0",
                "manifest_file": stored_manifest.name,
                "manifest_sha256": digest,
                "component_capture_ids": [item[0]["id"] for item in resolved],
            },
        )
        region_ids = []
        for component, (source_capture, source_region, _, _, _, start, _, _, _, _, _, rendered_count) in zip(components, resolved):
            component_generator = source_capture["source_metadata"].get("generator")
            frequency_low_hz, frequency_high_hz = _scene_frequency_bounds(
                source_region,
                sample_rate=sample_rate,
                offset_hz=float(component.get("frequency_offset_hz", 0.0)),
                drift_hz_per_second=float(component.get("drift_hz_per_second", 0.0)),
                sample_count=rendered_count,
            )
            region_ids.append(store.add_region(
                capture_id,
                sample_start=start,
                sample_count=rendered_count,
                frequency_low_hz=frequency_low_hz,
                frequency_high_hz=frequency_high_hz,
                label=SignalLabel(
                    family=source_region["family"],
                    method=source_region["method"],
                    protocol=source_region["protocol"],
                    variant=source_region["variant"],
                    source="synthetic",
                    review_state="accepted",
                    evidence={
                        "generator": GENERATOR,
                        "manifest_sha256": digest,
                        "component_capture_id": source_capture["id"],
                        "component_sha256": source_capture["sha256"],
                        "source_region_id": source_region["id"],
                        "component_generator": component_generator,
                        "noise_only": component_generator == "sig-id-noise",
                        "scene_start_sample": start,
                        "scene_sample_count": rendered_count,
                        "frequency_offset_hz": component.get("frequency_offset_hz", 0.0),
                        "drift_hz_per_second": component.get("drift_hz_per_second", 0.0),
                        "lowpass_hz": component.get("lowpass_hz"),
                        "clock_error_ppm": component.get("clock_error_ppm", 0.0),
                    },
                    notes="Constituent label from a validated synthetic source mixed into a deterministic scene.",
                ),
            ))
    except Exception:
        if capture_id is not None:
            store.delete_capture(capture_id)
        stored_audio.unlink(missing_ok=True)
        stored_manifest.unlink(missing_ok=True)
        raise
    return {"capture": store.capture(capture_id), "regions": store.regions(capture_id)}


def create_main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic mixed scene from accepted synthetic captures")
    parser.add_argument("recipe", type=Path, help="JSON containing components and optional sample_count")
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    store = CaptureStore(args.data_dir.expanduser() / "dataset.sqlite3")
    print(json.dumps(create_mixed_scene(args.output, recipe["components"], store=store, sample_count=recipe.get("sample_count")), indent=2))


def import_main() -> None:
    parser = argparse.ArgumentParser(description="Import a verified mixed synthetic scene and its constituent labels")
    parser.add_argument("waveform", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser()
    store = CaptureStore(data_dir / "dataset.sqlite3")
    print(json.dumps(import_mixed_scene(args.waveform, args.manifest, store=store, capture_dir=data_dir / "captures"), indent=2))
