"""Create deterministic burst-in-noise scenes for optional CV detector training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
import wave

import numpy as np

from .cv_detector import MAX_FREQUENCY_HZ, WINDOW_SECONDS
from .cv_dataset import _is_human_verified_noise_sample
from .dataset import CaptureStore, SignalLabel
from .scenes import create_mixed_scene, import_mixed_scene
from .synthetic import generate_noise_wav
from .synthetic_dataset import import_synthetic_wav
from .wav import WavSource


SIGNAL_GENERATORS = {
    "sig-id-2fsk", "sig-id-mfsk", "sig-id-bpsk", "sig-id-qpsk",
    "sig-id-ook", "sig-id-ask", "sig-id-ppm", "sig-id-fldigi",
}
SCHEMA_VERSION = 4
SCENE_RECIPE_VERSION = 2
MAX_NOISE_ONLY_EXAMPLES = 5000
RECORDED_NOISE_SEGMENT_SECONDS = 2
MIN_SIGNAL_SNR_DB = -40.0
MAX_SIGNAL_SNR_DB = -15.0


def _progress(message: str) -> None:
    print(f"[cv-scenes] {message}", file=sys.stderr, flush=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wav_window_rms(path: str | Path, start: int, count: int) -> float:
    """Measure normalized mono PCM16 RMS for one exact sample interval."""
    with wave.open(str(path), "rb") as source:
        if source.getcomptype() != "NONE" or source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("scene SNR measurement requires uncompressed mono PCM16 audio")
        if start < 0 or count <= 0 or start + count > source.getnframes():
            raise ValueError("scene SNR interval is outside the source waveform")
        source.setpos(start)
        pcm = source.readframes(count)
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    if samples.size != count:
        raise ValueError("scene SNR interval was truncated while reading audio")
    return float(np.sqrt(np.mean(samples * samples)))


def _wav_band_rms(
    path: str | Path, start: int, count: int, sample_rate: int,
    low_hz: float, high_hz: float,
) -> float:
    """Measure window-corrected RMS energy inside one frequency band."""
    with wave.open(str(path), "rb") as source:
        if source.getcomptype() != "NONE" or source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("scene SNR measurement requires uncompressed mono PCM16 audio")
        if start < 0 or count <= 0 or start + count > source.getnframes():
            raise ValueError("scene SNR interval is outside the source waveform")
        if source.getframerate() != sample_rate:
            raise ValueError("scene SNR sources must use the same sample rate")
        source.setpos(start)
        pcm = source.readframes(count)
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    if samples.size != count:
        raise ValueError("scene SNR interval was truncated while reading audio")
    frequencies = np.fft.rfftfreq(count, d=1.0 / sample_rate)
    selected = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not selected.any():
        return 0.0
    window = np.hanning(count)
    spectrum = np.fft.rfft(samples * window)
    weights = np.full(spectrum.size, 2.0, dtype=np.float64)
    weights[0] = 1.0
    if count % 2 == 0:
        weights[-1] = 1.0
    power = float(np.sum(np.abs(spectrum[selected]) ** 2 * weights[selected]))
    return float(np.sqrt(power / max(1.0, count * np.sum(window * window))))


def _set_signal_gains_for_snr(
    recipes: list[dict[str, Any]], *, noise_capture: dict[str, Any],
    noise_gain: float, store: CaptureStore,
) -> tuple[float, list[dict[str, Any]]]:
    """Set each active signal to its selected in-band SNR against aligned noise."""
    noise_path = noise_capture["audio_path"]
    sample_rate = int(noise_capture["sample_rate"])
    intervals: list[tuple[dict[str, Any], float, float, float, float, float, float]] = []
    for recipe in recipes:
        source = store.capture(recipe["capture_id"])
        region = store.regions(recipe["capture_id"])[0]
        signal_low = float(region["frequency_low_hz"])
        signal_high = float(region["frequency_high_hz"])
        duration = recipe["sample_count"] / sample_rate
        offset = float(recipe["frequency_offset_hz"])
        drift = float(recipe["drift_hz_per_second"])
        shifted_low = signal_low + min(offset, offset + drift * duration)
        shifted_high = signal_high + max(offset, offset + drift * duration)
        noise_low = max(0.0, shifted_low)
        noise_high = min(sample_rate / 2.0, shifted_high)
        signal_rms = _wav_band_rms(
            source["audio_path"], recipe["source_start_sample"], recipe["sample_count"],
            sample_rate, signal_low, signal_high,
        )
        background_rms = _wav_band_rms(
            noise_path, recipe["scene_start_sample"], recipe["sample_count"],
            sample_rate, noise_low, noise_high,
        )
        if signal_rms <= 0 or background_rms <= 0:
            raise ValueError("cannot target scene SNR for a silent signal or noise interval")
        target_snr_db = float(recipe.get("target_snr_db", -10.0))
        if not MIN_SIGNAL_SNR_DB <= target_snr_db <= MAX_SIGNAL_SNR_DB:
            raise ValueError("scene signal SNR must be between -40 and -15 dB")
        target_ratio = 10.0 ** (target_snr_db / 20.0)
        intervals.append((recipe, signal_rms, background_rms, noise_low, noise_high, target_ratio, target_snr_db))

    # If a particularly quiet recorded transmission would need more than the
    # supported component gain, lower the whole scene level and preserve SNR.
    if intervals:
        max_noise_gain = min(
            4.0 * signal_rms / (background_rms * target_ratio)
            for _, signal_rms, background_rms, _, _, target_ratio, _ in intervals
        )
        noise_gain = min(noise_gain, max_noise_gain)
    measured: list[dict[str, Any]] = []
    for recipe, signal_rms, unscaled_noise_rms, noise_low, noise_high, target_ratio, target_snr_db in intervals:
        background_rms = unscaled_noise_rms * noise_gain
        gain = background_rms * target_ratio / signal_rms
        if not np.isfinite(gain) or gain <= 0 or gain > 4.0 + 1e-9:
            raise ValueError(
                f"required signal gain {gain:.4g} is outside the supported range for {target_snr_db:g} dB SNR"
            )
        # The preceding noise-gain clamp caps the theoretical component gain at
        # 4.0. Floating-point rounding can put the recomputed value a few ULPs
        # above that cap, so clamp that harmless excess instead of aborting a
        # large deterministic scene build.
        gain = min(gain, 4.0)
        recipe["gain"] = float(gain)
        measured.append({
            "capture_id": recipe["capture_id"],
            "signal_gain": float(gain),
            "signal_rms": signal_rms * gain,
            "noise_rms": background_rms,
            "target_snr_db": target_snr_db,
            "measured_snr_db": float(20.0 * np.log10(signal_rms * gain / background_rms)),
            "snr_band_hz": [noise_low, noise_high],
        })
    return noise_gain, measured


def _derived_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _scene_groups(captures: list[dict[str, Any]], seed: int, overlap_rate: float) -> list[list[dict[str, Any]]]:
    by_rate: dict[int, list[dict[str, Any]]] = {}
    for capture in captures:
        by_rate.setdefault(int(capture["sample_rate"]), []).append(capture)
    groups = []
    for sample_rate, sources in sorted(by_rate.items()):
        rng = np.random.default_rng(_derived_seed(seed, f"sample-rate:{sample_rate}"))
        ordered = sorted(sources, key=lambda item: item["sha256"])
        order = rng.permutation(len(ordered)).tolist()
        pending = [ordered[index] for index in order]
        while pending:
            first = pending.pop()
            if pending and rng.random() < overlap_rate:
                groups.append([first, pending.pop()])
            else:
                groups.append([first])
    return groups


def _signal_recipe(
    capture: dict[str, Any], region: dict[str, Any], *, rng: np.random.Generator,
    scene_sample_count: int, sample_rate: int, scene_start_sample: int | None = None,
    requested_sample_count: int | None = None,
) -> dict[str, Any]:
    if requested_sample_count is None:
        max_duration = min(4.0, capture["sample_count"] / sample_rate)
        min_duration = min(0.75, max_duration)
        duration = float(rng.uniform(min_duration, max_duration)) if max_duration > min_duration else max_duration
        count = max(4096, min(capture["sample_count"], int(round(duration * sample_rate))))
    else:
        count = requested_sample_count
    if count > capture["sample_count"]:
        count = capture["sample_count"]
    source_start = int(rng.integers(0, capture["sample_count"] - count + 1))
    maximum_start = scene_sample_count - count
    if scene_start_sample is None:
        scene_start_sample = int(rng.integers(0, maximum_start + 1))

    low, high = region["frequency_low_hz"], region["frequency_high_hz"]
    offsets = [offset for offset in (-200.0, -100.0, 0.0, 100.0, 200.0)
               if low + offset >= 0 and high + offset <= sample_rate / 2]
    offset = float(rng.choice(offsets or [0.0]))
    return {
        "capture_id": capture["id"],
        "source_start_sample": source_start,
        "sample_count": count,
        "scene_start_sample": scene_start_sample,
        "gain": 1.0,
        "frequency_offset_hz": offset,
        "drift_hz_per_second": float(rng.choice([-3.0, -1.5, 0.0, 1.5, 3.0])),
    }


def _nonoverlapping_pair_starts(
    first_count: int, second_count: int, *, scene_sample_count: int,
    sample_rate: int, rng: np.random.Generator,
) -> tuple[int, int]:
    # Leave room for the spectrogram's temporal windowing around each signal.
    minimum_gap = int(round(0.5 * sample_rate))
    free_samples = scene_sample_count - first_count - second_count
    if free_samples < minimum_gap:
        raise ValueError("two signal intervals and their separation do not fit in this scene")
    extra_space = free_samples - minimum_gap
    leading_gap = int(rng.integers(0, extra_space + 1))
    extra_between = int(rng.integers(0, extra_space - leading_gap + 1))
    reverse = bool(rng.integers(0, 2))
    if reverse:
        second_start = leading_gap
        first_start = leading_gap + second_count + minimum_gap + extra_between
    else:
        first_start = leading_gap
        second_start = leading_gap + first_count + minimum_gap + extra_between
    return first_start, second_start


def _ensure_synthetic_review(store: CaptureStore, capture: dict[str, Any], note: str) -> None:
    reviews = store.scene_coverages(capture["id"])
    if any(
        review["task"] == "detection" and review["state"] == "complete"
        and not review["is_stale"] and review["sample_start"] == 0
        and review["sample_count"] == capture["sample_count"]
        and review["frequency_low_hz"] in (None, 0.0)
        and review["frequency_high_hz"] in (None, MAX_FREQUENCY_HZ)
        and not review["ignore_regions"] and not review["uncertain_regions"]
        for review in reviews
    ):
        return
    store.create_scene_coverage(
        capture["id"], sample_start=0, sample_count=capture["sample_count"],
        frequency_low_hz=0.0, frequency_high_hz=MAX_FREQUENCY_HZ,
        task="detection", taxonomy_scope=["signal"], state="complete", notes=note,
    )


def _recorded_noise_candidates(store: CaptureStore, captures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect disjoint two-second slices only from human-verified 10s noise samples."""
    candidates: list[dict[str, Any]] = []
    for capture in captures:
        reviews = store.scene_coverages(capture["id"])
        regions = store.regions(capture["id"])
        if not _is_human_verified_noise_sample(capture, reviews, regions):
            continue
        sample_rate = int(capture["sample_rate"])
        segment_count = sample_rate * RECORDED_NOISE_SEGMENT_SECONDS
        if capture["sample_count"] < segment_count:
            continue
        try:
            metadata = WavSource(capture["audio_path"]).metadata()
        except (OSError, ValueError, wave.Error):
            continue
        if metadata["sample_rate"] != sample_rate or metadata["sample_count"] != capture["sample_count"]:
            continue
        if _file_sha256(capture["audio_path"]) != capture["sha256"]:
            continue
        for start in range(0, int(capture["sample_count"]) - segment_count + 1, segment_count):
            candidates.append({"capture": capture, "sample_start": start, "sample_count": segment_count})
    return candidates


def _recorded_noise_audio(candidate: dict[str, Any], target_rate: int) -> np.ndarray:
    capture = candidate["capture"]
    if _file_sha256(capture["audio_path"]) != capture["sha256"]:
        raise ValueError("recorded-noise source audio no longer matches its saved capture hash")
    with wave.open(str(capture["audio_path"]), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        if source.getcomptype() != "NONE" or not 1 <= channels <= 8:
            raise ValueError("recorded noise must be uncompressed PCM audio")
        source.setpos(int(candidate["sample_start"]))
        raw = source.readframes(int(candidate["sample_count"]))
    if width == 2:
        pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).astype(np.float64) / 32768.0
    elif width == 1:
        pcm = (np.frombuffer(raw, dtype=np.uint8).reshape(-1, channels).astype(np.float64) - 128.0) / 128.0
    elif width == 4:
        pcm = np.frombuffer(raw, dtype="<i4").reshape(-1, channels).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported recorded noise sample width: {width}")
    mono = pcm.mean(axis=1)
    rate = int(capture["sample_rate"])
    if rate != target_rate:
        from math import gcd
        from scipy.signal import resample_poly
        divisor = gcd(rate, target_rate)
        mono = resample_poly(mono, target_rate // divisor, rate // divisor)
    expected = target_rate * RECORDED_NOISE_SEGMENT_SECONDS
    if mono.size != expected:
        mono = np.interp(np.linspace(0, mono.size - 1, expected), np.arange(mono.size), mono)
    # Short fades suppress clicks at joins without changing the recorded texture.
    fade = min(target_rate // 100, mono.size // 4)
    ramp = np.linspace(0.0, 1.0, fade, endpoint=False)
    mono[:fade] *= ramp
    mono[-fade:] *= ramp[::-1]
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 0.98:
        mono *= 0.98 / peak
    return mono


def _make_recorded_noise_capture(
    store: CaptureStore, *, selected: list[dict[str, Any]], output: Path,
    sample_rate: int, sample_count: int,
) -> dict[str, Any]:
    if sample_count != sample_rate * 10 or sample_count % (sample_rate * RECORDED_NOISE_SEGMENT_SECONDS):
        raise ValueError("recorded-noise backgrounds must be an exact ten seconds")
    pieces = [_recorded_noise_audio(item, sample_rate) for item in selected]
    audio = np.concatenate(pieces)
    if audio.size != sample_count:
        raise ValueError("recorded-noise segments do not fill the requested background window")
    pcm = np.clip(np.round(audio * 32767.0), -32768, 32767).astype("<i2")
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(pcm.tobytes())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    existing = store.capture_by_sha256(digest)
    if existing is not None:
        if existing["source_metadata"].get("generator") != "sig-id-recorded-noise":
            raise ValueError("recorded-noise waveform hash collides with an incompatible capture")
        return existing
    sources = [
        {"capture_id": item["capture"]["id"], "sha256": item["capture"]["sha256"],
         "sample_start": item["sample_start"], "sample_count": item["sample_count"]}
        for item in selected
    ]
    source_ids = sorted({item["capture"]["id"] for item in selected})
    manifest_path = output.with_suffix(output.suffix + ".json")
    manifest = {
        "schema_version": 1, "source_kind": "synthetic",
        "generator": "sig-id-recorded-noise", "generator_version": "0.1.0",
        "deterministic": True, "waveform": output.name, "waveform_sha256": digest,
        "sample_rate": sample_rate, "sample_count": sample_count,
        "sample_width_bytes": 2, "channels": 1,
        "selection_policy": "five disjoint two-second segments from saved 10-second samples with fresh complete human review and no active annotations",
        "source_segments": sources,
    }
    _atomic_json(manifest_path, manifest)
    manifest_digest = _file_sha256(manifest_path)
    metadata = {
        "generator": "sig-id-recorded-noise", "generator_version": "0.1.0",
        "source_kind": "recorded-noise-background", "sample_rate": sample_rate,
        "sample_count": sample_count, "sources": sources,
        "component_capture_ids": source_ids,
        "selection_policy": "saved 10-second samples with fresh complete human review and no active annotations",
        "manifest_file": manifest_path.name, "manifest_sha256": manifest_digest,
    }
    capture_id = store.add_wav_capture(output, source_kind="synthetic", source_metadata=metadata)
    store.add_region(
        capture_id, sample_start=0, sample_count=sample_count,
        frequency_low_hz=None, frequency_high_hz=None,
        label=SignalLabel(
            family=None, method=None, protocol=None, variant=None, source="synthetic",
            review_state="accepted",
            evidence={"generator": "sig-id-recorded-noise", "source_segments": sources, "noise_only": True},
            notes="Recorded two-second slices with no active annotated box; no modulation or protocol is asserted.",
        ),
    )
    return store.capture(capture_id)


def build_cv_synthetic_scenes(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 1,
    overlap_rate: float = 0.25,
    noise_only_examples: int = 0,
    noise_data_dirs: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Generate reviewed, source-held-out-ready signal bursts over known noise.

    Inputs must be accepted, manifest-backed synthetic modulation captures.
    Each source is used in at most one scene, and each scene gets its own
    independently seeded 10-second noise capture so lineage groups stay distinct.
    Optional noise-only captures add independent reviewed negative windows.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if not isinstance(overlap_rate, (int, float)) or isinstance(overlap_rate, bool) or not 0 <= overlap_rate <= 1:
        raise ValueError("overlap_rate must be between 0 and 1")
    if (
        not isinstance(noise_only_examples, int) or isinstance(noise_only_examples, bool)
        or not 0 <= noise_only_examples <= MAX_NOISE_ONLY_EXAMPLES
    ):
        raise ValueError(f"noise_only_examples must be between 0 and {MAX_NOISE_ONLY_EXAMPLES}")
    root = Path(data_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    started_at = time.monotonic()
    store = CaptureStore(root / "dataset.sqlite3")
    noise_roots = [Path(item).expanduser().resolve() for item in (noise_data_dirs or [])]
    noise_stores = [CaptureStore(item / "dataset.sqlite3") for item in dict.fromkeys(noise_roots) if item != root]
    index_path = output / "cv-scenes.json"
    existing_index = None
    if index_path.exists():
        existing_index = json.loads(index_path.read_text(encoding="utf-8"))
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")

    _progress("scanning saved captures for eligible synthetic signals and box-free noise intervals")
    captures = store.list_captures()
    noise_candidates = _recorded_noise_candidates(store, captures)
    for noise_store in noise_stores:
        noise_candidates.extend(_recorded_noise_candidates(noise_store, noise_store.list_captures()))
    noise_pool_sha256 = hashlib.sha256("|".join(sorted(
        f"{item['capture']['sha256']}:{item['sample_start']}:{item['sample_count']}"
        for item in noise_candidates
    )).encode()).hexdigest()
    signals = []
    seen_source_hashes: set[str] = set()
    for capture in captures:
        metadata = capture["source_metadata"]
        if (
            capture["source_kind"] != "synthetic"
            or metadata.get("generator") not in SIGNAL_GENERATORS
            or capture["parent_capture_id"] is not None
            or capture["sample_rate"] / 2 < MAX_FREQUENCY_HZ
            or capture["sha256"] in seen_source_hashes
        ):
            continue
        regions = store.regions(capture["id"])
        if len(regions) != 1:
            continue
        region = regions[0]
        if (
            region["sample_start"] != 0 or region["sample_count"] != capture["sample_count"]
            or region["review_state"] != "accepted" or region["source"] != "synthetic"
            or region["frequency_low_hz"] is None or region["frequency_high_hz"] is None
        ):
            continue
        signals.append((capture, region))
        seen_source_hashes.add(capture["sha256"])
    if len(signals) < 3:
        raise ValueError("CV scene generation requires at least three accepted synthetic signal sources with frequency bounds")
    _progress(
        f"found {len(signals)} accepted signal sources and {len(noise_candidates)} eligible "
        f"two-second noise intervals; noise-only windows requested: {noise_only_examples}"
    )
    source_pool_sha256 = hashlib.sha256("|".join(sorted(seen_source_hashes)).encode()).hexdigest()
    config = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "overlap_rate": float(overlap_rate),
        "scene_recipe_version": SCENE_RECIPE_VERSION,
        "two_signal_layout": "temporally_disjoint-v1",
        "window_seconds": WINDOW_SECONDS,
        "signal_generators": sorted(SIGNAL_GENERATORS),
        "source_pool_sha256": source_pool_sha256,
        "recorded_noise_pool_sha256": noise_pool_sha256,
        "recorded_noise_candidate_count": len(noise_candidates),
        "recorded_noise_data_dirs": [str(item) for item in noise_roots],
        "noise_background_policy": "recorded_two_second_signal_free_slices_when_available_else_seeded_uniform",
        "target_signal_snr_db_range": [MIN_SIGNAL_SNR_DB, MAX_SIGNAL_SNR_DB],
        "snr_convention": "per-signal active-interval in-band mean-square power relative to aligned noise power",
    }
    if noise_only_examples:
        config["noise_only_examples"] = noise_only_examples
    if existing_index is not None:
        if existing_index.get("config") != config:
            raise ValueError("existing scene index uses a different recipe or source pool; choose another output directory")
        index = existing_index
    else:
        output.mkdir(parents=True, exist_ok=True)
        index = {"config": config, "records": []}
        _atomic_json(index_path, index)

    for capture, _ in signals:
        _ensure_synthetic_review(
            store, capture,
            "Controlled synthetic modulation capture; validated generator manifest provides complete signal truth.",
        )

    for capture in captures:
        if (
            capture["source_kind"] == "synthetic"
            and capture["sample_rate"] / 2 >= MAX_FREQUENCY_HZ
            and capture["source_metadata"].get("generator") in {"sig-id-noise", "sig-id-recorded-noise"}
        ):
            generator = capture["source_metadata"].get("generator")
            _ensure_synthetic_review(
                store, capture,
                ("Record-derived noise background assembled from two-second intervals with no active annotated box."
                 if generator == "sig-id-recorded-noise" else
                 "Controlled noise-only capture; validated generator manifest provides complete no-signal truth."),
            )

    noise_only_records = list(index.get("noise_only_records", []))
    if noise_only_examples:
        records_by_index = {int(record["index"]): record for record in noise_only_records}
        for item_index in range(noise_only_examples):
            _progress(f"noise-only window {item_index + 1}/{noise_only_examples}: generating 10-second negative")
            noise_seed = _derived_seed(seed, f"noise-only:{item_index}")
            waveform = output / "noise-only" / f"{item_index:05d}.wav"
            manifest_path = waveform.with_suffix(waveform.suffix + ".json")
            if not waveform.is_file() or not manifest_path.is_file():
                generate_noise_wav(
                    waveform, seed=noise_seed, sample_rate=12_000,
                    sample_count=12_000 * WINDOW_SECONDS, amplitude=0.35,
                )
            digest = hashlib.sha256(waveform.read_bytes()).hexdigest()
            manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            previous_record = records_by_index.get(item_index)
            if previous_record and (
                previous_record.get("sha256") != digest
                or previous_record.get("manifest_sha256") != manifest_digest
            ):
                raise ValueError(f"noise-only source {item_index} no longer matches its generated index record")
            noise_capture = store.capture_by_sha256(digest)
            if noise_capture is None:
                noise_capture = import_synthetic_wav(
                    waveform, manifest_path, store=store, capture_dir=root / "captures",
                )["capture"]
            if (
                noise_capture["source_kind"] != "synthetic"
                or noise_capture["source_metadata"].get("generator") != "sig-id-noise"
                or noise_capture["sha256"] != digest
                or int(noise_capture["sample_rate"]) != 12_000
                or int(noise_capture["sample_count"]) != 12_000 * WINDOW_SECONDS
            ):
                raise ValueError("noise-only corpus source hash is registered with incompatible provenance")
            _ensure_synthetic_review(
                store, noise_capture,
                "Independent controlled noise-only window; validated generator manifest provides complete no-signal truth.",
            )
            records_by_index[item_index] = {
                "index": item_index,
                "capture_id": noise_capture["id"],
                "sha256": digest,
                "manifest_sha256": manifest_digest,
                "seed": noise_seed,
                "sample_rate": 12_000,
                "duration_seconds": WINDOW_SECONDS,
            }
            index["noise_only_records"] = [records_by_index[key] for key in sorted(records_by_index)]
            _atomic_json(index_path, index)
            _progress(f"noise-only window {item_index + 1}/{noise_only_examples}: saved")
        noise_only_records = index["noise_only_records"]

    output_records = list(index.get("records", []))
    processed = {
        digest
        for record in output_records
        for digest in record.get("source_sha256", [])
    }
    scenes_created = 0
    all_scene_groups = _scene_groups([capture for capture, _ in signals], seed, float(overlap_rate))
    pending_scene_groups = [
        sources for sources in all_scene_groups
        if not all(item["sha256"] in processed for item in sources)
    ]
    _progress(
        f"building {len(pending_scene_groups)} new 10-second scenes "
        f"({len(all_scene_groups) - len(pending_scene_groups)} already complete)"
    )
    for scene_index, sources in enumerate(pending_scene_groups, start=1):
        _progress(f"scene {scene_index}/{len(pending_scene_groups)}: composing {len(sources)} signal(s) and noise")
        source_regions = {capture["id"]: region for capture, region in signals if capture["id"] in {item["id"] for item in sources}}
        source_hashes = sorted(item["sha256"] for item in sources)
        sample_rate = int(sources[0]["sample_rate"])
        scene_sample_count = sample_rate * WINDOW_SECONDS
        recipe_key = hashlib.sha256(
            f"{SCENE_RECIPE_VERSION}:{seed}:{'|'.join(source_hashes)}".encode()
        ).hexdigest()
        rng = np.random.default_rng(_derived_seed(seed, recipe_key))
        signal_recipes: list[dict[str, Any]] = []
        first = _signal_recipe(
            sources[0], source_regions[sources[0]["id"]], rng=rng,
            scene_sample_count=scene_sample_count, sample_rate=sample_rate,
        )
        signal_recipes.append(first)
        if len(sources) > 1:
            second_region = source_regions[sources[1]["id"]]
            second_duration = min(4.0, sources[1]["sample_count"] / sample_rate)
            second_min_duration = min(0.75, second_duration)
            second_count = max(4096, min(
                sources[1]["sample_count"],
                int(round(float(rng.uniform(second_min_duration, second_duration)) * sample_rate)),
            ))
            first_start, second_start = _nonoverlapping_pair_starts(
                first["sample_count"], second_count,
                scene_sample_count=scene_sample_count, sample_rate=sample_rate, rng=rng,
            )
            first["scene_start_sample"] = first_start
            # Keep the two boxes separate in time while drawing independent
            # source offsets and impairment recipes.
            second = _signal_recipe(
                sources[1], second_region, rng=rng, scene_sample_count=scene_sample_count,
                sample_rate=sample_rate, scene_start_sample=second_start,
                requested_sample_count=second_count,
            )
            signal_recipes.append(second)

        noise_seed = _derived_seed(seed, f"noise:{recipe_key}")
        noise_waveform = output / "noise" / f"{recipe_key}.wav"
        noise_waveform.parent.mkdir(parents=True, exist_ok=True)
        noise_manifest = noise_waveform.with_suffix(noise_waveform.suffix + ".json")
        if noise_candidates:
            noise_rng = np.random.default_rng(_derived_seed(seed, f"recorded-noise:{recipe_key}"))
            eligible = sorted(
                noise_candidates,
                key=lambda item: (item["capture"]["id"], int(item["sample_start"])),
            )
            nonoverlapping: list[dict[str, Any]] = []
            last_end_by_capture: dict[str, int] = {}
            for item in eligible:
                capture_id = item["capture"]["id"]
                start = int(item["sample_start"])
                if start >= last_end_by_capture.get(capture_id, -1):
                    nonoverlapping.append(item)
                    last_end_by_capture[capture_id] = start + int(item["sample_count"])
            chosen_noise: list[dict[str, Any]] = []
            if len(nonoverlapping) >= 5:
                indices = noise_rng.choice(len(nonoverlapping), size=5, replace=False)
                chosen_noise = [nonoverlapping[int(index)] for index in indices]
            if len(chosen_noise) == 5:
                noise_capture = _make_recorded_noise_capture(
                    store, selected=chosen_noise, output=noise_waveform,
                    sample_rate=sample_rate, sample_count=scene_sample_count,
                )
                noise_source = "recorded_signal_free_slices"
            else:
                chosen_noise = []
        else:
            chosen_noise = []
        if not chosen_noise:
            if not noise_waveform.is_file() or not noise_manifest.is_file():
                generate_noise_wav(
                    noise_waveform, seed=noise_seed, sample_rate=sample_rate,
                    sample_count=scene_sample_count, amplitude=0.35,
                )
            noise_capture = store.capture_by_sha256(hashlib.sha256(noise_waveform.read_bytes()).hexdigest())
            if noise_capture is None:
                noise_capture = import_synthetic_wav(
                    noise_waveform, noise_manifest, store=store, capture_dir=root / "captures",
                )["capture"]
            noise_source = "seeded_uniform_fallback"
        _ensure_synthetic_review(
            store, noise_capture,
            "Generated noise background; validated generator manifest provides complete no-signal truth.",
        )
        noise_gain = float(rng.uniform(0.25, 0.55))
        for recipe in signal_recipes:
            recipe["target_snr_db"] = float(rng.uniform(MIN_SIGNAL_SNR_DB, MAX_SIGNAL_SNR_DB))
        noise_gain, signal_snr = _set_signal_gains_for_snr(
            signal_recipes, noise_capture=noise_capture, noise_gain=noise_gain, store=store,
        )
        scene_recipe = [*signal_recipes, {
            "capture_id": noise_capture["id"],
            "sample_count": scene_sample_count,
            "scene_start_sample": 0,
            "gain": noise_gain,
        }]
        scene_waveform = output / "scenes" / f"{recipe_key}.wav"
        scene_waveform.parent.mkdir(parents=True, exist_ok=True)
        scene_manifest = create_mixed_scene(
            scene_waveform, scene_recipe, store=store, sample_count=scene_sample_count,
        )
        scene_capture = store.capture_by_sha256(scene_manifest["waveform_sha256"])
        if scene_capture is None:
            scene_capture = import_mixed_scene(
                scene_waveform, scene_waveform.with_suffix(scene_waveform.suffix + ".json"),
                store=store, capture_dir=root / "captures",
            )["capture"]
        _ensure_synthetic_review(
            store, scene_capture,
            "Deterministic synthetic scene; validated component manifests provide complete signal and noise truth.",
        )
        record = {
            "source_capture_ids": [item["id"] for item in sources],
            "source_sha256": source_hashes,
            "noise_capture_id": noise_capture["id"],
            "noise_seed": noise_seed,
            "noise_source": noise_source,
            "target_signal_snr_db_range": [MIN_SIGNAL_SNR_DB, MAX_SIGNAL_SNR_DB],
            "signal_snr_measurements": signal_snr,
            "scene_capture_id": scene_capture["id"],
            "scene_sha256": scene_capture["sha256"],
            "component_count": len(signal_recipes) + 1,
            "scene_seconds": WINDOW_SECONDS,
        }
        output_records.append(record)
        processed.update(source_hashes)
        scenes_created += 1
        index["records"] = output_records
        _atomic_json(index_path, index)
        _progress(
            f"scene {scene_index}/{len(pending_scene_groups)}: saved "
            f"({noise_source.replace('_', ' ')}, capture {scene_capture['id']})"
        )

    _progress(
        f"complete: {scenes_created} scenes created, {len(output_records)} total, "
        f"{time.monotonic() - started_at:.1f}s elapsed"
    )

    return {
        "index": str(index_path),
        "source_count": len(signals),
        "scene_count": len(output_records),
        "scenes_created_this_run": scenes_created,
        "noise_only_windows": len(output_records),
        "additional_noise_only_windows": len(noise_only_records),
        "recorded_noise_candidate_count": len(noise_candidates),
        "recorded_noise_backgrounds_used": sum(record.get("noise_source") == "recorded_signal_free_slices" for record in output_records),
        "lineage_split_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("~/.local/share/sig-id").expanduser())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--overlap-rate", type=float, default=0.25,
        help="fraction of scenes containing two signals; paired signals are placed in separate time intervals",
    )
    parser.add_argument(
        "--noise-only-examples", type=int, default=0,
        help=f"add this many independent reviewed 10-second uniform-noise captures (max {MAX_NOISE_ONLY_EXAMPLES})",
    )
    parser.add_argument(
        "--noise-data-dir", type=Path, action="append", default=[],
        help="additional dataset containing human-reviewed saved 10-second noise samples; may be repeated",
    )
    args = parser.parse_args()
    try:
        report = build_cv_synthetic_scenes(
            args.data_dir, args.output_dir, seed=args.seed, overlap_rate=args.overlap_rate,
            noise_only_examples=args.noise_only_examples, noise_data_dirs=args.noise_data_dir,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
