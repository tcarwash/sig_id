from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigid.dataset import CaptureStore
from sigid.detection import detect_tracks
from sigid.detection_evaluation import evaluate_synthetic_detector
from sigid.scenes import create_mixed_scene, import_mixed_scene
from sigid.spectrum import waterfall
from sigid.synthetic import generate_2fsk_wav, generate_bpsk_wav, generate_noise_wav
from sigid.synthetic_dataset import import_synthetic_wav


def _import_source(tmp_path: Path, store: CaptureStore, filename: str, generator, **kwargs):
    waveform = tmp_path / filename
    generator(waveform, **kwargs)
    return import_synthetic_wav(
        waveform,
        waveform.with_suffix(waveform.suffix + ".json"),
        store=store,
        capture_dir=tmp_path / "captures",
    )


def test_mixed_scene_is_deterministic_and_imports_overlapping_constituent_labels(tmp_path):
    store = CaptureStore(tmp_path / "dataset.sqlite3")
    fsk = _import_source(
        tmp_path, store, "fsk.wav", generate_2fsk_wav,
        bits="01" * 40, sample_rate=8000, symbol_rate=100,
        tone_low_hz=800, tone_high_hz=1100,
    )
    bpsk = _import_source(
        tmp_path, store, "bpsk.wav", generate_bpsk_wav,
        bits="00110110" * 10, sample_rate=8000, symbol_rate=100,
        carrier_hz=3000,
    )
    components = [
        {"capture_id": fsk["capture"]["id"], "sample_count": 6400, "scene_start_sample": 0, "gain": 0.7},
        {"capture_id": bpsk["capture"]["id"], "sample_count": 6400, "scene_start_sample": 3200, "gain": 0.5},
    ]

    first = tmp_path / "scene.wav"
    manifest = create_mixed_scene(first, components, store=store)
    first_bytes = first.read_bytes()
    second = tmp_path / "scene-again.wav"
    create_mixed_scene(second, components, store=store)
    assert second.read_bytes() == first_bytes
    assert manifest["sample_count"] == 9600
    assert manifest["labels"]["modulation_method"] is None
    candidates = detect_tracks(waterfall(first, fft_size=512, hop_size=128, max_rows=256))
    centers = [candidate["center_hz"] for candidate in candidates]
    assert len(candidates) >= 2
    assert any(800 <= center <= 1100 for center in centers)
    assert any(2800 <= center <= 3200 for center in centers)

    imported = import_mixed_scene(
        first, first.with_suffix(".wav.json"), store=store, capture_dir=tmp_path / "captures"
    )
    assert imported["capture"]["source_kind"] == "synthetic"
    assert [region["sample_start"] for region in imported["regions"]] == [0, 3200]
    assert [region["sample_count"] for region in imported["regions"]] == [6400, 6400]
    assert {region["method"] for region in imported["regions"]} == {
        "binary frequency shift keying", "binary phase-shift keying"
    }
    assert all(region["review_state"] == "accepted" for region in imported["regions"])
    assert imported["regions"][0]["sample_start"] + imported["regions"][0]["sample_count"] > imported["regions"][1]["sample_start"]

    noise_path = tmp_path / "noise.wav"
    generate_noise_wav(noise_path, seed=882, sample_rate=8000, sample_count=16000)
    noise = import_synthetic_wav(
        noise_path,
        noise_path.with_suffix(".wav.json"),
        store=store,
        capture_dir=tmp_path / "captures",
    )
    evaluation = evaluate_synthetic_detector(
        store,
        capture_ids=[imported["capture"]["id"], noise["capture"]["id"]],
        fft_size=512,
        hop_size=128,
    )
    assert evaluation["ground_truth_components"] == 2
    assert evaluation["detected_components"] == 2
    assert evaluation["component_recall"] == 1.0
    assert evaluation["candidate_precision"] == 1.0
    assert evaluation["noise_only_false_alarms_per_hour"] == 0.0
    assert evaluation["false_candidates_per_hour"] == 0.0


def test_mixed_scene_import_rejects_recipe_tampering_without_storing_capture(tmp_path):
    store = CaptureStore(tmp_path / "dataset.sqlite3")
    first = _import_source(
        tmp_path, store, "first.wav", generate_2fsk_wav, bits="01" * 25,
        sample_rate=8000, symbol_rate=100,
    )
    second = _import_source(
        tmp_path, store, "second.wav", generate_bpsk_wav, bits="01" * 25,
        sample_rate=8000, symbol_rate=100,
    )
    output = tmp_path / "tamper-scene.wav"
    recipe = [
        {"capture_id": first["capture"]["id"], "sample_count": 4000, "scene_start_sample": 0},
        {"capture_id": second["capture"]["id"], "sample_count": 4000, "scene_start_sample": 0},
    ]
    create_mixed_scene(output, recipe, store=store)
    sidecar = output.with_suffix(".wav.json")
    manifest = json.loads(sidecar.read_text())
    manifest["components"][0]["gain"] = 0.1
    sidecar.write_text(json.dumps(manifest))

    before = len(store.list_captures())
    with pytest.raises(ValueError, match="recipe"):
        import_mixed_scene(output, sidecar, store=store, capture_dir=tmp_path / "captures")
    assert len(store.list_captures()) == before
