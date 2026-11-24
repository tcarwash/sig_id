"""Regression and integration tests for the isolated synthetic-corpus v4 recipe."""
from __future__ import annotations

import json
from pathlib import Path
import wave

import numpy as np
import pytest

from sigid.cv_scene_corpus import (
    FREQUENCY_LAYOUTS, SNR_CURRICULUM, _apply_receiver_variation,
    _choose_pair_center, _source_snr_target, build_cv_synthetic_scenes,
)
from sigid.dataset import CaptureStore
from sigid.synthetic import generate_2fsk_wav, generate_bpsk_wav, generate_ook_wav
from sigid.synthetic_dataset import _validate_manifest, import_synthetic_wav
from sigid.synthetic_quality import measure_occupied_band, audit_scene_corpus


def test_balanced_snr_covers_clean_and_weak_bands_deterministically():
    examples = [_source_snr_target(77, f"source-{i:04d}", "balanced") for i in range(300)]
    assert examples == [_source_snr_target(77, f"source-{i:04d}", "balanced") for i in range(300)]
    counts = {key: sum(label == key for label, _ in examples) for key, *_ in SNR_CURRICULUM}
    assert all(value > 20 for value in counts.values()), counts
    for label, snr in examples:
        _, lower, upper, _ = next(item for item in SNR_CURRICULUM if item[0] == label)
        assert lower <= snr <= upper
    assert all(-40 <= _source_snr_target(77, f"source-{i}", "legacy_weak")[1] <= -15 for i in range(40))


@pytest.mark.parametrize("layout", ["separated", "adjacent", "cochannel"])
def test_pair_layout_generates_expected_frequency_relationship(layout):
    first = {"frequency_offset_hz": 100.0}
    first_region = {"frequency_low_hz": 900.0, "frequency_high_hz": 1100.0}
    second = {"frequency_low_hz": 1500.0, "frequency_high_hz": 1560.0}
    center, decision = _choose_pair_center(first, first_region, second,
                                           sample_rate=12000, layout=layout,
                                           rng=np.random.default_rng(88))
    assert center is not None
    assert decision == layout
    first_low, first_high = 1000.0, 1200.0
    second_low, second_high = center - 30.0, center + 30.0
    if layout == "separated":
        assert second_high < first_low - 90.0 or second_low > first_high + 90.0
    elif layout == "adjacent":
        assert 15 <= max(first_low - second_high, second_low - first_high) <= 65
    else:
        assert first_low < center < first_high


def test_clock_variation_stays_inside_scene_and_never_filters_annotated_passband():
    region = {"frequency_low_hz": 1400.0, "frequency_high_hz": 1600.0}
    for seed in range(100):
        recipe = {
            "scene_start_sample": 90000, "sample_count": 30000,
            "frequency_offset_hz": 0.0,
        }
        _apply_receiver_variation(recipe, region, sample_rate=12000,
                                  scene_sample_count=120000, rng=np.random.default_rng(seed))
        ppm = recipe["clock_error_ppm"]
        rendered = round(recipe["sample_count"] / (1 + ppm * 1e-6))
        assert recipe["scene_start_sample"] >= 0
        assert recipe["scene_start_sample"] + rendered <= 120000
        if recipe.get("lowpass_hz") is not None:
            assert recipe["lowpass_hz"] > region["frequency_high_hz"]


def test_measured_waveform_band_is_tight_and_reproducible(tmp_path: Path):
    path = tmp_path / "bpsk.wav"
    generate_bpsk_wav(path, "01101001" * 50, symbol_rate=100, carrier_hz=1500.0)
    measured = measure_occupied_band(path)
    assert measured == measure_occupied_band(path)
    assert measured["frequency_high_hz"] - measured["frequency_low_hz"] < 350
    assert measured["frequency_low_hz"] < 1500 < measured["frequency_high_hz"]
    assert 0.95 < measured["active_fraction"] <= 1.0

    silent = tmp_path / "silent.wav"
    with wave.open(str(silent), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(12000)
        wav.writeframes(b"\x00\x00" * 12000)
    with pytest.raises(ValueError, match="too little signal energy"):
        measure_occupied_band(silent)


def test_fldigi_v2_manifest_band_measurement_is_verified(tmp_path: Path):
    from sigid.modes import MODE_REGISTRY
    # We don't launch Fldigi or key a radio: test the manifest validator
    # against a generated BPSK waveform and a known registry mode.
    candidates = [
        (name, entry) for name, entry in MODE_REGISTRY.items()
        if entry.get("backend_selectable")
        and str(entry.get("catalog_source", "")).startswith("Fldigi ")
        and entry.get("backend_name")
    ]
    assert candidates
    name, mode = next(((name, mode) for name, mode in candidates
                       if "bpsk31" in name.lower()), candidates[0])
    path = tmp_path / "synth.wav"
    generate_bpsk_wav(path, "01011001" * 50, symbol_rate=100, carrier_hz=1500)
    m = measure_occupied_band(path)
    import hashlib
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    labels = {key: mode[source] for key, source in [
        ("modulation_family", "family"), ("modulation_method", "method"),
        ("protocol", "protocol"), ("variant", "variant"),
    ]}
    params = {
        "native_mode": mode["backend_name"], "backend": "fldigi",
        "fldigi_version": "dummy-test-version", "transmit_payload": "HELLO",
        "waveform_sha256": sha, "band_measurement": m,
        "frequency_low_hz": m["frequency_low_hz"],
        "frequency_high_hz": m["frequency_high_hz"],
    }
    manifest = {
        "schema_version": 1, "source_kind": "synthetic",
        "generator": "sig-id-fldigi", "generator_version": "0.2.0",
        "deterministic": True, "mode_id": name,
        "labels": labels, "parameters": params, "reference_text": "HELLO",
        "sample_rate": 12000, "sample_count": 48000, "channels": 1,
        "sample_width_bytes": 2,
    }
    _validate_manifest(manifest, path)
    manifest["parameters"]["frequency_high_hz"] += 100
    with pytest.raises(ValueError, match="labeled frequency bounds"):
        _validate_manifest(manifest, path)
    manifest["parameters"]["frequency_high_hz"] -= 100
    manifest["parameters"]["band_measurement"] = {**m, "active_fraction": 0}
    with pytest.raises(ValueError, match="measurement disagrees"):
        _validate_manifest(manifest, path)


def test_v4_scene_generation_provenance_and_frequency_layout(tmp_path: Path):
    root, output = tmp_path / "dataset", tmp_path / "scenes"
    store = CaptureStore(root / "dataset.sqlite3")
    cases = [
        (generate_2fsk_wav, {"tone_low_hz": 850, "tone_high_hz": 1020}),
        (generate_bpsk_wav, {"carrier_hz": 1450}),
        (generate_ook_wav, {"carrier_hz": 1850}),
    ]
    for index, (generator, kwargs) in enumerate(cases):
        path = tmp_path / f"source-{index}.wav"
        generator(path, "01100110" * 50, sample_rate=12000,
                  symbol_rate=100, **kwargs)
        imported = import_synthetic_wav(path, path.with_suffix(".wav.json"),
                                        store=store, capture_dir=root / "captures")
        assert imported["region"]["frequency_low_hz"] is not None
    # An independent frozen-holdout SHA exclusion is part of the reproducible recipe.
    excluded_file = tmp_path / "excluded-holdout.json"
    excluded_file.write_text(json.dumps({
        "excluded_capture_sha256": ["a" * 64],
        "frozen_holdout_capture_sha256": ["a" * 64],
    }))
    report = build_cv_synthetic_scenes(
        root, output, seed=501, overlap_rate=1.0, simultaneous_rate=1.0,
        noise_only_examples=1, snr_profile="balanced", frequency_layout="separated",
        exclude_noise_sha_file=excluded_file,
    )
    assert report["scene_count"] == 2
    assert report["additional_noise_only_windows"] == 1
    index = json.loads((output / "cv-scenes.json").read_text())
    assert index["config"]["scene_recipe_version"] == 4
    assert index["config"]["snr_profile"] == "balanced"
    assert index["config"]["frequency_layout"] == "separated"
    assert len(index["records"]) == 2
    assert len({s for r in index["records"] for s in r["source_sha256"]}) == 3
    paired = next(r for r in index["records"] if r["component_count"] == 3)
    assert paired["frequency_layout_requested"] == "separated"
    regs = [r for r in store.regions(paired["scene_capture_id"]) if r.get("method")]
    assert len(regs) == 2
    assert regs[0]["frequency_high_hz"] < regs[1]["frequency_low_hz"] or (
        regs[1]["frequency_high_hz"] < regs[0]["frequency_low_hz"]
    )
    for rec in index["records"]:
        for item in rec["signal_snr_measurements"]:
            assert -40 <= item["target_snr_db"] <= 25
            assert abs(item["target_snr_db"] - item["measured_snr_db"]) < 1e-7
            assert item["snr_profile_bin"] in {b[0] for b in SNR_CURRICULUM}
    # Validate the generated manifests and waveform ownership again by
    # reusing the exact source-pool output path (idempotent scene build).
    repeated = build_cv_synthetic_scenes(
        root, output, seed=501, overlap_rate=1.0, simultaneous_rate=1.0,
        noise_only_examples=1, snr_profile="balanced", frequency_layout="separated",
        exclude_noise_sha_file=excluded_file,
    )
    assert repeated["scenes_created_this_run"] == 0
    quality = audit_scene_corpus(root, output / "cv-scenes.json")
    assert quality["scene_count"] == 2
    assert quality["signal_components"] == 3
    assert quality["two_signal_scenes"] == 1
    assert quality["temporally_overlapping_pairs"] == 1
    # The v4 supplement MUST NOT create held-out validation/test images.
    from sigid.cv_dataset import export_cv_dataset
    exported = tmp_path / "train-only-export"
    report = export_cv_dataset(root, exported, train_only=True, scene_index=output / "cv-scenes.json")
    assert report["train_only"] is True
    assert report["split_images"]["test"] == report["split_images"]["validation"] == 0
    rows = [json.loads(line) for line in (exported / "windows.jsonl").read_text().splitlines()]
    assert rows and all(row["split"] == "train" for row in rows)
    scene_rows = [row for row in rows if row["capture_id"] in {
        record["scene_capture_id"] for record in index["records"]
    }]
    assert len(scene_rows) == len(index["records"])
    assert all(row["conditions"]["synthetic_components"] for row in scene_rows)
    assert all(row["conditions"]["synthetic_snr_profile"] == "balanced" for row in scene_rows)
    from tools.verify_synthetic_train_release import audit_train_only_release
    handoff = audit_train_only_release(root, output / "cv-scenes.json", exported, excluded_file)
    assert handoff["valid_train_only_release"] is True
    assert handoff["train_images"] == len(rows)
    assert handoff["scene_count"] == len(index["records"])
    assert handoff["validation_images"] == handoff["test_images"] == 0
    first_image = exported / rows[0]["image"]
    original_png = first_image.read_bytes()
    first_image.write_bytes(original_png + b"\x00")
    with pytest.raises(ValueError, match="train image hash changed"):
        audit_train_only_release(root, output / "cv-scenes.json", exported, excluded_file)
    first_image.write_bytes(original_png)
    with pytest.raises(ValueError, match="train-only"):
        export_cv_dataset(root, tmp_path / "invalid-export", scene_index=output / "cv-scenes.json")


def test_migrate_old_fldigi_source_create_only_and_idempotent(tmp_path: Path):
    import hashlib
    from sigid.modes import MODE_REGISTRY
    from tools.relabel_fldigi_sources import migrate_fldigi_sources
    name, mode = next((name, mode) for name, mode in MODE_REGISTRY.items()
                      if "bpsk31" in name.lower() and mode.get("backend_selectable")
                      and str(mode.get("catalog_source", "")).startswith("Fldigi "))
    original_root = tmp_path / "original"
    target_root = tmp_path / "corrected"
    wav = tmp_path / "source.wav"
    generate_bpsk_wav(wav, "01100110" * 40, symbol_rate=100, carrier_hz=1500)
    sha = hashlib.sha256(wav.read_bytes()).hexdigest()
    params = {
        "native_mode": mode["backend_name"], "backend": "fldigi",
        "fldigi_version": "offline-test", "transmit_payload": "CQ TEST",
        "frequency_low_hz": 750.0, "frequency_high_hz": 2250.0,
        "waveform_sha256": sha,
    }
    manifest = {
        "schema_version": 1, "source_kind": "synthetic",
        "generator": "sig-id-fldigi", "generator_version": "0.1.0",
        "deterministic": True, "mode_id": name, "reference_text": "CQ TEST",
        "sample_rate": 12000, "sample_count": 38400,
        "channels": 1, "sample_width_bytes": 2,
        "labels": {
            "modulation_family": mode["family"],
            "modulation_method": mode["method"],
            "protocol": mode["protocol"], "variant": mode["variant"],
        }, "parameters": params,
    }
    sidecar = wav.with_suffix(".wav.json")
    sidecar.write_text(json.dumps(manifest))
    old_store = CaptureStore(original_root / "dataset.sqlite3")
    imported = import_synthetic_wav(wav, sidecar, store=old_store,
                                    capture_dir=original_root / "captures")
    old_sha = imported["capture"]["sha256"]
    first = migrate_fldigi_sources(original_root, target_root)
    assert first["source_count"] == first["imported_count"] == 1
    assert first["skipped_count"] == 0
    assert first["mean_original_bandwidth_hz"] == 1500.0
    assert first["mean_measured_bandwidth_hz"] < 400.0
    old_region = old_store.regions(imported["capture"]["id"])[0]
    assert old_region["frequency_high_hz"] - old_region["frequency_low_hz"] == 1500.0
    new_store = CaptureStore(target_root / "dataset.sqlite3")
    corrected = new_store.capture_by_sha256(old_sha)
    assert corrected is not None
    corrected_region = new_store.regions(corrected["id"])[0]
    assert corrected_region["frequency_high_hz"] - corrected_region["frequency_low_hz"] < 400.0
    assert corrected["source_metadata"]["generator_version"] == "0.2.0"
    again = migrate_fldigi_sources(original_root, target_root)
    assert again["imported_count"] == 0
    assert again["already_imported_count"] == 1


def test_exported_scene_geometry_separates_box_count_from_true_collision():
    from sigid.cv_dataset import _window_conditions
    capture = {"source_kind": "synthetic", "source_metadata": {"generator": "sig-id-mixed-scene"}}

    def signal(start, count, low, high):
        return {
            "lineage_state": "active", "review_state": "accepted",
            "sample_start": start, "sample_count": count,
            "frequency_low_hz": low, "frequency_high_hz": high,
            "method": "PSK", "family": "PSK", "evidence": {},
        }

    first = signal(0, 4000, 1000, 1200)
    later = signal(6000, 4000, 1000, 1200)
    separated_band = signal(2000, 4000, 1900, 2000)
    collision = signal(2000, 4000, 1100, 1300)

    sequential = _window_conditions([first, later], capture, 0, 12000)
    assert sequential["signal_overlap"] == "multiple"
    assert sequential["signal_count"] == 2
    assert sequential["temporal_overlap"] is False
    assert sequential["frequency_overlap"] is True
    assert sequential["time_frequency_collision"] is False

    adjacent = _window_conditions([first, separated_band], capture, 0, 12000)
    assert adjacent["temporal_overlap"] is True
    assert adjacent["frequency_overlap"] is False
    assert adjacent["time_frequency_collision"] is False

    collided = _window_conditions([first, collision], capture, 0, 12000)
    assert collided["temporal_overlap"] is True
    assert collided["frequency_overlap"] is True
    assert collided["time_frequency_collision"] is True


def test_holdout_noise_excludes_every_capture_from_same_session(tmp_path: Path):
    from tools.prepare_synthetic_holdout_exclusions import prepare_exclusions
    root = tmp_path / "recorded"
    store = CaptureStore(root / "dataset.sqlite3")
    saved = []
    for i, session in enumerate(("eval-session", "eval-session", "unrelated")):
        wav = tmp_path / f"real-{i}.wav"
        generate_bpsk_wav(wav, "01100110" * (35 + i), symbol_rate=100)
        id_ = store.add_wav_capture(wav, source_kind="live", source_metadata={
            "auto_capture_session_id": session,
        })
        saved.append(store.capture(id_))
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(json.dumps({"split": "test", "capture_sha256": saved[0]["sha256"]}) + "\n")
    result = prepare_exclusions(root, [heldout])
    assert result["frozen_holdout_count"] == 1
    assert result["frozen_capture_session_count"] == 1
    assert set(result["excluded_capture_sha256"]) == {
        saved[0]["sha256"], saved[1]["sha256"],
    }
    missing = tmp_path / "missing.jsonl"
    missing.write_text(json.dumps({"split": "test", "capture_sha256": "0" * 64}) + "\n")
    with pytest.raises(ValueError, match="unavailable"):
        prepare_exclusions(root, [missing])


def test_diverse_fldigi_payloads_are_deterministic_and_distinct():
    from sigid.fldigi_synthetic import diverse_fldigi_payload
    import re
    for mode in ("fldigi-cw", "rtty", "bpsk31"):
        messages = [diverse_fldigi_payload(seed=24, mode_id=mode, example_index=i) for i in range(16)]
        assert messages == [diverse_fldigi_payload(seed=24, mode_id=mode, example_index=i) for i in range(16)]
        assert len(set(messages)) == 16
        assert all(re.fullmatch(r"[A-Z0-9 ]+", value) for value in messages)
        assert all(16 <= len(value) <= 100 for value in messages)


def test_scene_excerpt_avoids_declared_silent_tx_lead_in():
    from sigid.cv_scene_corpus import _signal_recipe

    rate = 12000
    source = {
        "id": "source-1", "sample_count": 20 * rate,
        "source_metadata": {"parameters": {
            "band_measurement": {"active_start_seconds": 2.0, "active_end_seconds": 18.0},
        }},
    }
    region = {"frequency_low_hz": 1400.0, "frequency_high_hz": 1600.0}
    for seed in range(35):
        result = _signal_recipe(source, region, rng=np.random.default_rng(seed),
                                sample_rate=rate, scene_sample_count=10 * rate)
        assert result["source_start_sample"] >= 2 * rate
        assert result["source_start_sample"] + result["sample_count"] <= 18 * rate
        assert result["scene_start_sample"] + result["sample_count"] <= 10 * rate


def test_train_only_export_rejects_real_source_audio(tmp_path: Path):
    from sigid.cv_dataset import export_cv_dataset

    root = tmp_path / "live-input"
    store = CaptureStore(root / "dataset.sqlite3")
    sample = tmp_path / "example.wav"
    generate_bpsk_wav(sample, "01100110" * 40, sample_rate=12000)
    store.add_wav_capture(sample, source_kind="live", source_metadata={})
    with pytest.raises(ValueError, match="isolated synthetic-only dataset root"):
        export_cv_dataset(root, tmp_path / "should-never-export", train_only=True)
