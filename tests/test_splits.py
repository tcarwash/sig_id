from __future__ import annotations

import json

import pytest

from sigid.corpus import build_corpus
from sigid.dataset import CaptureStore
from sigid.scenes import create_mixed_scene, import_mixed_scene
from sigid.splits import create_split_manifest
from sigid.synthetic import generate_bpsk_wav
from sigid.synthetic_dataset import import_synthetic_wav


def test_split_manifest_keeps_clean_sources_and_awgn_siblings_together(tmp_path):
    output = tmp_path / "corpus"
    data = tmp_path / "data"
    build_corpus(output, data, examples_per_mode=3, snr_levels_db=[10], seed=202)
    store = CaptureStore(data / "dataset.sqlite3")
    split = create_split_manifest(output / "corpus.json", store, seed=7, assignment_version=2)
    split_again = create_split_manifest(output / "corpus.json", store, seed=7, assignment_version=2)
    split_other_seed = create_split_manifest(output / "corpus.json", store, seed=8, assignment_version=2)
    assert split == split_again
    assignments = {record["sha256"]: record["split"] for record in split["records"]}
    other_assignments = {record["sha256"]: record["split"] for record in split_other_seed["records"]}
    assert assignments != other_assignments
    assert split["counts"]["train"]["source_groups"] > 0
    assert split["counts"]["validation"]["source_groups"] > 0
    assert split["counts"]["test"]["source_groups"] > 0
    assert all(
        split["counts"][name]["by_mode"][mode] == 1
        for name in ("train", "validation", "test")
        for mode in ("2fsk", "4fsk", "8fsk", "16fsk", "bpsk", "qpsk")
    )
    assert all(split["counts"][name]["by_mode"]["noise"] == 1 for name in ("train", "validation", "test"))

    assignments = {record["lineage_group"]: record["split"] for record in split["records"]}
    assert len(assignments) == 21
    assert all(assignments[record["lineage_group"]] == record["split"] for record in split["records"])
    for record in split["records"]:
        if record["kind"] == "awgn":
            root = next(item for item in split["records"] if item["capture_id"] == record["lineage_group"])
            assert root["split"] == record["split"]


def test_split_manifest_rejects_too_few_groups_and_tampered_lineage(tmp_path):
    output = tmp_path / "small"
    data = tmp_path / "data"
    build_corpus(output, data, examples_per_mode=1, snr_levels_db=[10], seed=3)
    store = CaptureStore(data / "dataset.sqlite3")
    with pytest.raises(ValueError, match="at least three independent"):
        create_split_manifest(output / "corpus.json", store)

    corpus_path = output / "corpus.json"
    corpus = json.loads(corpus_path.read_text())
    child = next(record for record in corpus["records"] if record["kind"] == "awgn")
    child["lineage_group"] = "invented-parent"
    corpus_path.write_text(json.dumps(corpus))
    with pytest.raises(ValueError, match="not a direct child"):
        create_split_manifest(corpus_path, store)


def test_split_assignment_is_stable_across_fresh_databases(tmp_path):
    assignments = []
    for name in ("first", "second"):
        output = tmp_path / name / "corpus"
        data = tmp_path / name / "data"
        build_corpus(output, data, examples_per_mode=5, snr_levels_db=[], seed=91)
        split = create_split_manifest(output / "corpus.json", CaptureStore(data / "dataset.sqlite3"), seed=37)
        assignments.append({(record["mode"], record["sha256"]): record["split"] for record in split["records"]})
    assert assignments[0] == assignments[1]


def test_mixed_scene_unifies_all_component_lineages_in_one_split(tmp_path):
    output = tmp_path / "corpus"
    data = tmp_path / "data"
    build_corpus(output, data, examples_per_mode=4, snr_levels_db=[], seed=811)
    store = CaptureStore(data / "dataset.sqlite3")
    corpus = json.loads((output / "corpus.json").read_text())
    fsk = next(record for record in corpus["records"] if record["mode"] == "2fsk" and record["kind"] == "clean")
    bpsk = next(record for record in corpus["records"] if record["mode"] == "bpsk" and record["kind"] == "clean")
    scene_path = tmp_path / "scenes" / "mixed.wav"
    create_mixed_scene(
        scene_path,
        [
            {"capture_id": fsk["capture_id"]},
            {"capture_id": bpsk["capture_id"], "scene_start_sample": 1000, "gain": 0.5},
        ],
        store=store,
    )
    imported_scene = import_mixed_scene(
        scene_path,
        scene_path.with_suffix(".wav.json"),
        store=store,
        capture_dir=data / "captures",
    )
    external_waveform = tmp_path / "external-bpsk.wav"
    generate_bpsk_wav(external_waveform, "01" * 32)
    external = import_synthetic_wav(
        external_waveform,
        external_waveform.with_suffix(".wav.json"),
        store=store,
        capture_dir=data / "captures",
    )
    external_scene_path = tmp_path / "scenes" / "not-in-corpus.wav"
    create_mixed_scene(
        external_scene_path,
        [{"capture_id": fsk["capture_id"]}, {"capture_id": external["capture"]["id"]}],
        store=store,
    )
    imported_external_scene = import_mixed_scene(
        external_scene_path,
        external_scene_path.with_suffix(".wav.json"),
        store=store,
        capture_dir=data / "captures",
    )

    split = create_split_manifest(output / "corpus.json", store, seed=19)
    dependency = next(item for item in split["mixed_scene_dependencies"] if item["capture_id"] == imported_scene["capture"]["id"])
    assignments = {record["lineage_group"]: record["split"] for record in split["records"] if record["capture_id"] == record["lineage_group"]}
    linked_splits = {assignments[group] for group in dependency["lineage_groups"]}
    assert len(linked_splits) == 1
    assert dependency["split"] == next(iter(linked_splits))
    assert dependency["split"] == "test"
    assert len(dependency["lineage_groups"]) == 2
    exclusion = next(item for item in split["excluded_mixed_scenes"] if item["capture_id"] == imported_external_scene["capture"]["id"])
    assert exclusion["missing_component_capture_ids"] == [external["capture"]["id"]]


def test_scene_lineage_join_rejects_insufficient_independent_test_groups(tmp_path):
    output = tmp_path / "corpus"
    data = tmp_path / "data"
    build_corpus(output, data, examples_per_mode=3, snr_levels_db=[], seed=812)
    store = CaptureStore(data / "dataset.sqlite3")
    corpus = json.loads((output / "corpus.json").read_text())
    sources = [
        record for record in corpus["records"]
        if record["mode"] == "2fsk" and record["kind"] == "clean"
    ][:2]
    scene_path = tmp_path / "scene.wav"
    create_mixed_scene(
        scene_path,
        [{"capture_id": source["capture_id"]} for source in sources],
        store=store,
    )
    import_mixed_scene(
        scene_path,
        scene_path.with_suffix(".wav.json"),
        store=store,
        capture_dir=data / "captures",
    )
    with pytest.raises(ValueError, match="after mixed-scene lineage joins"):
        create_split_manifest(output / "corpus.json", store)
