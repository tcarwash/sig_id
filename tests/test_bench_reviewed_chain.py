"""Frozen real-source review benchmark: lineage guards, integrity, scoring."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import wave

import numpy as np
from PIL import Image
import pytest

from tools.bench_reviewed_chain import (
    benchmark_reviewed_chain, load_frozen_manifest, load_payload_references,
    resolve_sources,
)


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path):
    dataset = tmp_path / "export"
    roots = tmp_path / "roots"
    (dataset / "images" / "test").mkdir(parents=True)
    (dataset / "labels" / "test").mkdir(parents=True)
    roots.mkdir()
    pcm_path = roots / "reviewed.wav"
    with wave.open(str(pcm_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes((np.sin(np.arange(8000) * 2 * np.pi * 1050 / 8000)
                            * 24000).astype("<i2").tobytes())
    img = dataset / "images" / "test" / "reviewed.png"
    Image.new("RGB", (1024, 640)).save(img)
    labels = dataset / "labels" / "test" / "reviewed.txt"
    labels.write_text("0 0.35 0.55 0.1 0.7\n")
    digest = _hash(pcm_path.read_bytes())
    row = {
        "split": "test", "source_kind": "live", "source_generator": None,
        "capture_id": "capture-real", "capture_sha256": digest,
        "sample_count": 8000, "sample_start": 0, "sample_rate": 8000,
        "lineage_group": _hash(b"real-receiver-1"),
        "box_count": 1, "box_types": ["CW"],
        "image": "images/test/reviewed.png",
        "image_sha256": _hash(img.read_bytes()),
        "label_sha256": _hash(labels.read_bytes()),
    }
    train = {
        **row, "split": "train", "capture_id": "train-1",
        "capture_sha256": _hash(b"train"), "lineage_group": _hash(b"train-lineage"),
        "image": "images/train/train.png",
    }
    valid = {
        **row, "split": "validation", "capture_id": "valid-1",
        "capture_sha256": _hash(b"validation"), "lineage_group": _hash(b"validation-lineage"),
        "image": "images/validation/validation.png",
    }
    (dataset / "windows.jsonl").write_text(
        "\n".join(json.dumps(x) for x in (train, valid, row)) + "\n"
    )
    (dataset / "export-report.json").write_text(
        json.dumps({"schema_version": 2, "preprocessing_version": "sig-id-live-waterfall-v2"})
    )
    with sqlite3.connect(roots / "dataset.sqlite3") as db:
        db.execute(
            "CREATE TABLE captures (id TEXT,sha256 TEXT,audio_path TEXT,"
            "sample_count INTEGER,sample_rate INTEGER,channels INTEGER,source_kind TEXT)"
        )
        db.execute("INSERT INTO captures VALUES (?,?,?,?,?,?,?)",
                   ("capture-real", digest, str(pcm_path), 8000, 8000, 1, "live"))
    return dataset, roots, pcm_path, row


class _FakeDetector:
    model_sha256 = "f" * 64
    confidence_threshold = 0.5
    def __init__(self):
        self.called = 0
    def detect(self, image, *, sample_start, sample_count, sample_rate):
        self.called += 1
        return [{
            "frequency_low_hz": 900.0, "frequency_high_hz": 1200.0,
            "center_hz": 1050.0, "sample_start": 800,
            "sample_count": 5600, "confidence": .95,
        }]


def test_test_manifest_cannot_contain_train_or_validation_lineages(tmp_path):
    dataset, _, _, _ = _fixture(tmp_path)
    rows, report = load_frozen_manifest(dataset, "test")
    assert len(rows) == 1
    assert report["lineage_group_source_counts"] == {"live": 1}
    assert report["all_cross_split_source_checks_passed"] is True
    contents = (dataset / "windows.jsonl").read_text().splitlines()
    train = json.loads(contents[0])
    test = json.loads(contents[2])
    test["lineage_group"] = train["lineage_group"]
    contents[2] = json.dumps(test)
    (dataset / "windows.jsonl").write_text("\n".join(contents) + "\n")
    with pytest.raises(ValueError, match="source leakage"):
        load_frozen_manifest(dataset, "test")


def test_heldout_benchmark_checks_originals_and_keeps_group_macro(tmp_path):
    dataset, roots, wav_path, row = _fixture(tmp_path)
    d = _FakeDetector()
    result = benchmark_reviewed_chain(
        dataset, [roots], tmp_path / "fake-model.pt",
        detector_override=d, split="test", verify_pixels=False,
    )
    assert d.called == 1
    assert result["metrics"]["iou_0_50"]["tp"] == 1
    assert result["metrics"]["iou_0_50"]["recall"] == 1
    group = result["metrics"]["source_breakdown"]["live"]
    assert group["positive_lineage_macro_recall"] == 1
    assert group["ground_truth_boxes"] == 1
    assert result["metrics"]["signal_type_recall"]["CW"]["matched"] == 1
    assert result["metrics"]["native_decoder"]["correct_payloads"] is None
    assert result["data_integrity"]["verified_audio_hashes"] == 1
    assert len(resolve_sources([row], [roots])) == 1

    # Neither an edited source WAV nor edited reviewed image is accepted.
    wav_path.write_bytes(wav_path.read_bytes() + b"!\x00")
    with pytest.raises(ValueError, match="held-out audio hash"):
        benchmark_reviewed_chain(
            dataset, [roots], tmp_path / "fake-model.pt",
            detector_override=d, verify_pixels=False,
        )




def test_shadow_snapshot_must_remain_bitwise_frozen(tmp_path):
    dataset, roots, _, _ = _fixture(tmp_path)
    snapshot = dataset / "source_snapshot"
    snapshot.mkdir()
    dbcopy = snapshot / "dataset.sqlite3"
    dbcopy.write_bytes((roots / "dataset.sqlite3").read_bytes())
    report_path = dataset / "export-report.json"
    report = json.loads(report_path.read_text())
    report["source_snapshot_sha256"] = _hash(dbcopy.read_bytes())
    report_path.write_text(json.dumps(report))
    _, checks = load_frozen_manifest(dataset, "test")
    assert checks["all_cross_split_source_checks_passed"]
    dbcopy.write_bytes(dbcopy.read_bytes() + b"unexpected-change")
    with pytest.raises(ValueError, match="SQLite snapshot hash changed"):
        load_frozen_manifest(dataset, "test")


def test_independent_payload_truth_is_only_used_after_decoder(tmp_path, monkeypatch):
    dataset, roots, _, row = _fixture(tmp_path)
    original = tmp_path / "reviewed-payloads.json"
    original.write_text(json.dumps({
        "schema_version": 1,
        "references": [{
            "capture_sha256": row["capture_sha256"],
            "sample_start": 1600,
            "sample_count": 3200,
            "expected_text": "TEST PAYLOAD",
            "independently_reviewed": True,
        }],
    }))
    refs = load_payload_references(original)
    seen = []

    def fake_decode(pcm, rate, sample_start, carrier, modes):
        seen.append((len(pcm), rate, sample_start, carrier, modes))
        return {"grade": "coherent-unverified", "text": "TEST PAYLOAD",
                "cpu_ms": 1.0}

    monkeypatch.setattr("tools.bench_reviewed_chain.decode_live_candidate", fake_decode)
    result = benchmark_reviewed_chain(
        dataset, [roots], tmp_path / "fake-model.pt",
        detector_override=_FakeDetector(), verify_pixels=False,
        decode_source="real", max_decode_attempts=1, payload_references=refs,
    )
    assert seen
    assert seen[0][1] == 8000
    assert result["metrics"]["native_decoder"]["payload_truth_eligible"] == 1
    assert result["metrics"]["native_decoder"]["correct_payloads"] == 1
    assert "TEST PAYLOAD" not in json.dumps(result)

    unreviewed = json.loads(original.read_text())
    unreviewed["references"][0]["independently_reviewed"] = False
    original.write_text(json.dumps(unreviewed))
    with pytest.raises(ValueError, match="unreviewed"):
        load_payload_references(original)

def test_heldout_benchmark_rejects_mutated_image_and_missing_source(tmp_path):
    dataset, roots, _, row = _fixture(tmp_path)
    img = dataset / row["image"]
    img.write_bytes(img.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="held-out image or labels changed"):
        benchmark_reviewed_chain(
            dataset, [roots], tmp_path / "fake-model.pt",
            detector_override=_FakeDetector(), verify_pixels=False,
        )
    (dataset / "images" / "test" / "reviewed.png").unlink()
    with pytest.raises(FileNotFoundError, match="dataset.sqlite3"):
        resolve_sources([row], [tmp_path / "other-data"])
