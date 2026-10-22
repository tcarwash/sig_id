from __future__ import annotations

import hashlib
import json

import pytest

from sigid.dataset import CaptureStore, SignalLabel
from sigid.synthetic import generate_2fsk_wav


@pytest.fixture
def indexed_capture(tmp_path):
    audio = tmp_path / "source.wav"
    generate_2fsk_wav(audio, "001101", sample_rate=8000, symbol_rate=100)
    store = CaptureStore(tmp_path / "dataset.sqlite3")
    capture_id = store.add_wav_capture(
        audio,
        source_kind="synthetic",
        source_metadata={"generator": "test-fixture"},
    )
    return store, capture_id, audio


def test_capture_stores_sample_format_hash_and_provenance(indexed_capture):
    store, capture_id, audio = indexed_capture
    capture = store.capture(capture_id)
    assert capture["source_kind"] == "synthetic"
    assert capture["sample_rate"] == 8000
    assert capture["sample_count"] == 480
    assert capture["channels"] == 1
    assert capture["sha256"] == hashlib.sha256(audio.read_bytes()).hexdigest()
    assert capture["source_metadata"] == {
        "generator": "test-fixture",
        "sample_width_bytes": 2,
    }


def test_regions_keep_partial_labels_and_revision_history(indexed_capture, tmp_path):
    store, capture_id, _ = indexed_capture
    region_id = store.add_region(
        capture_id,
        sample_start=80,
        sample_count=240,
        frequency_low_hz=900,
        frequency_high_hz=1400,
        label=SignalLabel(family="FSK", source="synthetic", evidence={"generator": "2fsk"}),
    )
    store.revise_label(
        region_id,
        SignalLabel(
            family="FSK",
            method="binary frequency shift keying",
            source="human",
            review_state="accepted",
            notes="Protocol unknown",
        ),
    )
    history = store.label_history(region_id)
    assert [item["revision"] for item in history] == [1, 2]
    assert history[0]["protocol"] is None
    assert history[0]["evidence"] == {"generator": "2fsk"}
    assert history[1]["review_state"] == "accepted"

    output = tmp_path / "exports" / "regions.jsonl"
    assert store.export_jsonl(output) == 1
    exported = json.loads(output.read_text().strip())
    assert exported["revision"] == 2
    assert exported["family"] == "FSK"
    assert exported["protocol"] is None
    assert exported["sample_start"] == 80
    assert exported["sample_count"] == 240


def test_overlapping_regions_are_independent(indexed_capture):
    store, capture_id, _ = indexed_capture
    left = store.add_region(capture_id, sample_start=0, sample_count=200)
    right = store.add_region(capture_id, sample_start=100, sample_count=200)
    assert left != right
    assert len(store.label_history(left)) == 1
    assert store.label_history(left)[0]["source"] == "unknown"


@pytest.mark.parametrize(
    "start,count",
    [(-1, 10), (0, 0), (479, 2)],
)
def test_rejects_invalid_or_out_of_capture_region(indexed_capture, start, count):
    store, capture_id, _ = indexed_capture
    with pytest.raises(ValueError):
        store.add_region(capture_id, sample_start=start, sample_count=count)


def test_rejects_unknown_capture_and_unpaired_frequency_bounds(indexed_capture):
    store, _, _ = indexed_capture
    with pytest.raises(KeyError, match="unknown capture"):
        store.add_region("missing", sample_start=0, sample_count=1)
    with pytest.raises(ValueError, match="both frequency bounds"):
        store.add_region("some-id", sample_start=0, sample_count=1, frequency_low_hz=100)


def test_bit_annotations_keep_demod_provenance_deduplicate_stream_and_export(indexed_capture, tmp_path):
    store, capture_id, _ = indexed_capture
    settings = {"sample_start": 0, "sample_count": 480, "symbol_rate": 100}
    saved = store.add_bit_annotation(
        capture_id, mode="2fsk", sample_start=0, sample_count=480,
        settings=settings, hypothesis="bits", bits="01100001", bit_start=0,
        bit_count=8, label="ASCII a", color="#f4bf75",
    )
    repeated = store.add_bit_annotation(
        capture_id, mode="2fsk", sample_start=0, sample_count=480,
        settings=settings, hypothesis="bits", bits="01100001", bit_start=0,
        bit_count=8, label="first byte", color="#80c0ff",
    )
    assert saved["bitstream_id"] == repeated["bitstream_id"]
    annotations = store.bit_annotations(capture_id)
    assert [item["label"] for item in annotations] == ["ASCII a", "first byte"]
    assert annotations[0]["stream_sha256"] == saved["stream_sha256"]
    assert annotations[0]["settings"] == settings
    assert "bits" not in annotations[0]
    output = tmp_path / "exports" / "bit-annotations.jsonl"
    assert store.export_bit_annotations_jsonl(output) == 2
    exported = [json.loads(line) for line in output.read_text().splitlines()]
    assert exported[0]["bits"] == "01100001"
    assert exported[0]["capture_sha256"] == store.capture(capture_id)["sha256"]


def test_bit_annotations_reject_ranges_outside_bits_or_capture(indexed_capture):
    store, capture_id, _ = indexed_capture
    arguments = dict(
        mode="2fsk", sample_start=0, sample_count=480,
        settings={"sample_start": 0, "sample_count": 480}, hypothesis="bits",
        bits="0101", bit_start=0, bit_count=1, label="marker", color="#aabbcc",
    )
    with pytest.raises(ValueError, match="exceeds the recovered"):
        store.add_bit_annotation(capture_id, **{**arguments, "bit_start": 4})
    with pytest.raises(ValueError, match="sample range"):
        store.add_bit_annotation(capture_id, **{**arguments, "sample_count": 481, "settings": {"sample_start": 0, "sample_count": 481}})


def test_operator_stencil_is_validated_and_survives_store_reopen(indexed_capture, tmp_path):
    store, capture_id, _ = indexed_capture
    saved = store.add_protocol_stencil(
        name="Training preamble", field="preamble", pattern="1010????1100",
        protocols=["Example protocol", " Example protocol "], source_capture_id=capture_id,
    )
    reopened = CaptureStore(tmp_path / "dataset.sqlite3")
    assert reopened.protocol_stencils() == [{**saved, "protocols": ["Example protocol"], "source": "operator"}]
    for bad_pattern in ("???0", "010", "01xx"):
        with pytest.raises(ValueError):
            store.add_protocol_stencil(name="bad", field="header", pattern=bad_pattern, protocols=[])
