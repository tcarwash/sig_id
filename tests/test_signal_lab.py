"""Multi-mode receive-only synthetic audio lab regression checks."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from sigid import signal_lab
from sigid.libsigdecode import create_library, DecodeWindow


def test_catalog_has_only_grounded_pinned_decodable_audio():
    modes = signal_lab.discover()
    assert {"cw", "rtty", "bpsk31", "ax25"} <= modes.keys()
    for fixture in modes.values():
        assert fixture.fixture_id
        assert fixture.expected_text or fixture.expected_kind == "synthetic"
        assert fixture.path.is_file()
        assert len(fixture.sha256) == 64
        samples, rate = signal_lab.read_fixture(fixture)
        assert rate >= 8000 and samples.size > 0


def test_fixture_integrity_catches_tampering(tmp_path):
    original = signal_lab.discover()["rtty"]
    invalid = tmp_path / original.path.name
    data = bytearray(original.path.read_bytes())
    data[-1] ^= 0x01
    invalid.write_bytes(data)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        signal_lab.read_fixture(replace(original, path=invalid))


def test_invalid_missing_and_unsafe_fixture_catalog(tmp_path):
    manifest_dir = tmp_path / "rtty"
    manifest_dir.mkdir()
    source = signal_lab.discover()["rtty"]
    (tmp_path / "outside.wav").write_bytes(source.path.read_bytes())
    fake = {
        "fixtures": [{
            "decoder_id": "rtty",
            "id": "escape",
            "carrier_hz": 1500,
            "expected": {"kind": "positive", "text": "CQ TEST"},
            "provenance": {"kind": "synthetic"},
            "audio": {"path": "../outside.wav", "sha256": source.sha256}
        }]
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(fake))
    assert signal_lab.discover(tmp_path) == {}


def test_layout_uses_distinct_passbands_and_groups_when_needed():
    catalog = signal_lab.discover()
    fixtures = [catalog[x] for x in ("cw", "rtty", "bpsk31", "ax25")]
    lanes = signal_lab.layout(fixtures, {})
    assert [lane.group for lane in lanes] == [0, 0, 0, 0]
    for i, lane in enumerate(lanes):
        assert 160 <= lane.edge_low_hz < lane.edge_high_hz <= 2900
        for other in lanes[i + 1:]:
            assert (lane.edge_high_hz + 85 <= other.edge_low_hz or
                    other.edge_high_hz + 85 <= lane.edge_low_hz)

    groups = signal_lab.layout(fixtures[1:3], {"rtty": 595, "bpsk31": 595})
    assert [station.group for station in groups] == [0, 1]
    collisions = signal_lab.layout(fixtures[1:3], {"rtty": 595, "bpsk31": 595},
                                   overlap=True)
    assert [station.group for station in collisions] == [0, 0]
    with pytest.raises(ValueError, match="cannot fit"):
        signal_lab.layout([catalog["ax25"]], {"ax25": 2800})


def test_carrier_parser_validates_inputs_and_selection():
    assert signal_lab.parse_carriers("rtty=680,bpsk31=1050", ["rtty", "bpsk31"]) == {
        "rtty": 680.0, "bpsk31": 1050.0
    }
    for item in ["rtty=nan", "rtty=120,rtty=350", "ax25=1200", "rtty", "rtty=-1"]:
        with pytest.raises(ValueError):
            signal_lab.parse_carriers(item, ["rtty"])


def test_staggered_render_reproducible_nonoverlap_and_native_decodes(tmp_path):
    modes = ["rtty", "bpsk31", "ax25", "cw"]
    a = tmp_path / "first.wav"
    b = tmp_path / "again.wav"
    options = dict(modes=modes, duration=17, snr_db=20, seed=777)
    info = signal_lab.render(output=a, **options)
    other = signal_lab.render(output=b, **options)

    assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()
    assert info["traffic"] == "staggered"
    assert info["frequency_groups"] == 1
    assert info["stations"] == other["stations"]
    assert [station["mode"] for station in info["stations"]] == modes
    assert len(info["transmissions"]) >= 4
    assert info["transmissions"] == other["transmissions"]
    assert a.with_suffix(".json").is_file()
    assert Path(info["stereo_wav"]).is_file()

    with wave.open(str(a), "rb") as audio:
        assert audio.getframerate() == 48000
        assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
        assert audio.getnframes() == 17 * 48000
        samples = np.frombuffer(audio.readframes(audio.getnframes()),
                                dtype="<i2").astype(float) / 32768.0
    with wave.open(info["stereo_wav"], "rb") as audio:
        assert audio.getnchannels() == 2
        assert audio.getnframes() == len(samples)

    truth = info["transmissions"]
    for previous, following in zip(truth, truth[1:]):
        assert previous["start_sample"] + previous["sample_count"] <= following["start_sample"]

    library = create_library()
    for mode in modes:
        burst = next(item for item in truth if item["mode"] == mode)
        start = burst["start_sample"]
        end = start + burst["sample_count"]
        result = library.decode(mode, DecodeWindow(
            samples[start:end], 48000, sample_start=start,
            carrier_hz=burst["carrier_hz"]))
        assert result.status == "decoded", (mode, result.note, result.evidence)
        assert result.text == burst["expected_text"], (mode, result.text)
        if mode == "ax25":
            assert result.frames and result.frames[0].integrity[0].outcome == "passed"


def test_parallel_render_contains_overlapping_transmissions(tmp_path):
    info = signal_lab.render(
        modes=["cw", "rtty"], output=tmp_path / "crowded.wav",
        duration=16, traffic="parallel", snr_db=12)
    truth = info["transmissions"]
    assert any(
        first["start_sample"] < second["start_sample"] + second["sample_count"]
        and second["start_sample"] < first["start_sample"] + first["sample_count"]
        and first["mode"] != second["mode"]
        for first in truth for second in truth
    )


@pytest.mark.parametrize("max_active", [1, 2])
def test_sparse_loop_limits_active_stations_and_preserves_native_decodes(tmp_path, max_active):
    modes = ["cw", "rtty", "bpsk31", "ax25"]
    output = tmp_path / f"sparse-{max_active}.wav"
    info = signal_lab.render(modes=modes, output=output, duration=100,
                             traffic="sparse", max_active=max_active,
                             bursts_per_mode=2, snr_db=22)
    assert info["traffic"] == "sparse"
    assert info["max_active"] == max_active
    assert info["continuous_qrm"] is False
    assert len(info["transmissions"]) == 8
    assert {s["mode"]: s["carrier_hz"] for s in info["stations"]} == {
        "cw": 260, "rtty": 740, "bpsk31": 2520, "ax25": 1200}
    assert all(sum(b["mode"] == mode for b in info["transmissions"]) == 2 for mode in modes)
    events = []
    for burst in info["transmissions"]:
        start = burst["start_sample"]
        events.extend([(start, 1), (start + burst["sample_count"], -1)])
    active = peak_active = 0
    for _, delta in sorted(events):
        active += delta
        peak_active = max(active, peak_active)
        assert active <= max_active
    assert active == 0 and peak_active == max_active

    if max_active == 2:
        starts = {}
        for burst in info["transmissions"]:
            starts.setdefault(burst["start_sample"], set()).add(burst["mode"])
        assert {frozenset(pair) for pair in starts.values()} == {
            frozenset(("cw", "ax25")), frozenset(("rtty", "bpsk31"))}

    with wave.open(str(output), "rb") as audio:
        samples = np.frombuffer(audio.readframes(audio.getnframes()),
                                dtype="<i2").astype(float) / 32768.0
    library = create_library()
    for mode in modes:
        b = next(item for item in info["transmissions"] if item["mode"] == mode)
        start, stop = b["start_sample"], b["start_sample"] + b["sample_count"]
        result = library.decode(mode, DecodeWindow(
            samples[start:stop], 48000, sample_start=start,
            carrier_hz=b["carrier_hz"]))
        assert result.status == "decoded", (mode, max_active, result.note)
        assert result.text == b["expected_text"], (mode, max_active, result.text)
        if mode == "ax25":
            assert result.frames and result.frames[0].integrity[0].outcome == "passed"


def test_sparse_rejects_invalid_configuration_and_fixture_overcrowding(tmp_path):
    base = dict(modes=["cw", "rtty", "bpsk31", "ax25"], output=tmp_path / "scene.wav",
                traffic="sparse", duration=100)
    with pytest.raises(ValueError, match="max-active"):
        signal_lab.render(**base, max_active=3)
    with pytest.raises(ValueError, match="bursts-per-mode"):
        signal_lab.render(**base, bursts_per_mode=13)
    with pytest.raises(ValueError, match="insufficient room"):
        signal_lab.render(**base, bursts_per_mode=12, max_active=1)
    with_qrm = signal_lab.render(**base, bursts_per_mode=1, qrm=True)
    without_qrm = signal_lab.render(**base, bursts_per_mode=1, qrm=False)
    assert with_qrm["continuous_qrm"] and not without_qrm["continuous_qrm"]


def test_multi_group_reuses_passband_only_in_different_time_slots(tmp_path):
    info = signal_lab.render(
        modes=["rtty", "bpsk31"], output=tmp_path / "split.wav",
        duration=38, carrier_overrides={"rtty": 595, "bpsk31": 595})
    assert info["frequency_groups"] == 2
    assert {station["group"] for station in info["stations"]} == {0, 1}
    assert {burst["mode"] for burst in info["transmissions"]} == {"rtty", "bpsk31"}
    assert min(item["start_seconds"] for item in info["transmissions"]
               if item["mode"] == "bpsk31") >= 19


def test_render_rejects_unknown_modes_and_short_windows(tmp_path):
    with pytest.raises(ValueError, match="no verified"):
        signal_lab.render(modes=["imaginary_modem"], output=tmp_path / "bad.wav")
    with pytest.raises(ValueError, match="duration must be"):
        signal_lab.render(modes=["cw"], output=tmp_path / "short.wav", duration=10)
    with pytest.raises(ValueError, match="traffic"):
        signal_lab.render(modes=["cw"], output=tmp_path / "bad.wav", traffic="unknown")


def test_null_sink_only_routing(monkeypatch):
    operations = []
    created = False

    def fake_pactl(*args):
        nonlocal created
        operations.append(args)
        if args == ("list", "short", "sinks"):
            return ("0\talsa_output.real_speaker\n" +
                    ("123\tsigid_signal_lab\n" if created else ""))
        if args == ("list", "short", "sources"):
            return "123\tsigid_signal_lab.monitor\n" if created else ""
        if args == ("list", "sinks"):
            return ('Sink #123\n    Name: sigid_signal_lab\n'
                    '    Driver: PipeWire\n'
                    '    Properties:\n'
                    '       factory.name = "support.null-audio-sink"\n')
        assert args[0] == "load-module" and args[1] == "module-null-sink"
        assert "sink_name=sigid_signal_lab" in args
        created = True
        return "42\n"

    monkeypatch.setattr(signal_lab, "_pactl", fake_pactl)
    signal_lab.ensure_null_sink()
    assert created
    assert sum(1 for op in operations if op[0] == "load-module") == 1
    signal_lab.ensure_null_sink()
    assert sum(1 for op in operations if op[0] == "load-module") == 1


def test_stop_refuses_unrelated_process_id(tmp_path):
    (tmp_path / "player.pid").write_text(str(__import__("os").getpid()))
    assert signal_lab._worker_pid(tmp_path) is None
    assert signal_lab.stop_loop(tmp_path) is False


def test_cli_catalog_and_status(tmp_path, capsys):
    assert signal_lab.main(["list"]) == 0
    assert "ax25" in capsys.readouterr().out
    assert signal_lab.main(["status", "--state-dir", str(tmp_path)]) == 0
    assert "STOPPED" in capsys.readouterr().out
