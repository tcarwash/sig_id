from __future__ import annotations

import pytest

from sigid.dataset import CaptureStore
from sigid.synthetic import generate_2fsk_wav
from sigid.trials import FldigiTrialRunner


class FakeFldigi:
    def __init__(self, modes=("BPSK31", "RTTY", "THOR16")):
        self.modes = list(modes)
        self.mode = "BPSK31"
        self.output = b"old text"
        self.selections = []

    def modem_names(self):
        return self.modes

    def mode_names(self):
        return self.modem_names()

    def select_modem(self, name):
        if name not in self.modes:
            raise ValueError(name)
        self.mode = name
        self.selections.append(name)
        return name

    def select_mode(self, name):
        return self.select_modem(name)

    def receive_text_length(self):
        return len(self.output)

    def receive_text(self, *, start=0, length=65536):
        return self.output[start : start + length]


def test_trial_runner_advances_by_hypothesis_and_stops_on_new_tentative_text():
    now = [100.0]
    client = FakeFldigi()
    runner = FldigiTrialRunner(client, dwell_seconds=6, clock=lambda: now[0])

    started = runner.start(track_id="session:0001", modulation_hypothesis="FSK-like", live=True)
    assert started["current_mode"] == "RTTY"
    assert started["attempted_modes"] == ["RTTY"]
    assert client.selections == ["RTTY"]
    assert runner.advance()["status"] == "running"

    now[0] += 6
    advanced = runner.advance()
    assert advanced["current_mode"] == "THOR16"
    assert advanced["attempted_modes"] == ["RTTY", "THOR16"]

    client.output += b" decoded words"
    tentative = runner.advance()
    assert tentative["status"] == "tentative_output"
    assert tentative["text"] == " decoded words"
    assert tentative["current_mode"] == "THOR16"
    assert "no protocol label" in tentative["note"]

    with pytest.raises(ValueError, match="already had"):
        runner.start(track_id="session:0001", modulation_hypothesis="FSK-like", live=True)


def test_trial_runner_is_live_only_catalog_scoped_and_rate_limited():
    now = [0.0]
    client = FakeFldigi(modes=("RTTY",))
    runner = FldigiTrialRunner(
        client,
        max_starts_per_minute=1,
        clock=lambda: now[0],
    )
    with pytest.raises(ValueError, match="active live capture"):
        runner.start(track_id="one", modulation_hypothesis="FSK-like", live=False)

    started = runner.start(track_id="one", modulation_hypothesis="FSK-like", live=True)
    assert started["modes"] == ["RTTY"]
    runner.cancel()
    with pytest.raises(ValueError, match="rate limit"):
        runner.start(track_id="two", modulation_hypothesis="unknown", live=True)

    now[0] = 61
    with pytest.raises(ValueError, match="already had"):
        runner.start(track_id="one", modulation_hypothesis="FSK-like", live=True)


def test_decode_output_persists_with_track_window_and_links_to_saved_capture(tmp_path):
    now = [100.0]
    client = FakeFldigi()
    store = CaptureStore(tmp_path / "dataset.sqlite3")
    runner = FldigiTrialRunner(client, run_store=store, clock=lambda: now[0])
    started = runner.start(
        track_id="session:track-4",
        modulation_hypothesis="FSK-like",
        live=True,
        session_id="session",
        sample_start=3200,
        sample_count=48000,
        sample_rate=48000,
    )
    assert started["run_id"]
    client.output += b"TRACK WINDOW OUTPUT"
    received = runner.advance()
    assert received["status"] == "tentative_output"

    audio = tmp_path / "capture.wav"
    generate_2fsk_wav(audio, "010101", sample_rate=8000, symbol_rate=100)
    capture_id = store.add_wav_capture(audio, source_kind="live")
    assert store.associate_decoder_runs("session", capture_id) == 1

    reopened = CaptureStore(tmp_path / "dataset.sqlite3")
    [run] = reopened.decoder_runs(capture_id=capture_id)
    assert run["id"] == started["run_id"]
    assert run["track_id"] == "session:track-4"
    assert run["sample_start"] == 3200
    assert run["sample_count"] == 48000
    assert run["sample_rate"] == 48000
    assert run["current_mode"] == "RTTY"
    assert run["output_text"] == "TRACK WINDOW OUTPUT"
    assert run["output_scope"] == "shared_live_input"
    assert run["association"] == "track_window_tentative"
