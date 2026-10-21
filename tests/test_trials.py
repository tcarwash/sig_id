from __future__ import annotations

import pytest

from sigid.trials import FldigiTrialRunner


class FakeFldigi:
    def __init__(self, modes=("BPSK31", "RTTY", "THOR16")):
        self.modes = list(modes)
        self.mode = "BPSK31"
        self.output = b"old text"
        self.selections = []

    def modem_names(self):
        return self.modes

    def select_modem(self, name):
        if name not in self.modes:
            raise ValueError(name)
        self.mode = name
        self.selections.append(name)
        return name

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
