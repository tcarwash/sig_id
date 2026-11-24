"""Wave A: use available tracked audio for isolated decoder trials."""
from __future__ import annotations

from sigid.api import _bounded_live_trial_span
from sigid.trials import DecoderTrialRunner, plausible_receive_text


def test_cv_trial_uses_four_second_track_context_not_classifier_crop():
    rate = 12000
    snapshot = {
        "sample_rate": rate, "sample_offset": 0,
        "window_sample_count": round(5.5 * rate),
    }
    # The accepted classifier crop is only 2 seconds, but the track has
    # 5.5 seconds of observable audio available.
    track = {"sample_start": round(3.5 * rate), "sample_count": 2 * rate}
    start, count = _bounded_live_trial_span(track, snapshot)
    assert count >= 4 * rate
    assert start <= track["sample_start"]
    assert start + count >= track["sample_start"] + track["sample_count"]
    assert start + count <= snapshot["window_sample_count"]


def test_trial_region_never_reads_expired_audio_or_future_samples():
    rate = 48000
    ring_start = 80 * rate
    snapshot = {
        "sample_rate": rate, "sample_offset": ring_start,
        "window_sample_count": 10 * rate,
    }
    track = {"sample_start": 70 * rate, "sample_count": 19 * rate}
    span = _bounded_live_trial_span(track, snapshot)
    assert span is not None
    start, count = span
    assert start >= ring_start
    assert start + count <= ring_start + 10 * rate
    assert 4 * rate <= count <= 10 * rate
    assert _bounded_live_trial_span({"sample_start": 0, "sample_count": rate}, snapshot) is None


def test_trial_region_defers_incomplete_capture_without_consuming_attempt():
    rate = 12000
    snapshot = {"sample_rate": rate, "sample_offset": 0, "window_sample_count": 2 * rate}
    assert _bounded_live_trial_span(
        {"sample_start": rate, "sample_count": rate}, snapshot,
    ) is None


class _FakeFldigi:
    def __init__(self):
        self.output = b""
        self.mode = "BPSK31"
        self.selections = []

    def mode_names(self):
        return ["BPSK31", "RTTY", "THOR16"]

    def select_mode(self, name):
        self.mode = name
        self.selections.append(name)
        return name

    def receive_text_length(self):
        return len(self.output)

    def receive_text(self, *, start=0, length=65536):
        return self.output[start : start + length]


def test_noisy_or_partial_bytes_cannot_terminate_all_mode_trials():
    assert not plausible_receive_text("X")
    assert not plausible_receive_text("\x00\ufffd")
    assert plausible_receive_text("CQ CQ DE K7ABC")
    now = [0.0]
    fd = _FakeFldigi()
    runner = DecoderTrialRunner(fd, dwell_seconds=6, clock=lambda: now[0])
    assert runner.start(track_id="session:track", modulation_hypothesis="FSK-like", live=True)["current_mode"] == "RTTY"

    fd.output += b"X"
    assert runner.advance()["status"] == "running"
    assert runner.advance()["status"] == "running"
    now[0] = 6.0
    advanced = runner.advance()
    assert advanced["current_mode"] == "THOR16"
    assert advanced["attempted_modes"] == ["RTTY", "THOR16"]

    fd.output += b"CQ CQ DE K7ABC"
    accepted = runner.advance()
    assert accepted["status"] == "tentative_output"
    assert accepted["text"] == "CQ CQ DE K7ABC"
    assert accepted["current_mode"] == "THOR16"
