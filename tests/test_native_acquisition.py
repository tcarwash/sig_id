"""Conservative intra-window acquisition of native modem messages."""
from __future__ import annotations

import numpy as np
import pytest

from sigid.native_acquisition import propose_burst_windows
from sigid.native_live_decode import decode_live_candidate
from sigid.signal_lab import render
from tools.bench_signal_chain_snr import read_mono


@pytest.mark.parametrize("mode,snr_db", [
    ("rtty", 25.0),
    ("rtty", 10.0),
    ("rtty", 0.0),
    ("bpsk31", 25.0),
    ("bpsk31", 10.0),
])
def test_live_burst_acquisition_recovers_embedded_complete_message(tmp_path, mode, snr_db):
    manifest = render(
        modes=[mode], output=tmp_path / (mode + ".wav"),
        duration=30, sample_rate=12000, seed=71062, traffic="sparse",
        bursts_per_mode=1, burst_seconds=9, snr_db=snr_db,
        snr_reference_hz=2500, qrm=False, stereo=False,
    )
    pcm, rate, _ = read_mono(manifest["mono_wav"])
    [burst] = manifest["transmissions"]
    begin = burst["start_sample"] + round(0.35 * rate)
    four_sec = pcm[begin:begin + 4 * rate]
    expected = burst["embedded_messages"][0]["expected_text"]
    # Ground truth is only used to score the production decoder's output.
    # Window starts partway through a message, but contains a later *whole*
    # independent message. Correct acquisition must not rely on those offsets.
    assert any(message["sample_start"] >= begin
               and message["sample_start"] + message["sample_count"] <= begin + len(four_sec)
               for message in burst["embedded_messages"])
    spans = propose_burst_windows(
        four_sec.astype(float) / 32768, rate, burst["carrier_hz"], mode)
    assert 1 <= len(spans) <= 2
    result = decode_live_candidate(
        four_sec.tobytes(), rate, begin, burst["carrier_hz"], (mode,))
    assert result["grade"] == "coherent-unverified"
    assert result["text"] == expected
    attempts = result["attempted"][0]["acquisition"]["attempts"]
    assert attempts[-1]["kind"] == "energy-gap-segment"
    assert 0 < len(attempts) <= 2
    chosen = result["decoder_result"]["live_acquisition"]
    assert chosen["selected_sample_start"] >= begin
    assert chosen["selected_sample_count"] < len(four_sec)
    assert chosen["confirmed_identity"] is False
    assert result["wall_ms"] >= 0 and result["cpu_ms"] >= 0


@pytest.mark.parametrize("mode", ["rtty", "bpsk31"])
def test_acquisition_abstains_on_noise_and_does_not_multiply_work(mode):
    rate = 12000
    noise = np.random.default_rng(81321).normal(0, 0.09, rate * 4)
    assert propose_burst_windows(noise, rate, 740 if mode == "rtty" else 2520, mode) == ()
    pcm = np.rint(np.clip(noise, -1, 1) * 32767).astype("<i2").tobytes()
    result = decode_live_candidate(
        pcm, rate, 100_000, 740 if mode == "rtty" else 2520, (mode,))
    assert result["grade"] == "none"
    assert len(result["attempted"][0]["acquisition"]["attempts"]) == 1
    assert result["attempted"][0]["acquisition"]["attempts"][0]["kind"] == "full-window"


def test_acquisition_does_not_split_continuous_single_tone_or_unsupported_modes():
    rate = 12000
    t = np.arange(rate * 4) / rate
    tone = 0.25 * np.sin(2 * np.pi * 740 * t)
    assert propose_burst_windows(tone, rate, 740, "rtty") == ()
    assert propose_burst_windows(tone, rate, 740, "cw") == ()
    assert propose_burst_windows(tone, rate, 740, "ax25") == ()
    assert propose_burst_windows(tone[:rate // 4], rate, 740, "rtty") == ()


def test_acquisition_rejects_invalid_carrier_and_retains_original_positions():
    rate = 12000
    a = np.zeros(rate * 4)
    assert propose_burst_windows(a, rate, 40, "rtty") == ()
    assert propose_burst_windows(a, rate, rate/2-5, "bpsk31") == ()
    assert propose_burst_windows(a, rate, 740, "rtty") == ()
