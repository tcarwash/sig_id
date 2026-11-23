"""Continuous CW edge stream: real fixture, provenance and negative controls."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sigid.cw_keying_stream import CWKeyingStream
from sigid.cw_keying_worker import stream_cw_candidate
from sigid.signal_lab import render
from tools.bench_signal_chain_snr import read_mono


def _fixture(tmp_path: Path, *, snr: float, seed: int):
    result = render(modes=["cw"], output=tmp_path / ("cw-%s-%s.wav" % (snr, seed)),
                    duration=30, sample_rate=12000, traffic="sparse", bursts_per_mode=1,
                    burst_seconds=16, snr_db=snr, seed=seed, snr_reference_hz=2500,
                    qrm=False, stereo=False)
    pcm, rate, _ = read_mono(result["mono_wav"])
    return pcm, rate, result["transmissions"][0]["carrier_hz"]


def _poll(stream, pcm, rate):
    words = []
    processed = 0
    for end_s in (8, 11, 14, 17, 20, 23, 26, 29):
        end, start = end_s*rate, (end_s-8)*rate
        out = stream.feed(pcm[start:end], sample_start=start)
        assert out["confirmed_identity"] is False and out["not_verified"]
        assert out["strategy"] == "continuous-key-transition-cw-v1"
        words.extend(out["delta_words"])
        processed += out["processed_new_samples"]
        assert stream.feed(pcm[start:end], sample_start=start)["delta_text"] == ""
    return words, processed


@pytest.mark.parametrize("snr", [25, 10, 0, -5])
def test_stream_reconstructs_cq_test_once_without_reprocessing_overlap(tmp_path, snr):
    pcm, rate, carrier = _fixture(tmp_path, snr=snr, seed=71062)
    stream = CWKeyingStream("s", "s:cw", rate, carrier)
    words, processed = _poll(stream, pcm, rate)
    got = " ".join(w["text"] for w in words)
    assert "CQ TEST" in got
    assert processed == 29*rate
    starts = [w["start_sample"] for w in words]
    assert len(starts) == len(set(starts))
    assert starts == sorted(starts)
    assert "CQ TEST" in stream.cumulative_text


def test_noise_only_and_weak_single_false_word_are_suppressed():
    rate = 12000
    for seed in range(8):
        noise = np.random.default_rng(seed).normal(0, 0.035, rate*30)
        pcm = np.rint(np.clip(noise, -1, 1) * 32767).astype("<i2")
        stream = CWKeyingStream("s", f"s:noise:{seed}", rate, 740)
        words, processed = _poll(stream, pcm, rate)
        assert words == [], (seed, words)
        assert processed == 29*rate



@pytest.mark.parametrize("seed", [21, 22, 23, 24, 25, 26])
def test_weak_negative_ten_db_does_not_publish_isolated_false_words(tmp_path, seed):
    pcm, rate, carrier = _fixture(tmp_path, snr=-10, seed=seed)
    stream = CWKeyingStream("s", f"s:weak:{seed}", rate, carrier)
    words, _ = _poll(stream, pcm, rate)
    assert not words


@pytest.mark.parametrize("mode", ["rtty", "bpsk31", "ax25"])
def test_wrong_mode_carrier_does_not_emit_morse(tmp_path, mode):
    manifest = render(modes=[mode], output=tmp_path / f"{mode}.wav",
                      duration=30, sample_rate=12000, traffic="sparse",
                      bursts_per_mode=1, burst_seconds=16, seed=75,
                      snr_db=10, snr_reference_hz=2500, stereo=False, qrm=False)
    pcm, rate, _ = read_mono(manifest["mono_wav"])
    carrier = manifest["transmissions"][0]["carrier_hz"]
    stream = CWKeyingStream("s", f"s:wrong:{mode}", rate, carrier)
    words, _ = _poll(stream, pcm, rate)
    assert not words


def test_gap_out_of_order_and_carrier_reset_are_explicit():
    rate = 12000
    stream = CWKeyingStream("s", "s:cw", rate, 740)
    data = np.zeros(rate, dtype="<i2")
    stream.feed(data, sample_start=0)
    with pytest.raises(ValueError, match="source gap"):
        stream.feed(data, sample_start=rate+1)
    with pytest.raises(ValueError, match="out-of-order"):
        stream.feed(data[:rate//2], sample_start=0)
    with pytest.raises(ValueError, match="unsupported CW sample rate"):
        CWKeyingStream("s", "s:cw", 2000, 740)


def test_worker_keys_session_and_track_and_refuses_cross_session_join(tmp_path):
    pcm, rate, carrier = _fixture(tmp_path, snr=0, seed=71062)
    first = stream_cw_candidate(pcm[:8*rate], rate, 0, carrier, ("one", "one:cw"))
    assert first["decoder_result"]["cw_stream"]["strategy"] == "continuous-key-transition-cw-v1"
    later = stream_cw_candidate(pcm[3*rate:11*rate], rate, 3*rate, carrier, ("one", "one:cw"))
    assert later["decoder_result"]["cw_stream"]["processed_new_samples"] == 3*rate
    assert later["decoder_result"]["cw_stream"]["confirmed_identity"] is False
    other = stream_cw_candidate(pcm[3*rate:11*rate], rate, 3*rate, carrier, ("two", "two:cw"))
    # Different session is a new decoder, consuming the complete eight seconds.
    assert other["decoder_result"]["cw_stream"]["processed_new_samples"] == 8*rate
    assert all(x["grade"] in ("none", "coherent-unverified")
               for x in (first, later, other))
