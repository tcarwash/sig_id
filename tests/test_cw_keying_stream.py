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


def test_bounded_carrier_retunes_preserve_one_pass_state_and_cw_words(tmp_path):
    """A drifting signal should not repeatedly decode its 8-second ring."""
    from tools.bench_cw_carrier_tracking import drift_audio
    pcm, rate, carrier = _fixture(tmp_path, snr=0, seed=71062)
    shifted = drift_audio(pcm, rate, 90)
    session = "cw-drift-continuous"
    transcript = []
    events = []
    for end_s in (8, 11, 14, 17, 20, 23, 26, 29):
        start = (end_s-8)*rate
        offset = 90*min(end_s/20, 1)
        result = stream_cw_candidate(
            shifted[start:end_s*rate], rate, start, carrier+offset,
            (session, session+":cw"))
        event = result["decoder_result"]["cw_stream"]
        events.append(event)
        transcript.extend(word["text"] for word in event["delta_words"])
        assert event["confirmed_identity"] is False
        assert event["mixer_phase_continuous"] is True
        assert event["state_reset_reason"] == ("new-stream" if end_s == 8 else None)
    assert "CQ TEST" in " ".join(transcript)
    assert sum(e["processed_new_samples"] for e in events) == 29*rate
    assert events[-1]["carrier_retune_count"] >= 4
    assert abs(events[-1]["carrier_hz"] - (carrier+90)) < 1e-6
    assert events[-1]["initial_carrier_hz"] == carrier+36


def test_source_sample_oscillator_phase_is_continuous_during_retune():
    rate = 12000
    origin = 500*rate
    frequency = 741.3
    stream=CWKeyingStream("s", "s:phase", rate, frequency)
    first=stream.feed(np.zeros(2*rate,dtype="<i2"),sample_start=origin)
    assert first["mixer_phase_continuous"] is True
    phase0=stream._mix_phase
    expected0=(((-2*np.pi*frequency/rate)*origin)
               +(-2*np.pi*frequency/rate)*(2*rate)) % (2*np.pi)
    assert np.isclose(phase0,expected0,atol=1e-8)
    out=stream.feed(np.zeros(2*rate,dtype="<i2"),sample_start=origin,
                    carrier_hz=frequency+4)
    assert out["processed_new_samples"]==0
    assert stream._mix_phase==phase0
    assert out["carrier_retune_count"]==0
    second=stream.feed(np.zeros(2*rate,dtype="<i2"),
                       sample_start=origin+rate,carrier_hz=frequency+4)
    expected1=(phase0+(-2*np.pi*(frequency+4)/rate)*rate)%(2*np.pi)
    assert np.isclose(stream._mix_phase,expected1,atol=1e-8)
    assert second["processed_new_samples"]==rate
    assert second["carrier_retune_count"]==1
    assert second["last_carrier_step_hz"]==4
    with pytest.raises(ValueError,match="carrier jump"):
        stream.feed(np.zeros(rate,dtype="<i2"),
                    sample_start=origin+3*rate,carrier_hz=frequency+40)
    # An exactly contiguous chunk is not a source discontinuity.
    third=stream.feed(np.zeros(rate,dtype="<i2"),
                      sample_start=origin+3*rate,carrier_hz=frequency+7)
    assert third["processed_new_samples"]==rate
    assert third["carrier_retune_count"]==2


def test_worker_resets_on_carrier_jump_rather_than_merging_other_tone(tmp_path):
    pcm, rate, carrier = _fixture(tmp_path,snr=10,seed=71062)
    session="cw-carrier-jump"
    key=(session,session+":cw")
    first=stream_cw_candidate(pcm[:8*rate],rate,0,carrier,key)
    assert first["decoder_result"]["cw_stream"]["state_reset_reason"]=="new-stream"
    second=stream_cw_candidate(pcm[3*rate:11*rate],rate,3*rate,carrier+45,key)
    event=second["decoder_result"]["cw_stream"]
    assert event["state_reset_reason"]=="carrier-jump"
    assert event["processed_new_samples"]==8*rate
    assert event["initial_carrier_hz"]==carrier+45
    assert event["carrier_retune_count"]==0


@pytest.mark.parametrize("seed", [400, 401, 402, 403])
def test_random_duration_keying_does_not_claim_morse(seed):
    from tools.bench_cw_false_positives import irregular_keyed_tone
    rate=12000
    pcm=irregular_keyed_tone(seed,rate)
    stream=CWKeyingStream("negative",f"negative:random:{seed}",rate,740.)
    words,_=_poll(stream,pcm,rate)
    assert not words


@pytest.mark.parametrize("mode", ["rtty", "bpsk31", "ax25"])
@pytest.mark.parametrize("snr", [-5, 0])
def test_weak_wrong_mode_does_not_emit_cw_words(tmp_path,mode,snr):
    manifest=render(modes=[mode],output=tmp_path/f"negative-{mode}-{snr}.wav",
                    duration=30,sample_rate=12000,traffic="sparse",seed=400,
                    bursts_per_mode=1,burst_seconds=16,snr_db=snr,
                    snr_reference_hz=2500,stereo=False,qrm=False)
    pcm,rate,_=read_mono(manifest["mono_wav"])
    carrier=manifest["transmissions"][0]["carrier_hz"]
    stream=CWKeyingStream("negative",f"negative:{mode}:{snr}",rate,carrier)
    words,_=_poll(stream,pcm,rate)
    assert not words


def test_actual_spectral_detector_carrier_tracking_under_audio_drift():
    """No oracle center: detector tracks a +90 Hz synthetic CW sweep."""
    from tools.bench_cw_carrier_tracking import run
    cases=run([0.0],[71062],[90.0])
    case=next(x for x in cases if x["kind"]=="detected")
    assert case["exact_phrase"]
    assert case["incorrect_words"]==0
    assert case["all_words"]==8
    assert sum(e["processed_new_samples"] for e in case["events"])==29*12000
    assert sum(e["measured_candidates"]>0 for e in case["events"])>=4
    assert all(e["state_reset_reason"] in (None,"new-stream")
               for e in case["events"])
