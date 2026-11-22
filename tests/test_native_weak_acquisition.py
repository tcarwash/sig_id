"""CW-specific longer live windows and negative-SNR RTTY acquisition tests."""
from __future__ import annotations

from concurrent.futures import Future

import numpy as np
import pytest

from sigid.dataset import CaptureStore
from sigid.native_acquisition import propose_burst_windows
from sigid.native_live_decode import (
    MAX_CW_WINDOW_SECONDS, MAX_WINDOW_SECONDS, NativeLiveRunner,
    decode_live_candidate,
)
from sigid.signal_lab import render
from tools.bench_signal_chain_snr import read_mono
from tools.bench_weak_acquisition import score_case


@pytest.mark.parametrize("rate", [12000,48000])
def test_negative_five_db_rtty_recovers_complete_message_after_tone_gate(tmp_path, rate):
    row=score_case(mode="rtty", snr_db=-5, seed=71062,
                   sample_rate=rate, output_dir=tmp_path)
    assert row["complete_messages_inside_crop"] >= 1
    assert row["legacy"]["expected_present"] is False
    assert row["updated"]["expected_present"] is True
    assert row["updated"]["text"] == "CQ TEST"
    assert row["updated"]["grade"] == "coherent-unverified"
    assert row["noise"]["false_output"] is False
    acquisition=row["updated"]["acquisition"]
    assert acquisition["selected_sample_count"] < row["sample_count"]
    assert acquisition["selected_sample_start"] >= row["sample_start"]
    assert acquisition["attempts"][0]["kind"] == "energy-gap-segment"
    assert acquisition["confirmed_identity"] is False


@pytest.mark.parametrize("rate", [12000,48000])
@pytest.mark.parametrize("snr", [25.0, 0.0, -5.0])
def test_long_cw_lookback_contains_full_repeated_message(tmp_path, rate, snr):
    row=score_case(mode="cw", snr_db=snr, seed=71062,
                   sample_rate=rate, output_dir=tmp_path)
    assert row["sample_count"] == rate * MAX_CW_WINDOW_SECONDS
    assert row["legacy"]["sample_count"] == rate * MAX_WINDOW_SECONDS
    assert row["initial_four_seconds"]["expected_present"] is False
    # At these controlled SNRs the longer window contains a complete
    # repeated message even though the first four seconds start mid-token.
    assert row["updated"]["expected_present"] is True
    assert row["updated"]["grade"] == "coherent-unverified"
    assert row["noise"]["false_output"] is False
    chosen = row["updated"]["acquisition"]
    assert chosen["strategy"] == "cw-long-or-recent-window-v2"
    assert chosen["confirmed_identity"] is False
    assert len(chosen["attempts"]) == 2
    assert sum(a.get("selected", False) for a in chosen["attempts"]) == 1
    assert chosen["selected_sample_start"] >= row["sample_start"]
    assert chosen["selected_sample_start"] + chosen["selected_sample_count"] <= row["sample_start"] + row["sample_count"]


def test_weak_rtty_noise_only_does_not_generate_segment_trials():
    for rate in (12000,48000):
        for seed in (9,100,81321,71062,72110):
            noise=np.random.default_rng(seed).normal(0,.09,4*rate)
            assert propose_burst_windows(noise,rate,740.0,"rtty") == ()


def test_weak_rtty_does_not_accept_single_tone_as_dual_tone():
    rate=12000
    t=np.arange(rate*4)/rate
    # A bursty keyed CW-looking tone in the RTTY passband is not two-tone FSK.
    key=np.zeros(4*rate);key[rate:3*rate]=1
    value=0.25*key*np.sin(2*np.pi*(740+85)*t)
    assert propose_burst_windows(value,rate,740,"rtty")==()


class _Hold:
    def __init__(self):
        self.future=Future()
        self.args=None
    def submit(self,*args):
        self.args=args
        return self.future


@pytest.mark.parametrize("method,low,high,expected_seconds", [
    ("Morse",200.0,320.0,8),
    ("RTTY",650.0,850.0,4),
    # An accepted RTTY method at a CW-like high pitch must not get the
    # eight-second lookback just because CW is a second fallback mode.
    ("RTTY",200.0,320.0,4),
])
def test_live_runner_cw_only_primary_mode_uses_eight_second_ring(
        tmp_path,method,low,high,expected_seconds):
    rate=12000
    session="bounded-session"
    length=10*rate
    track={
        "track_id":session+":cv:track",
        "track_state":"persistent",
        "detector_source":"cv",
        "sample_start":0,"sample_count":length,
        "frequency_low_hz":low,"frequency_high_hz":high,
        "classifier_proposal":{"available":True,"abstained":False,
                               "method":method,"family":""},
    }
    snapshot={
        "active":True, "session_id":session, "sample_rate":rate,
        "channels":1,"sample_offset":0, "sample_count":length,
        "window_sample_count":length,"pcm":b"\x00\x00"*length,
    }
    store=CaptureStore(tmp_path/"state.db")
    held=_Hold()
    runner=NativeLiveRunner(executor=held)
    try:
        response=runner.submit(session_id=session,track=track,
                               snapshot=snapshot,store=store)
        assert response["status"]=="queued"
        assert held.args is not None
        func,packed,worker_rate,original_start,_,modes=held.args
        assert worker_rate==rate
        assert len(packed)==expected_seconds*rate*2
        assert original_start==length-expected_seconds*rate
        assert modes[0] == ("cw" if method == "Morse" else "rtty")
        [run]=store.decoder_runs(session_id=session)
        assert run["sample_count"]==expected_seconds*rate
        held.future.set_exception(RuntimeError("controlled test completion"))
    finally:
        if not held.future.done():
            held.future.set_exception(RuntimeError("controlled test cleanup"))


def test_worker_rejects_long_non_cw_input():
    with pytest.raises(ValueError, match="bounded"):
        decode_live_candidate(b"\0\0"*(8*12000),12000,0,740,("rtty",))


def test_long_cw_child_process_persists_selected_sample_provenance(tmp_path):
    import time
    rate=12000
    manifest=render(modes=["cw"],output=tmp_path/"cw-extended.wav",
                    duration=30,sample_rate=rate,traffic="sparse",
                    bursts_per_mode=1,burst_seconds=16,snr_db=25,
                    snr_reference_hz=2500,qrm=False,stereo=False)
    pcm,_,_=read_mono(manifest["mono_wav"])
    [burst]=manifest["transmissions"]
    session="extended-cw"
    frame_count=10*rate
    snapshot={"active":True,"session_id":session,"sample_rate":rate,
              "channels":1,"sample_offset":0,"sample_count":frame_count,
              "window_sample_count":frame_count,
              "pcm":pcm[:frame_count].tobytes()}
    start=round(1.35*rate)
    track={"track_id":session+":cv:cw","track_state":"persistent",
           "detector_source":"cv","sample_start":start,
           "sample_count":round(8*rate),
           "frequency_low_hz":burst["carrier_hz"]-25,
           "frequency_high_hz":burst["carrier_hz"]+25,
           "confidence":0.88,
           "classifier_proposal":{"available":True,"abstained":False,"method":"Morse"}}
    store=CaptureStore(tmp_path/"captures.db")
    runner=NativeLiveRunner(max_attempts_per_minute=1)
    try:
        queued=runner.submit(session_id=session,track=track,snapshot=snapshot,store=store)
        assert queued["status"]=="queued"
        assert queued["modes"][0]=="cw"
        deadline=time.monotonic()+20
        completed=runner.status(session)
        while completed["busy"] and time.monotonic()<deadline:
            time.sleep(.04)
            completed=runner.status(session)
        assert not completed["busy"]
        [run]=store.decoder_runs(session_id=session)
        assert run["sample_count"]==8*rate
        evidence=run["decoder_result"]
        assert evidence["source"]=="sigid-independent-implementation"
        chosen=evidence["live_acquisition"]
        assert chosen["strategy"]=="cw-long-or-recent-window-v2"
        assert chosen["original_sample_count"]==8*rate
        assert chosen["confirmed_identity"] is False
        assert chosen["selected_sample_start"]>=run["sample_start"]
        assert chosen["selected_sample_start"]+chosen["selected_sample_count"]<=run["sample_start"]+run["sample_count"]
    finally:
        runner.close()
