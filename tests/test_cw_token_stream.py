"""CW token overlap state, strict track boundaries, and real spawned worker."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from sigid.cw_token_stream import CWTokenStream, _candidate_words
from sigid.dataset import CaptureStore
from sigid.native_live_decode import NativeLiveRunner, decode_live_candidate
from sigid.signal_lab import render
from tools.bench_signal_chain_snr import read_mono


def _result(start: int, end: int, words: list[tuple[str, int, int]], rate=12000):
    tokens=[]
    for text,begin,stop in words:
        stride=(stop-begin)//(len(text)+1)
        for i,char in enumerate(text):
            a=begin+i*stride
            tokens.append({"text":char, "pattern":".-" if char=="A" else "-",
                           "start_sample":a,"end_sample":a+stride//2,
                           "confidence_hint":.92})
    return {
        "decoder_id":"cw","status":"decoded",
        "evidence":{"dot_ms":60},
        "tokens":tokens,
        "live_acquisition":{
            "selected_sample_start":start,
            "selected_sample_count":end-start,
            "confirmed_identity":False,
        },
    }


def test_confirmed_words_only_emitted_once_from_overlapping_windows():
    rate=12000
    # First window ends at t=8, second at t=11. Two source words are
    # identical in absolute time, despite different acquisition origins.
    words=[("TEST",int(4*rate),int(5*rate)),
           ("CQ",int(6*rate),int(7*rate))]
    state=CWTokenStream("s","s:cv:a",rate,740)
    first=state.feed(_result(0,8*rate,words),sample_rate=rate,carrier_hz=740)
    assert first["delta_text"]==""
    second=state.feed(_result(3*rate,11*rate,words),sample_rate=rate,carrier_hz=740)
    assert second["delta_text"]=="TEST CQ"
    assert second["cumulative_text"]=="TEST CQ"
    assert len(second["delta_words"])==2
    assert second["confirmed_identity"] is False
    assert second["not_verified"] is True
    same=state.feed(_result(3*rate,11*rate,words),sample_rate=rate,carrier_hz=740)
    assert same["delta_text"]==""
    later=state.feed(_result(5*rate,13*rate,words[1:]),sample_rate=rate,carrier_hz=740)
    assert later["delta_text"]==""
    assert state.last_text=="TEST CQ"


def test_source_gaps_carrier_drift_and_session_crossovers_cannot_join_words():
    state=CWTokenStream("session-a","session-a:cv:1",12000,740)
    first=_result(0,96000,[("TEST",40000,54000)])
    state.feed(first,sample_rate=12000,carrier_hz=740)
    with pytest.raises(ValueError,match="stable rate"):
        state.feed(_result(30000,126000,[("TEST",40000,54000)]),
                   sample_rate=48000,carrier_hz=740)
    with pytest.raises(ValueError,match="stable rate"):
        state.feed(_result(30000,126000,[("TEST",40000,54000)]),
                   sample_rate=12000,carrier_hz=760)
    with pytest.raises(ValueError,match="overlap lost"):
        state.feed(_result(97000,193000,[("TEST",120000,134000)]),
                   sample_rate=12000,carrier_hz=740)


def test_untrusted_partial_unknown_or_edge_words_never_publish():
    rate=12000
    state=CWTokenStream("s","track",rate,800)
    bad=_result(0,8*rate,[("TEST",int(.05*rate),int(.75*rate))])
    bad["tokens"][0]["confidence_hint"]=.1
    bad["tokens"][1]["text"]="?"
    assert _candidate_words(bad,rate)==()
    assert state.feed(bad,sample_rate=rate,carrier_hz=800)["delta_text"]==""
    later=_result(3*rate,11*rate,[("TEST",int(4*rate),int(5*rate))])
    assert state.feed(later,sample_rate=rate,carrier_hz=800)["delta_text"]==""
    assert state.last_text==""
    assert all(not x.get("confirmed_identity") for x in [bad["live_acquisition"]])


def test_noise_only_cw_has_no_incremental_text():
    rate=12000
    noise=np.random.default_rng(20261).normal(0,.04,rate*24)
    pcm=np.rint(np.clip(noise,-1,1)*32767).astype("<i2")
    state=CWTokenStream("s","track",rate,740)
    for t in (8,11,14,17,20):
        start=(t-8)*rate
        result=decode_live_candidate(
            pcm[start:t*rate].tobytes(),rate,start,740,("cw",))
        assert not state.feed(result["decoder_result"],
                              sample_rate=rate,carrier_hz=740)["delta_text"]


def _wait(runner,session,timeout=15):
    limit=time.monotonic()+timeout
    while time.monotonic()<limit:
        state=runner.status(session)
        if not state["busy"]:
            return state
        time.sleep(.04)
    raise AssertionError("native worker did not settle")


def test_real_spawned_live_cw_session_emits_incremental_not_repeated_text(tmp_path):
    rate=12000
    data=render(modes=["cw"],output=tmp_path/"cw-stream.wav",
                duration=30,sample_rate=rate,seed=71062,traffic="sparse",
                bursts_per_mode=1,burst_seconds=16,snr_db=25,
                snr_reference_hz=2500,qrm=False,stereo=False)
    pcm,_,_=read_mono(data["mono_wav"])
    carrier=data["transmissions"][0]["carrier_hz"]
    session="stream-real"
    track={
        "track_id":session+":cv:cw", "track_state":"persistent",
        "detector_source":"cv", "sample_start":0,
        "sample_count":8*rate,
        "frequency_low_hz":carrier-20,
        "frequency_high_hz":carrier+20,
        "classifier_proposal":{"available":True,"abstained":False,"method":"Morse"},
    }
    store=CaptureStore(tmp_path/"runs.db")
    runner=NativeLiveRunner(max_attempts_per_minute=5)
    events=[]
    try:
        for index,end_seconds in enumerate((8,11,14,17,20)):
            n=end_seconds*rate
            snap={
                "session_id":session,"active":True,
                "sample_rate":rate,"channels":1,
                "sample_offset":max(0,n-10*rate),
                "window_sample_count":min(n,10*rate),
                "sample_count":n,
                "pcm":pcm[max(0,n-10*rate):n].tobytes(),
            }
            track["sample_start"]=max(0,n-8*rate)
            track["sample_count"]=min(n,8*rate)
            if index:
                runner._last_started-=4
                runner._recent_tracks[track["track_id"]]-=4
            trial=runner.submit(session_id=session,track=track,snapshot=snap,store=store)
            assert trial["status"]=="queued",trial
            _wait(runner,session)
            events.append(runner.status(session)["recent"][0])
        assert any(e.get("cw_stream_delta_text") for e in events),events
        concatenated=" ".join(e.get("cw_stream_delta_text","") for e in events if e.get("cw_stream_delta_text"))
        assert "CQ" in concatenated or "TEST" in concatenated
        assert all(e.get("confirmed_identity") is False for e in events)
        runs=store.decoder_runs(session_id=session)
        assert len(runs)==5
        assert all(r["decoder_result"]["cw_stream"]["confirmed_identity"] is False for r in runs)
        assert all(r["output_text"]==r["decoder_result"]["cw_stream"]["delta_text"] for r in runs)
        assert runner.status(session)["attempts_last_minute"]==5
        assert runner.status(session)["max_cw_stream_attempts_per_minute"]==16
        assert runner.status(session)["cpu_ms_last_minute"]<3000
        # A real session change invalidates any previously assembled text.
        new=runner.status("next-session")
        assert new["cw_stream_tracks"]==0 and new["recent"]==[]
    finally:
        runner.close()


def test_secondary_or_wide_cw_candidate_keeps_original_rate_gate(tmp_path):
    from concurrent.futures import Future
    class Held:
        def __init__(self):self.future=Future()
        def submit(self,*args):return self.future
    session="wide"
    rate=12000
    track={"track_id":session+":cv:cw","track_state":"persistent",
           "sample_start":0,"sample_count":rate*4,
           "frequency_low_hz":650,"frequency_high_hz":800,
           "classifier_proposal":{"available":True,"abstained":False,"method":"Morse"}}
    snap={"active":True,"session_id":session,"sample_rate":rate,"channels":1,
          "sample_offset":0,"sample_count":rate*4,
          "window_sample_count":rate*4,"pcm":b"\0\0"*(rate*4)}
    store=CaptureStore(tmp_path/"other.db")
    executor=Held()
    runner=NativeLiveRunner(executor=executor)
    try:
        assert runner.submit(session_id=session,track=track,snapshot=snap,store=store)["status"]=="queued"
        # No queued backlog if same source polls again while a worker is active.
        assert runner.submit(session_id=session,track=track,snapshot=snap,store=store)["status"]=="skipped_busy"
        executor.future.set_exception(RuntimeError("controlled completion"))
        _wait(runner,session)
        runner._last_started-=4
        runner._recent_tracks[track["track_id"]]-=4
        assert runner.submit(session_id=session,track=track,snapshot=snap,store=store)["status"]=="skipped_track"
    finally:
        if not executor.future.done():executor.future.set_exception(RuntimeError("cleanup"))


def test_explicit_lower_native_admission_remains_authoritative():
    runner=NativeLiveRunner(max_attempts_per_minute=1)
    try:
        assert runner.status("budget-session")["max_cw_stream_attempts_per_minute"]==1
        assert runner.status("budget-session")["max_attempts_per_minute"]==1
        assert runner.status("budget-session")["max_cpu_ms_per_minute"]==3000
    finally:
        runner.close()
