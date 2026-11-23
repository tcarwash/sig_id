"""Bounded receive-only live decoder evidence and CPU admission behavior."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from sigid.dataset import CaptureStore
from sigid.native_live_decode import (
    NativeLiveRunner, candidate_modes, decode_live_candidate, _evidence_grade,
)


FIXTURES = Path(__file__).parent / "fixtures" / "decoder_conformance"


def fixture_snapshot(mode: str):
    manifest = json.loads((FIXTURES / mode / "manifest.json").read_text())
    case = next(item for item in manifest["fixtures"]
                if item["expected"]["kind"] == "positive")
    with wave.open(str(FIXTURES / mode / case["audio"]["path"]), "rb") as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2
        pcm = f.readframes(f.getnframes())
        rate = f.getframerate()
    length = len(pcm) // 2
    low = case["carrier_hz"] - (550 if mode == "ax25" else 90)
    high = case["carrier_hz"] + (550 if mode == "ax25" else 90)
    track = {
        "track_id": "session-1:cv:track-A", "track_state": "persistent",
        "detector_source": "cv", "sample_start": 0,
        "sample_count": length, "sample_rate": rate,
        "frequency_low_hz": low, "frequency_high_hz": high,
        "confidence": 0.75,
        "classifier_proposal": {"available": True, "abstained": False,
                                "method": {"cw": "Morse", "rtty": "RTTY",
                                           "bpsk31": "BPSK31", "ax25": "AX.25"}[mode],
                                "family": {"cw": "CW", "rtty": "FSK",
                                           "bpsk31": "PSK", "ax25": "FSK"}[mode]},
    }
    snapshot = {
        "active": True, "session_id": "session-1", "sample_offset": 0,
        "window_sample_count": length, "sample_rate": rate,
        "channels": 1, "pcm": pcm,
    }
    return track, snapshot, case


@pytest.mark.parametrize("family,method,frequency,span,expected", [
    ("FSK", "RTTY", 1500, 200, ("rtty", "bpsk31")),
    ("PSK", "BPSK31", 1000, 120, ("bpsk31", "cw")),
    ("", "", 2500, 90, ("bpsk31", "cw")),
    ("", "", 650, 90, ("cw", "rtty")),
    ("", "", 1200, 960, ("ax25", "rtty")),
    ("", "", 1200, 290, ("rtty", "bpsk31")),
    ("", "", 265, 300, ("cw", "rtty")),
    ("FSK", "RTTY", 265, 300, ("rtty", "cw")),
    ("", "", 2520, 380, ("bpsk31", "cw")),
    ("CW", "Morse", 2520, 380, ("cw", "bpsk31")),
    ("PSK", "BPSK31", 265, 300, ("bpsk31", "cw")),
    ("FSK", "AX.25", 650, 290, ("ax25", "rtty")),
    ("FSK", "RTTY", 2520, 380, ("rtty", "bpsk31")),
])
def test_mode_hypotheses_are_explicit_and_separate_from_classifier(family, method, frequency, span, expected):
    track = {"frequency_low_hz": frequency - span / 2,
             "frequency_high_hz": frequency + span / 2,
             "classifier_proposal": {"available": bool(family),
                                     "abstained": False,
                                     "family": family, "method": method}}
    assert candidate_modes(track) == expected


@pytest.mark.parametrize("mode,expected", [
    ("cw", "CQ TEST"), ("rtty", "CQ TEST"),
    ("bpsk31", "cq de"), ("ax25", "HELLO WORLD"),
])
def test_real_native_decoder_worker_recovers_pinned_positive(mode, expected):
    track, snapshot, case = fixture_snapshot(mode)
    result = decode_live_candidate(
        snapshot["pcm"], snapshot["sample_rate"], 0,
        case["carrier_hz"], (mode,))
    assert result["selected_mode"] == mode
    assert result["text"] == expected
    assert result["grade"] == (
        "frame-integrity-passed" if mode == "ax25" else "coherent-unverified")
    assert result["cpu_ms"] >= 0 and result["wall_ms"] >= 0
    assert result["attempted"][0]["mode"] == mode
    assert result["decoder_result"]["source"] == "sigid-independent-implementation"
    assert result["decoder_result"]["status"] == "decoded"


def test_native_decoder_results_do_not_claim_verified_identity():
    from sigid.libsigdecode import LocalDecodeResult
    result = LocalDecodeResult(decoder_id="rtty", version="1", status="decoded",
                               text="Q")
    assert _evidence_grade(result) == ("none", "")
    result = LocalDecodeResult(decoder_id="rtty", version="1", status="no_decode")
    assert _evidence_grade(result) == ("none", "")


def wait_finished(runner, store, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = runner.status("session-1")
        if not status["busy"]:
            return status
        time.sleep(0.03)
    raise AssertionError("native test job did not finish")


def test_runner_persists_coherent_results_and_perf_metrics(tmp_path):
    track, snapshot, case = fixture_snapshot("rtty")
    store = CaptureStore(tmp_path / "store.db")
    with ThreadPoolExecutor(max_workers=1) as executor:
        runner = NativeLiveRunner(executor=executor)
        admitted = runner.submit(session_id="session-1", track=track,
                                 snapshot=snapshot, store=store)
        assert admitted["status"] == "queued"
        stats = wait_finished(runner, store)
        runs = store.decoder_runs(session_id="session-1")
        assert len(runs) == 1
        run = runs[0]
        assert run["backend"] == "sigdecode-live-auto"
        assert run["association"] == "live_detector_track_tentative"
        assert run["status"] == "tentative_output"
        assert run["output_text"] == "CQ TEST"
        assert run["decoder_result"]["live_assistance"]["grade"] == "coherent-unverified"
        assert not run["decoder_result"]["live_assistance"]["confirmed_identity"]
        assert stats["attempts_last_minute"] == 1
        assert stats["cpu_ms_last_minute"] > 0
        assert stats["latency_by_mode"]["rtty"]["count"] == 1
        assert stats["recent"][0]["mode"] == "rtty"
        assert runner.for_track("session-1", track["track_id"])["text"] == "CQ TEST"


def test_runner_ignores_untrusted_or_out_of_ring_track(tmp_path):
    track, snapshot, _ = fixture_snapshot("cw")
    store = CaptureStore(tmp_path / "store.db")
    with ThreadPoolExecutor(max_workers=1) as executor:
        runner = NativeLiveRunner(executor=executor)
        for mutate in [
            {"track_state": "historical"},
            {"track_id": "other-session:cv:123"},
            {"sample_start": -1},
            {"sample_start": snapshot["window_sample_count"]},
            {"frequency_low_hz": -3},
            {"frequency_high_hz": 50000},
        ]:
            candidate = {**track, **mutate}
            assert runner.submit(session_id="session-1", track=candidate,
                                 snapshot=snapshot, store=store)["status"] == "not_eligible"
        assert not store.decoder_runs(session_id="session-1")


def test_runner_does_not_repeat_per_track_or_overrun_cpu_budget(tmp_path):
    track, snapshot, _ = fixture_snapshot("rtty")
    store = CaptureStore(tmp_path / "store.db")
    with ThreadPoolExecutor(max_workers=1) as executor:
        runner = NativeLiveRunner(executor=executor, max_cpu_ms_per_minute=1)
        first = runner.submit(session_id="session-1", track=track,
                              snapshot=snapshot, store=store)
        assert first["status"] == "queued"
        wait_finished(runner, store)
        runner._last_started -= 4
        repeated = runner.submit(session_id="session-1", track=track,
                                 snapshot=snapshot, store=store)
        assert repeated["status"] == "skipped_track"
        track2 = {**track, "track_id": "session-1:cv:another"}
        assert runner.submit(session_id="session-1", track=track2,
                             snapshot=snapshot, store=store)["status"] == "skipped_budget"
        assert len(store.decoder_runs(session_id="session-1")) == 1


class HoldingExecutor:
    def __init__(self):
        self.future = Future()

    def submit(self, *args):
        return self.future


def test_runner_never_queues_second_job_when_worker_busy(tmp_path):
    track, snapshot, _ = fixture_snapshot("cw")
    store = CaptureStore(tmp_path / "store.db")
    executor = HoldingExecutor()
    runner = NativeLiveRunner(executor=executor)
    first = runner.submit(session_id="session-1", track=track,
                          snapshot=snapshot, store=store)
    assert first["status"] == "queued"
    track2 = {**track, "track_id": "session-1:cv:another"}
    second = runner.submit(session_id="session-1", track=track2,
                           snapshot=snapshot, store=store)
    assert second["status"] == "skipped_busy"
    assert len(store.decoder_runs(session_id="session-1")) == 1
    assert runner.status("session-1")["counts"]["skipped_busy"] == 1
    executor.future.set_exception(RuntimeError("fault"))
    assert wait_finished(runner, store)["counts"]["failed"] == 1


def test_native_runner_actual_child_process_and_no_decode(tmp_path):
    # Proves the real isolated process is spawn-safe, not merely a thread mock.
    track, snapshot, _ = fixture_snapshot("bpsk31")
    store = CaptureStore(tmp_path / "db.sqlite")
    runner = NativeLiveRunner(max_attempts_per_minute=1)
    try:
        started = runner.submit(session_id="session-1", track=track,
                                snapshot=snapshot, store=store)
        assert started["status"] == "queued"
        status = wait_finished(runner, store, timeout=15)
        assert status["recent"][0]["text"] == "cq de"
        assert status["cpu_ms_last_minute"] >= 0
        assert len(store.decoder_runs(session_id="session-1")) == 1
    finally:
        runner.close()

@pytest.mark.parametrize("top,runner_up,expected", [
    ("fldigi-cw", "bpsk31", "cw"),
    ("bpsk31", "fldigi-cw", "bpsk31"),
    ("rtty", "bpsk31", "rtty"),
])
def test_ranked_abstention_is_only_used_to_order_experimental_trials(top,runner_up,expected):
    from sigid.native_live_decode import _ranked_mode_hint
    proposal={"available":True,"abstained":True,"method":None,
       "ranked_candidates":[
           {"mode_id":top,"relative_model_score":.73},
           {"mode_id":runner_up,"relative_model_score":.24}]}
    assert _ranked_mode_hint(proposal)==expected
    track={"frequency_low_hz":240,"frequency_high_hz":280,
           "model_proposal":proposal}
    assert candidate_modes(track)[0]==expected
    # Ranking is provisional and never a confirmed classifier method.
    assert track["model_proposal"]["method"] is None
    assert track["model_proposal"]["abstained"] is True


def test_ambiguous_or_invalid_ranked_abstentions_fall_back_to_spectrum():
    from sigid.native_live_decode import _ranked_mode_hint
    for scores in ((.5,.48),(.59,.55),(float("nan"),.15)):
        proposal={"available":True,"abstained":True,
           "ranked_candidates":[{"mode_id":"bpsk31","relative_model_score":scores[0]},
                                {"mode_id":"cw","relative_model_score":scores[1]}]}
        assert _ranked_mode_hint(proposal) is None
        track={"frequency_low_hz":240,"frequency_high_hz":280,
               "model_proposal":proposal}
        assert candidate_modes(track)[0]=="cw"


def test_provisional_cw_probe_requires_sustained_audio_and_explicit_hypothesis():
    from sigid.native_live_decode import provisional_probe_kind
    top={"available":True,"abstained":True,
         "ranked_candidates":[{"mode_id":"fldigi-cw","relative_model_score":.72},
                              {"mode_id":"bpsk31","relative_model_score":.25}]}
    track={"track_state":"provisional","track_observations":1,
           "sample_count":int(2.5*12000),"frequency_low_hz":250,
           "frequency_high_hz":270,"peak_snr_db":20,
           "model_proposal":top}
    assert provisional_probe_kind(track,12000)=="provisional_cw"
    assert provisional_probe_kind({**track,"sample_count":12000},12000) is None
    assert provisional_probe_kind({**track,"peak_snr_db":6},12000) is None
    assert provisional_probe_kind({**track,"frequency_high_hz":400},12000) is None
    assert provisional_probe_kind({**track,"model_proposal":{}},12000) is None
    assert provisional_probe_kind({**track,"track_state":"historical"},12000) is None
    assert provisional_probe_kind({**track,"model_proposal":{"available":True,"abstained":True,
      "ranked_candidates":[{"mode_id":"bpsk31","relative_model_score":.7},
                           {"mode_id":"cw","relative_model_score":.2}]}},12000) is None
    wide={**track,"frequency_low_hz":700,"frequency_high_hz":900,
          "model_proposal":{},"track_observations":2}
    assert provisional_probe_kind(wide,12000)=="provisional_modem"


def test_provisional_probe_rate_is_separate_and_never_escapes_cpu_budget(tmp_path):
    from concurrent.futures import Future
    from sigid.native_live_decode import MAX_PROVISIONAL_PROBES_PER_MINUTE
    class Held:
        def __init__(self):self.future=Future()
        def submit(self,*args):return self.future
    rate=12000
    session="probe-session"
    track={"track_state":"provisional","track_id":session+":0001",
           "track_observations":1,"sample_start":0,"sample_count":4*rate,
           "frequency_low_hz":250,"frequency_high_hz":270,"peak_snr_db":20,
           "model_proposal":{"available":True,"abstained":True,
             "ranked_candidates":[{"mode_id":"fldigi-cw","relative_model_score":.72},
                                  {"mode_id":"bpsk31","relative_model_score":.25}]}}
    snap={"active":True,"session_id":session,"sample_rate":rate,"channels":1,
          "sample_offset":0,"window_sample_count":4*rate,
          "pcm":b"\0\0"*(4*rate)}
    store=CaptureStore(tmp_path/"probe.db")
    held=Held()
    runner=NativeLiveRunner(executor=held)
    try:
        accepted=runner.submit(session_id=session,track=track,snapshot=snap,store=store)
        assert accepted["status"]=="queued" and accepted["admission"]=="provisional_cw"
        assert runner.status(session)["provisional_probes_last_minute"]==1
        assert runner.status(session)["max_provisional_probes_per_minute"]==3
        assert runner.submit(session_id=session,track=track,snapshot=snap,store=store)["status"]=="skipped_busy"
        held.future.set_exception(RuntimeError("intentionally held trial"))
        wait_end=time.monotonic()+3
        while runner.status(session)["busy"] and time.monotonic()<wait_end:
            time.sleep(.01)
        assert not runner.status(session)["busy"]
        # No chance to execute another speculative probe when three slots
        # were already consumed, regardless of the relaxed CW cadence.
        runner._provisional_probes.extend([time.monotonic()]*2)
        runner._last_started-=4
        runner._recent_tracks[track["track_id"]]-=4
        denied=runner.submit(session_id=session,track=track,snapshot=snap,store=store)
        assert denied["status"]=="skipped_provisional_budget"
        assert runner.status(session)["counts"]["skipped_provisional_budget"]==1
    finally:
        if not held.future.done():held.future.set_exception(RuntimeError("cleanup"))
        runner.close()
