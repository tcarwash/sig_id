"""Deterministic receive-only CW overlap-assembly bench across SNR and seeds."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from sigid.cw_token_stream import CWTokenStream
from sigid.native_live_decode import decode_live_candidate
from sigid.signal_lab import render
from tools.bench_signal_chain_snr import read_mono


POLLS = (8,11,14,17,20)
NOISE_POLLS = (32,35,38,41,44)
EXPECTED = "CQ TEST"
VALID_WORDS = {"CQ","TEST"}


def score_case(*, rate:int, snr:float, seed:int, folder:Path)->dict:
    manifest=render(modes=["cw"],output=folder/f"cw-{rate}-{snr}-{seed}.wav",
                    duration=45,sample_rate=rate,seed=seed,traffic="sparse",
                    bursts_per_mode=1,burst_seconds=16,stereo=False,qrm=False,
                    snr_db=snr,snr_reference_hz=2500)
    pcm,rate,digest=read_mono(manifest["mono_wav"])
    [burst]=manifest["transmissions"]
    carrier=burst["carrier_hz"]
    assert burst["start_sample"]+burst["sample_count"]<30*rate
    positive=CWTokenStream("synthetic-positive","cw",rate,carrier)
    negative=CWTokenStream("synthetic-negative","cw",rate,carrier)
    def poll(state,periods):
        events=[]
        for stop_seconds in periods:
            hi=stop_seconds*rate
            lo=hi-8*rate
            result=decode_live_candidate(
                pcm[lo:hi].tobytes(),rate,lo,carrier,("cw",))
            event=state.feed(result["decoder_result"],
                             sample_rate=rate,carrier_hz=carrier)
            events.append({
                "start_sample":lo,"end_sample":hi,
                "window_grade":result["grade"],
                "window_text":result["text"],
                "delta_text":event["delta_text"],
                "delta_words":event["delta_words"],
                "assembled_text":event["cumulative_text"],
                "reason":event["reason"],
                "worker_cpu_ms":result["cpu_ms"],
            })
        return events
    events=poll(positive,POLLS)
    negatives=poll(negative,NOISE_POLLS)
    all_words=[w for event in events for w in event["delta_words"]]
    source_keys=[(w["start_sample"],w["end_sample"]) for w in all_words]
    transcript=positive.last_text
    return {
        "seed":seed,"snr_db":snr,"sample_rate":rate,
        "mono_sha256":digest,
        "fixture_sha256":burst["fixture_sha256"],
        "reference_bandwidth_hz":2500,
        "expected_text":EXPECTED,
        "emitted_text":transcript,
        "contains_exact_phrase":EXPECTED in transcript,
        "emitted_words":len(all_words),
        "incorrect_emitted_words":sum(w["text"] not in VALID_WORDS for w in all_words),
        "duplicate_absolute_sample_intervals":len(source_keys)-len(set(source_keys)),
        "noise_only_emitted_words":sum(len(event["delta_words"]) for event in negatives),
        "cpu_ms_total":round(sum(e["worker_cpu_ms"] for e in events+negatives),2),
        "positive_events":events,
        "noise_events":negatives,
        "not_on_air_validation":True,
    }


def summary(cases:list[dict])->dict:
    groups={}
    for snr in sorted({c["snr_db"] for c in cases}):
        subset=[c for c in cases if c["snr_db"]==snr]
        groups[str(snr)]={
            "cases":len(subset),
            "exact_phrase_present":sum(c["contains_exact_phrase"] for c in subset),
            "incorrect_word_count":sum(c["incorrect_emitted_words"] for c in subset),
            "duplicate_source_interval_count":sum(c["duplicate_absolute_sample_intervals"] for c in subset),
            "noise_only_word_count":sum(c["noise_only_emitted_words"] for c in subset),
            "median_cpu_ms_ten_polls":round(float(np.median([c["cpu_ms_total"] for c in subset])),2),
        }
    return groups


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snrs",default="-10,-5,0,10,25")
    parser.add_argument("--seeds",type=int,default=6)
    parser.add_argument("--seed-start",type=int,default=71062)
    parser.add_argument("--rate",type=int,choices=(12000,48000),default=48000)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    snrs=[float(x) for x in args.snrs.split(",")]
    if (not snrs or len(snrs)>12
            or any(not np.isfinite(x) or x< -30 or x>55 for x in snrs)
            or not 1<=args.seeds<=20):
        parser.error("need <=12 finite SNRs in -30..55 and 1..20 seeds")
    cases=[]
    with tempfile.TemporaryDirectory(prefix="sigid-cw-stream-") as tmp:
        for seed in range(args.seed_start,args.seed_start+args.seeds):
            for snr in snrs:
                c=score_case(rate=args.rate,snr=snr,seed=seed,folder=Path(tmp))
                cases.append(c)
                print("seed",seed,"SNR",snr,"exact",c["contains_exact_phrase"],
                      "incorrect_words",c["incorrect_emitted_words"],
                      "duplicates",c["duplicate_absolute_sample_intervals"],
                      "noise_emitted",c["noise_only_emitted_words"],flush=True)
    report={"schema_version":1,"provenance":"synthetic-signal-lab-only",
            "not_on_air_validation":True,"sample_rate":args.rate,
            "seed_start":args.seed_start,"seed_count":args.seeds,"snrs":snrs,
            "summary":summary(cases),"cases":cases}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["summary"],indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
