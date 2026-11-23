"""Reproducible, receive-only CW carrier-drift/false-word evaluation.

Uses synthetic Signal Lab audio, a smooth analytic-signal carrier shift, and
worker-like overlapping 8-second source windows. Not on-air evidence.
"""
from __future__ import annotations
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

from sigid.cw_keying_worker import stream_cw_candidate
from sigid.signal_lab import render
from sigid.spectrum import waterfall_samples
from sigid.detection import detect_tracks
from tools.bench_signal_chain_snr import read_mono

POLLS = (8, 11, 14, 17, 20, 23, 26, 29)
EXPECTED_WORDS = {"CQ", "TEST"}


def drift_audio(pcm: np.ndarray, rate: int, end_hz: float,
                *, ramp_seconds: float = 20.0) -> np.ndarray:
    """Shift whole analytic mono signal by linear time-varying Hz offset."""
    signal = pcm.astype(np.float64)/32768.
    if end_hz == 0:
        return pcm.copy()
    t = np.arange(len(signal), dtype=np.float64)/rate
    offsets = end_hz*np.minimum(t/ramp_seconds, 1.0)
    phase = 2*np.pi*np.cumsum(offsets)/rate
    shifted = np.real(hilbert(signal)*np.exp(1j*phase))
    return np.rint(np.clip(shifted, -1, 1)*32767).astype("<i2")


def score(pcm: np.ndarray, rate: int, carrier: float,
          *, end_hz: float, seed: int, snr: float, kind: str) -> dict:
    session = f"bench-drift-{seed}-{snr}-{end_hz}-{kind}"
    transcript = []
    step_events = []
    last_center = carrier
    for end in POLLS:
        first = (end-8)*rate
        measured_candidates = 0
        if kind == "detected":
            window = pcm[first:end*rate].astype(np.float64)/32768.
            spectrum = waterfall_samples(window, rate, fft_size=4096,
                hop_size=512, max_rows=320, sample_offset=first,
                total_sample_count=end*rate)
            proposals = detect_tracks(spectrum, threshold_db=18.,
                minimum_peak_db=-32., maximum_tone_gap_hz=35.)
            eligible = [x for x in proposals if x["tone_count_candidate"] == 1
                and x["bandwidth_hz"] <= 45. and x["peak_snr_db"] >= 14.
                and abs(x["center_hz"]-last_center) <= 85.]
            measured_candidates = len(eligible)
            if eligible:
                last_center = max(eligible,key=lambda x: (x["peak_snr_db"],
                    -abs(x["center_hz"]-last_center)))["center_hz"]
            estimated = last_center
        else:
            estimated = carrier+end_hz*min(end/20,1) if kind == "following" else carrier
        outcome = stream_cw_candidate(pcm[first:end*rate], rate, first,
                                      estimated, (session, session+":cw"))
        cw = outcome["decoder_result"]["cw_stream"]
        transcript += [x["text"] for x in cw["delta_words"]]
        step_events.append({"end_second":end,"carrier_estimate_hz":round(estimated,2),
                            "measured_candidates":measured_candidates,
                            "state_reset_reason":cw.get("state_reset_reason"),
                            "words":cw["delta_text"],"reason":cw["reason"],
                            "processed_new_samples":cw["processed_new_samples"]})
    return {"seed":seed,"snr_db":snr,"drift_end_hz":end_hz,"kind":kind,
            "text":" ".join(transcript),
            "exact_phrase":any(transcript[i:i+2]==["CQ","TEST"] for i in range(max(0,len(transcript)-1))),
            "incorrect_words":sum(x not in EXPECTED_WORDS for x in transcript),
            "all_words":len(transcript),
            "events":step_events}


def run(snrs: list[float], seeds: list[int], drifts: list[float],
        *, rate: int = 12000) -> list[dict]:
    cases=[]
    with tempfile.TemporaryDirectory(prefix="sigid-cw-drift-") as tmp:
        path=Path(tmp)
        for seed in seeds:
            for snr in snrs:
                manifest=render(modes=["cw"],output=path/f"cw-{seed}-{snr}.wav",
                                duration=30,sample_rate=rate,seed=seed,traffic="sparse",
                                bursts_per_mode=1,burst_seconds=16,
                                snr_db=snr,snr_reference_hz=2500,
                                qrm=False,stereo=False)
                pcm,rate,_=read_mono(manifest["mono_wav"])
                carrier=manifest["transmissions"][0]["carrier_hz"]
                for drift in drifts:
                    trial=drift_audio(pcm,rate,drift)
                    for kind in ("fixed","following","detected"):
                        cases.append(score(trial,rate,carrier,end_hz=drift,
                                           seed=seed,snr=snr,kind=kind))
    return cases


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",type=int,default=4)
    ap.add_argument("--seed-start",type=int,default=71062)
    ap.add_argument("--snrs",default="0,-5")
    ap.add_argument("--drifts",default="0,30,45")
    ap.add_argument("--output",type=Path)
    ap.add_argument("--rate",type=int,choices=(12000,24000,48000),default=12000)
    args=ap.parse_args()
    cases=run([float(x) for x in args.snrs.split(",")],
              list(range(args.seed_start,args.seed_start+args.seeds)),
              [float(x) for x in args.drifts.split(",")], rate=args.rate)
    for snr in sorted({x["snr_db"] for x in cases}):
        for drift in sorted({x["drift_end_hz"] for x in cases}):
            for kind in ("fixed","following","detected"):
                subset=[x for x in cases if x["snr_db"]==snr and x["drift_end_hz"]==drift and x["kind"]==kind]
                resets=sum(sum(e.get("state_reset_reason") not in (None,"new-stream")
                               for e in x["events"]) for x in subset)
                detector_hits=sum(sum(e.get("measured_candidates",0)>0 for e in x["events"])
                                  for x in subset) if kind=="detected" else None
                print(f"SNR {snr:5g} drift {drift:5g} {kind:9s} correct {sum(x['exact_phrase'] for x in subset)}/{len(subset)} incorrect {sum(x['incorrect_words'] for x in subset)} words {sum(x['all_words'] for x in subset)} stream_resets {resets} detector_hit_polls {detector_hits}",flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps({"synthetic_only":True,"rate":args.rate,
                                          "cases":cases},indent=2)+"\n")
if __name__=="__main__":
    main()
