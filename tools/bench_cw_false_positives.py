"""Synthetic CW false-word controls: noise, irregular keyed tones, and wrong modes.

These are explicit negative controls, not representative field false-positive
rates. Every case uses a fresh stream, deterministic seed and bounded audio.
"""
from __future__ import annotations
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from sigid.cw_keying_stream import CWKeyingStream
from sigid.signal_lab import render
from tools.bench_signal_chain_snr import read_mono

POLL_SECONDS=(8,11,14,17,20,23,26,29)


def decode(pcm: np.ndarray, rate: int, hz: float, label: str) -> dict:
    stream=CWKeyingStream("negative",f"negative:{label}",rate,hz)
    words=[]
    for sec in POLL_SECONDS:
        end=sec*rate;start=end-8*rate
        out=stream.feed(pcm[start:end],sample_start=start)
        words.extend(out["delta_words"])
    return {"label":label, "text":" ".join(x["text"] for x in words),
            "word_count":len(words),"words":words}


def irregular_keyed_tone(seed: int, rate: int, duration: int = 30,
                         carrier: float=740.) -> np.ndarray:
    """An incoherent traffic/control signal with random non-Morse mark lengths."""
    rng=np.random.default_rng(seed)
    n=rate*duration
    audio=rng.normal(0,.028,n)
    t=np.arange(n,dtype=np.float64)/rate
    mask=np.zeros(n,dtype=np.float64)
    pos=int(.5*rate); active=True
    while pos<n-int(.5*rate):
        # Stretches of unpredictable random gate width and silence.
        step=int(rng.uniform(.026,.36)*rate)
        mask[pos:min(pos+step,n)]=active
        active=not active
        pos+=step
    audio+=.12*mask*np.sin(2*np.pi*carrier*t)
    return np.rint(np.clip(audio,-1,1)*32767).astype("<i2")


def run(seeds: list[int], snrs: list[float], rate: int=12000) -> list[dict]:
    results=[]
    with tempfile.TemporaryDirectory(prefix="sigid-cw-neg-") as directory:
        out=Path(directory)
        for seed in seeds:
            noise=np.random.default_rng(seed).normal(0,.035,30*rate)
            pcm=np.rint(np.clip(noise,-1,1)*32767).astype("<i2")
            results.append({"control":"awgn","seed":seed,"snr_db":None,
                            **decode(pcm,rate,740.,f"awgn-{seed}")})
            irregular=irregular_keyed_tone(seed,rate)
            results.append({"control":"irregular-keying","seed":seed,"snr_db":None,
                            **decode(irregular,rate,740.,f"random-key-{seed}")})
            for mode in ("rtty","bpsk31","ax25"):
                for snr in snrs:
                    manifest=render(modes=[mode],output=out/f"{mode}-{seed}-{snr}.wav",
                        duration=30,sample_rate=rate,seed=seed,traffic="sparse",
                        bursts_per_mode=1,burst_seconds=16,snr_db=snr,
                        snr_reference_hz=2500,qrm=False,stereo=False)
                    pcm,_,_=read_mono(manifest["mono_wav"])
                    carrier=manifest["transmissions"][0]["carrier_hz"]
                    results.append({"control":mode,"seed":seed,"snr_db":snr,
                                    **decode(pcm,rate,carrier,f"{mode}-{seed}-{snr}")})
    return results


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",type=int,default=8)
    ap.add_argument("--seed-start",type=int,default=400)
    ap.add_argument("--snrs",default="-5,0")
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    data=run(list(range(args.seed_start,args.seed_start+args.seeds)),
             [float(x) for x in args.snrs.split(",")])
    summary={}
    for x in data:
        key=f"{x['control']} SNR {x['snr_db']}"
        summary.setdefault(key,{"cases":0,"nonempty_cases":0,"false_words":0})
        summary[key]["cases"]+=1
        summary[key]["nonempty_cases"]+=bool(x["word_count"])
        summary[key]["false_words"]+=x["word_count"]
    print(json.dumps(summary,indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps({"synthetic_only":True,"sample_rate":12000,
                                         "summary":summary,"cases":data},indent=2)+"\n")

if __name__=="__main__":
    main()
