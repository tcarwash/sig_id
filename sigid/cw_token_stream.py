"""Bounded, session/track-scoped CW word accumulation across live polls.

The established CW modem decodes each up-to-eight-second audio window.
This state object corroborates *complete words* from two overlapping decodes,
using original absolute token sample positions, and emits each word once.
It never transforms a hypothesis into verified Morse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

MAX_CW_STREAM_WORDS = 100
MAX_CW_STREAM_TOKENS = 180
MAX_CW_STREAM_TRACKS = 12
MAX_CW_STREAM_DELTA = 80
MAX_CW_STREAM_HISTORY_SECONDS = 16
CW_STREAM_MIN_CHARS = 5


def _candidate_words(result: dict[str, Any], rate: int) -> tuple[dict[str, Any], ...]:
    if result.get("decoder_id") != "cw" or result.get("status") != "decoded":
        return ()
    raw = result.get("tokens") or []
    if not isinstance(raw, list) or not raw:
        return ()
    chosen = result.get("live_acquisition") or {}
    start, count = chosen.get("selected_sample_start"), chosen.get("selected_sample_count")
    if not isinstance(start,int) or not isinstance(count,int) or count<=0:
        return ()
    stop = start+count
    dot_ms = (result.get("evidence") or {}).get("dot_ms")
    dot = (float(dot_ms)*rate/1000 if isinstance(dot_ms,(int,float))
           and math.isfinite(dot_ms) and 15<=dot_ms<=240 else .06*rate)
    tokens=[]
    for x in raw[:MAX_CW_STREAM_TOKENS]:
        if not isinstance(x,dict):
            continue
        letter=x.get("text")
        pattern=x.get("pattern")
        lo,hi=x.get("start_sample"),x.get("end_sample")
        fit=x.get("confidence_hint",0.0)
        if (not isinstance(lo,int) or not isinstance(hi,int)
                or not start<=lo<hi<=stop):
            continue
        valid = (isinstance(letter,str) and len(letter)==1 and letter!="?"
                 and isinstance(pattern,str) and bool(pattern)
                 and not set(pattern)-set(".-") and len(pattern)<=6
                 and isinstance(fit,(int,float)) and math.isfinite(fit) and fit>=.55)
        # Preserve invalid symbols as word-break vetoes, rather than silently
        # compressing a corrupt multi-character word into plausible gibberish.
        tokens.append({"text":letter if valid else "?", "pattern":pattern if valid else "",
                       "start_sample":lo,"end_sample":hi})
    tokens.sort(key=lambda t:(t["start_sample"],t["end_sample"]))
    if not tokens:
        return ()
    groups=[]
    current=[]
    for token in tokens:
        if current and token["start_sample"]-current[-1]["end_sample"]>=5*dot:
            groups.append(current)
            current=[]
        current.append(token)
    if current:groups.append(current)
    guard=max(round(.16*rate),round(2.4*dot))
    words=[]
    for g in groups:
        begin,end=g[0]["start_sample"],g[-1]["end_sample"]
        # Do not publish edge-truncated words; a later overlapping poll can
        # supply an entire copy. A silence at the start is positive context,
        # not a reason to infer a missing prefix.
        if begin < start+guard or end > stop-guard:
            continue
        if any(g[i]["start_sample"] < g[i-1]["end_sample"] for i in range(1,len(g))):
            continue
        if any(t["text"]=="?" for t in g):
            continue
        text="".join(t["text"] for t in g)
        if not 2<=len(text)<=24:
            continue
        words.append({
            "text":text,"start_sample":begin,"end_sample":end,
            "patterns":" ".join(t["pattern"] for t in g),
            "token_count":len(g),
        })
    return tuple(words[-MAX_CW_STREAM_WORDS:])


@dataclass
class CWTokenStream:
    """Stateful reassembly of independently corroborated CW word evidence."""
    session_id: str
    track_id: str
    sample_rate: int
    carrier_hz: float
    last_window_end: int = -1
    last_emitted_end: int = -1
    previous: tuple[dict[str,Any], ...] = ()
    pending: list[dict[str,Any]] = field(default_factory=list)
    issued: list[dict[str,Any]] = field(default_factory=list)
    last_text: str = ""

    def feed(self,result:dict[str,Any],*,sample_rate:int,carrier_hz:float)->dict:
        chosen=result.get("live_acquisition") or {}
        start,count=chosen.get("selected_sample_start"),chosen.get("selected_sample_count")
        stop=start+count if isinstance(start,int) and isinstance(count,int) else None
        if sample_rate!=self.sample_rate or abs(carrier_hz-self.carrier_hz)>8:
            raise ValueError("CW stream requires a stable rate and audio frequency")
        if not isinstance(stop,int) or stop<=start:
            raise ValueError("CW stream requires original source sample provenance")
        if stop<=self.last_window_end:
            return self._event("",(),"non-increasing-window")
        if self.last_window_end >= 0 and start>self.last_window_end:
            raise ValueError("CW window overlap lost: reset the stream")
        words=_candidate_words(result,self.sample_rate)
        tolerance=round(.075*self.sample_rate)
        confirmed=[]
        for word in words:
            if word["end_sample"]<=self.last_emitted_end:
                continue
            match=next((old for old in self.previous
                if old["text"]==word["text"] and old["patterns"]==word["patterns"]
                   and abs(old["start_sample"]-word["start_sample"])<=tolerance
                   and abs(old["end_sample"]-word["end_sample"])<=tolerance),None)
            if match:
                confirmed.append(word)
        self.last_window_end=stop
        self.previous=words
        for word in confirmed:
            if (word["end_sample"]<=self.last_emitted_end
                    or any(abs(word["start_sample"]-p["start_sample"])<=tolerance
                           for p in self.pending)):
                continue
            self.pending.append(word)
        self.pending.sort(key=lambda w:w["start_sample"])
        self.pending=self.pending[-MAX_CW_STREAM_WORDS:]
        if not self.issued and sum(len(w["text"]) for w in self.pending)<CW_STREAM_MIN_CHARS:
            return self._event("",(),"awaiting-corroborated-words")
        fresh=[word for word in self.pending
               if word["end_sample"]>self.last_emitted_end]
        if not fresh:
            return self._event("",(),"duplicate-or-no-new-words")
        delta=" ".join(word["text"] for word in fresh)
        if len(delta)>MAX_CW_STREAM_DELTA:
            return self._event("",(),"oversized-untrusted-delta")
        self.last_emitted_end=max(x["end_sample"] for x in fresh)
        self.pending=[]
        self.issued.extend(fresh)
        self.issued=self.issued[-MAX_CW_STREAM_WORDS:]
        self.last_text=((self.last_text+" ") if self.last_text else "")+delta
        self.last_text=self.last_text[-MAX_CW_STREAM_DELTA*8:]
        return self._event(delta,tuple(fresh),"new-unverified-words")

    def _event(self,delta:str,words:tuple[dict[str,Any],...],reason:str)->dict:
        return {
            "strategy":"two-overlapping-window-cw-word-stream-v1",
            "session_id":self.session_id,"track_id":self.track_id,
            "delta_text":delta,"delta_words":list(words),
            "cumulative_text":self.last_text,
            "last_emitted_end_sample":self.last_emitted_end,
            "last_window_end_sample":self.last_window_end,
            "reason":reason,
            "confirmed_identity":False,
            "message_integrity":"not-available-for-cw",
            "not_verified":True,
        }
