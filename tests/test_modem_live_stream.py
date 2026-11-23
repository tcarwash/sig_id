"""Live RTTY/PSK31/AX.25 event deduplication and sample provenance."""
from __future__ import annotations

import copy
import json
import wave
from pathlib import Path

import pytest

from sigid.modem_live_stream import ModemEvidenceStream
from sigid.native_live_decode import decode_live_candidate, explicit_modem_hypothesis

FIXTURES = Path(__file__).parent / "fixtures" / "decoder_conformance"


def positive(mode):
    manifest = json.loads((FIXTURES / mode / "manifest.json").read_text())
    case = next(x for x in manifest["fixtures"] if x["expected"]["kind"] == "positive")
    with wave.open(str(FIXTURES / mode / case["audio"]["path"]), "rb") as source:
        pcm=source.readframes(source.getnframes())
        rate=source.getframerate()
    result=decode_live_candidate(pcm, rate, 0, case["carrier_hz"], (mode,))
    assert result["grade"] in ("coherent-unverified", "frame-integrity-passed")
    return result["decoder_result"], result["grade"], rate, case["carrier_hz"], len(pcm)//2


@pytest.mark.parametrize("mode,expected", [
    ("rtty", "CQ TEST"), ("bpsk31", "cq de"), ("ax25", "HELLO WORLD")
])
def test_live_modem_first_event_is_independent_and_no_overlap_duplication(mode, expected):
    raw,grade,rate,carrier,length=positive(mode)
    s=ModemEvidenceStream("s","s:one",mode,rate,carrier)
    out=s.feed(raw, sample_start=0, sample_count=length, grade=grade, carrier_hz=carrier)
    assert out["delta_text"] == expected
    assert out["grade"]==grade
    assert not out["confirmed_identity"]
    assert out["new_items"] == (1 if mode=="ax25" else len(raw["tokens"]))
    assert out["cumulative_text"] == expected
    duplicate=s.feed(raw,sample_start=0,sample_count=length,grade=grade,carrier_hz=carrier)
    assert duplicate["grade"]=="none"
    assert duplicate["new_items"]==0
    assert duplicate["delta_text"]==""
    assert duplicate["suppressed_overlap_items"]>=1
    assert duplicate["cumulative_text"]==expected

    # The same message at a *different source time* is a new transmission.
    later=copy.deepcopy(raw)
    shift=rate*5
    for field in ("tokens","frames"):
        for token in later[field]:
            token["start_sample"]+=shift
            token["end_sample"]+=shift
    second=s.feed(later,sample_start=shift,sample_count=length,grade=grade,carrier_hz=carrier)
    assert second["delta_text"]==expected
    assert second["new_items"]==(1 if mode=="ax25" else len(raw["tokens"]))
    assert second["cumulative_text"]==(expected+" | "+expected if mode=="ax25"
                                       else expected+"\n"+expected)
    assert second["emitted_frames"] == (2 if mode=="ax25" else 0)


@pytest.mark.parametrize("mode",["rtty","bpsk31","ax25"])
def test_live_modem_data_never_publishes_unverified_or_invalid_spans(mode):
    raw,grade,rate,carrier,length=positive(mode)
    s=ModemEvidenceStream("s","s:one",mode,rate,carrier)
    missing=s.feed(raw,sample_start=0,sample_count=length,grade="none",carrier_hz=carrier)
    assert missing["delta_text"]==""
    assert missing["grade"]=="none"
    wrong=copy.deepcopy(raw)
    wrong["decoder_id"]="cw"
    assert not s.feed(wrong,sample_start=0,sample_count=length,
                      grade=grade,carrier_hz=carrier)["delta_text"]
    wrong=copy.deepcopy(raw)
    for item in wrong["tokens"]:
        item["start_sample"]+=length*2
        item["end_sample"]+=length*2
    for item in wrong["frames"]:
        item["start_sample"]+=length*2
        item["end_sample"]+=length*2
    assert not s.feed(wrong,sample_start=0,sample_count=length,
                      grade=grade,carrier_hz=carrier)["delta_text"]
    if mode == "ax25":
        no_crc=copy.deepcopy(raw)
        no_crc["frames"][0]["integrity"][0]["outcome"]="failed"
        assert not s.feed(no_crc,sample_start=0,sample_count=length,
                          grade=grade,carrier_hz=carrier)["delta_text"]


@pytest.mark.parametrize("mode",["rtty","bpsk31","ax25"])
def test_modem_stream_carrier_and_session_guard(mode):
    raw,grade,rate,carrier,length=positive(mode)
    s=ModemEvidenceStream("s","s:one",mode,rate,carrier)
    assert not s.compatible(session_id="another",track_id="s:one",mode=mode,
                            sample_rate=rate,carrier_hz=carrier)
    assert not s.compatible(session_id="s",track_id="s:two",mode=mode,
                            sample_rate=rate,carrier_hz=carrier)
    assert not s.compatible(session_id="s",track_id="s:one",mode=mode,
                            sample_rate=rate,carrier_hz=carrier+150)
    with pytest.raises(ValueError,match="carrier"):
        s.feed(raw,sample_start=0,sample_count=length,grade=grade,carrier_hz=carrier+150)
    with pytest.raises(ValueError,match="source window"):
        s.feed(raw,sample_start=0,sample_count=rate*9,grade=grade,carrier_hz=carrier)


@pytest.mark.parametrize("mode",["rtty","bpsk31","ax25"])
def test_named_model_suggestion_allows_budgeted_followups(mode):
    name={"rtty":"RTTY","bpsk31":"BPSK31","ax25":"AX.25"}[mode]
    prop={"available":True,"abstained":False,"method":name}
    assert explicit_modem_hypothesis({"classifier_proposal":prop})==mode
    assert explicit_modem_hypothesis({"classifier_proposal":{
        "available":True,"abstained":True,"method":name}}) is None
    assert explicit_modem_hypothesis({"classifier_proposal":{
        "available":True,"abstained":True,"ranked_candidates":[
            {"mode_id":mode,"relative_model_score":.75},
            {"mode_id":"cw","relative_model_score":.2}]}})==mode
