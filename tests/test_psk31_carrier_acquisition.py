"""Conservative BPSK31 carrier acquisition: evidence, not decode authority."""
from __future__ import annotations
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from sigid.native_live_decode import decode_live_candidate
from sigid.psk31_carrier_acquisition import propose_bpsk31_carrier

ROOT=Path(__file__).parent/"fixtures"/"decoder_conformance"


def positive(mode):
    manifest=json.loads((ROOT/mode/"manifest.json").read_text())
    case=next(x for x in manifest["fixtures"] if x["expected"]["kind"]=="positive")
    with wave.open(str(ROOT/mode/case["audio"]["path"]),"rb") as f:
        pcm=f.readframes(f.getnframes())
        rate=f.getframerate()
    return np.frombuffer(pcm,dtype="<i2").astype(np.float64)/32768,pcm,rate,case["carrier_hz"]


def test_bpsk31_squared_carrier_refinement_recovers_proposal_from_broad_region():
    values,raw,rate,true=positive("bpsk31")
    assert propose_bpsk31_carrier(values,rate,true) is None
    p=propose_bpsk31_carrier(values,rate,true+11)
    assert p and p["strategy"]=="bpsk31-squared-analytic-carrier-v1"
    assert abs(p["proposed_carrier_hz"]-true)<.25
    assert p["peak_to_median"]>=18
    assert p["confirmed_identity"] is False
    result=decode_live_candidate(raw,rate,0,true+11,("bpsk31",))
    refined=result["decoder_result"]["live_acquisition"]["carrier_refinement"]
    assert refined and abs(refined["proposed_carrier_hz"]-true)<.25
    assert result["decoder_result"]["evidence"]["carrier_hz"]==refined["proposed_carrier_hz"]
    assert result["grade"] in ("none","coherent-unverified")
    assert result["decoder_result"]["live_acquisition"]["confirmed_identity"] is False


@pytest.mark.parametrize("mode",["rtty","ax25"])
def test_unrelated_modems_have_no_bpsk_spectral_refinement(mode):
    data,raw,rate,true=positive(mode)
    assert propose_bpsk31_carrier(data,rate,true+11) is None


def test_cw_spectral_line_does_not_count_as_psk31_decode():
    values,raw,rate,true=positive("cw")
    # A single keyed tone also has a square-law carrier line. The proposal
    # is NOT a mode ID; the BPSK31 decoder must still reject this source.
    proposed=propose_bpsk31_carrier(values,rate,true+11)
    assert proposed and proposed["confirmed_identity"] is False
    result=decode_live_candidate(raw,rate,0,true+11,("bpsk31",))
    assert result["grade"]=="none"
    assert not result["text"]


def test_gaussian_noise_silence_and_invalid_inputs_do_not_refine():
    rate=12000
    for seed in range(5):
        values=np.random.default_rng(seed).normal(0,.03,2*rate)
        assert propose_bpsk31_carrier(values,rate,1000) is None
    assert propose_bpsk31_carrier(np.zeros(2*rate),rate,1000) is None
    assert propose_bpsk31_carrier(np.ones(500),rate,1000) is None
    assert propose_bpsk31_carrier(np.zeros(2*rate),rate,rate) is None
