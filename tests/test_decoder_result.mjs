import assert from "node:assert/strict";
import test from "node:test";
import { summarizeDecoderResult } from "../sigid/web/decoder-result.mjs";

test("native partial or abstained evidence is displayed without claiming success", () => {
  const result = {
    source: "sigid-independent-implementation",
    decoder_id: "cw", version: "cw-envelope-v1", status: "no_decode",
    text: "", note: "Morse timing inconsistent",
    tokens: [{text: "E", pattern: ".", start_sample: 101, end_sample: 130}],
    evidence: {carrier_hz: 1100, estimated_wpm: 19.8,
      mean_symbol_timing_fit: 0.23, unverified_pattern_preview: "E T"},
  };
  const summary = summarizeDecoderResult(result);
  assert.match(summary, /Abstained/);
  assert.match(summary, /1100 Hz/);
  assert.match(summary, /0.23/);
  assert.match(summary, /E\(\.\)/);
  assert.match(summary, /Unverified interpretation \(NOT a decode\)/);
  assert.match(summary, /not a calibrated probability or confirmed identification/i);
});

test("generic modem evidence is mode-agnostic and does not invent confidence", () => {
  const result = {source:"sigid-independent-implementation", decoder_id:"bpsk31",
    status:"partial", version:"0.1", tokens:[], evidence:{sync_errors:3},
    note:"FEC unavailable"};
  const summary = summarizeDecoderResult(result);
  assert.match(summary, /bpsk31/);
  assert.match(summary, /Partial candidate/);
  assert.doesNotMatch(summary, /timing-fit heuristic/i);
  assert.equal(summarizeDecoderResult({source:"external", status:"decoded"}), "");
  assert.equal(summarizeDecoderResult(null), "");
});


test("live segmentation provenance appears in native and cross-backend result cards", () => {
  const result = {source: "sigid-independent-implementation", decoder_id: "rtty",
    version: "rtty-ita2-fsk-v2", status: "decoded", text: "CQ TEST",
    evidence: {}, live_acquisition: {
      confirmed_identity: false, original_sample_start: 1000,
      original_sample_count: 48000, selected_sample_start: 17400,
      selected_sample_count: 18500,
      attempts: [{kind: "energy-gap-segment", status: "decoded", grade: "coherent-unverified", selected:true}],
    }};
  const summary = summarizeDecoderResult(result);
  assert.match(summary, /Live audio window: samples 1000–49000/);
  assert.match(summary, /Selected decoder window: samples 17400–35900/);
  assert.match(summary, /energy-gap-segment \[selected\].*coherent-unverified/);
  assert.match(summary, /not verified message boundaries/);
  assert.match(summary, /not a calibrated probability or confirmed identification/);
  assert.doesNotMatch(summary, /confirmed mode/i);
});


test("CW incremental words show new delta and persistent unverified transcript", () => {
  const result={source: "sigid-independent-implementation",decoder_id:"cw",
    status:"decoded",text:"CQ TEST CQ",evidence:{},
    cw_stream:{delta_text:"TEST CQ",cumulative_text:"TEST CQ TEST CQ",
      last_emitted_end_sample:164500,reason:"new-unverified-words",
      confirmed_identity:false,not_verified:true}};
  const summary=summarizeDecoderResult(result);
  assert.match(summary,/Incremental CW .* TEST CQ/);
  assert.match(summary,/CW stream so far: TEST CQ TEST CQ/);
  assert.match(summary,/sample: 164500/);
  assert.match(summary,/provisional words, no Morse checksum or operator verification/);
  assert.doesNotMatch(summary,/confirmed mode|verified CW/);
});

test("continuous CW stream explains key-transition evidence without claiming verification", () => {
  const result={source:"sigid-independent-implementation",decoder_id:"cw",
    version:"cw-key-transition-v1",status:"decoded",text:"CQ TEST",evidence:{},
    cw_stream:{strategy:"continuous-key-transition-cw-v1",delta_text:"CQ TEST",
      cumulative_text:"CQ TEST",last_emitted_end_sample:198200,
      reason:"new-unverified-words",confirmed_identity:false,not_verified:true}};
  const summary=summarizeDecoderResult(result);
  assert.match(summary,/Continuous CW \(new key-transition words\): CQ TEST/);
  assert.match(summary,/CW stream so far: CQ TEST/);
  assert.match(summary,/provisional words, no Morse checksum or operator verification/);
  assert.doesNotMatch(summary,/overlap-confirmed|verified CW/);
});

test("continuous CW carrier corrections show unverified provenance", () => {
  const summary=summarizeDecoderResult({
    source:"sigid-independent-implementation",decoder_id:"cw",status:"no_decode",evidence:{},
    cw_stream:{strategy:"continuous-key-transition-cw-v1",delta_text:"",
      cumulative_text:"CQ",reason:"awaiting-complete-word",
      carrier_hz:309.5,initial_carrier_hz:296.0,
      carrier_retune_count:1,last_carrier_step_hz:13.5,
      state_reset_reason:"carrier-jump",confirmed_identity:false}});
  assert.match(summary,/CW audio carrier: 309\.5 Hz \(initial 296\.0 Hz\) · 1 bounded retunes/);
  assert.match(summary,/oscillator phase continuous, carrier identity unverified/);
  assert.match(summary,/CW stream initialized\/reset: carrier-jump/);
  assert.doesNotMatch(summary,/verified station|verified CW/);
});

test("live RTTY/PSK31 modem stream shows only new source tokens", () => {
  const summary=summarizeDecoderResult({source:"sigid-independent-implementation",
    decoder_id:"rtty",status:"decoded",text:"CQ TEST",evidence:{},
    modem_stream:{strategy:"overlap-deduplicated-modem-evidence-v1",
      mode:"rtty",delta_text:" TEST",cumulative_text:"CQ TEST",
      new_items:5,suppressed_overlap_items:2,last_emitted_end_sample:25400,
      confirmed_identity:false}});
  assert.match(summary,/RTTY incremental new text tokens:  TEST/);
  assert.match(summary,/Modem stream so far: CQ TEST/);
  assert.match(summary,/5 new · 2 overlapping duplicates suppressed/);
  assert.match(summary,/sample: 25400/);
  assert.match(summary,/identity remain unverified/);
});

test("live AX25 stream shows CRC-passed new frame, not verified identity", () => {
  const summary=summarizeDecoderResult({source:"sigid-independent-implementation",
    decoder_id:"ax25",status:"decoded",text:"HELLO WORLD",evidence:{},
    modem_stream:{strategy:"overlap-deduplicated-modem-evidence-v1",
      mode:"ax25",delta_text:"HELLO WORLD",cumulative_text:"HELLO WORLD",
      new_items:1,suppressed_overlap_items:0,last_emitted_end_sample:3150,
      confirmed_identity:false}});
  assert.match(summary,/AX25 incremental CRC-checked new packet frames: HELLO WORLD/);
  assert.match(summary,/1 new · 0 overlapping duplicates suppressed/);
  assert.match(summary,/identity remain unverified/);
});

test("BPSK31 carrier refinement is presented as unverified DSP evidence", () => {
  const summary=summarizeDecoderResult({source:"sigid-independent-implementation",
    decoder_id:"bpsk31",status:"no_decode",evidence:{},
    live_acquisition:{original_sample_start:0,original_sample_count:12000,
      selected_sample_start:0,selected_sample_count:12000,
      carrier_refinement:{strategy:"bpsk31-squared-analytic-carrier-v1",
        initial_carrier_hz:1011,proposed_carrier_hz:999.932,
        confirmed_identity:false}}});
  assert.match(summary,/BPSK31 suggested carrier refinement: 1011\.00 → 999\.93 Hz/);
  assert.match(summary,/the decoder must still establish varicode coherence/);
  assert.match(summary,/Abstained/);
});
