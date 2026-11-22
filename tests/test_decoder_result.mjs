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
