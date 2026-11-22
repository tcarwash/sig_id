import assert from "node:assert/strict";
import test from "node:test";
import {matchingDecoderSuggestions} from "../sigid/web/decoder-mode-suggestions.mjs";

const catalog = [
  {id:"rtty", display_name:"RTTY", method:"2-FSK", variant:"RTTY"},
  {id:"bpsk31", display_name:"BPSK31", method:"BPSK", variant:"BPSK31"},
  {id:"bpsk63", display_name:"BPSK63", method:"BPSK", variant:"BPSK63"},
  {id:"cw", display_name:"CW", method:"OOK", variant:"Morse"},
];
const bindings = [
  {backend_id:"sigdecode", mode_id:"rtty", display_name:"RTTY (Baudot)"},
  {backend_id:"sigdecode", mode_id:"bpsk31", display_name:"BPSK31"},
  {backend_id:"sigdecode", mode_id:"cw", display_name:"CW (Morse)"},
];
test("specific FSK and PSK hypotheses can stage registered native decoders without Fldigi", () => {
  const rtty = matchingDecoderSuggestions("2-FSK", catalog, bindings, []);
  assert.deepEqual(rtty.native.map((item) => item.mode_id), ["rtty"]);
  assert.deepEqual(rtty.fldigi, []);
  const bpsk = matchingDecoderSuggestions("bpsk", catalog, bindings, []);
  assert.deepEqual(bpsk.native.map((item) => item.mode_id), ["bpsk31"]);
});
test("only actually available decoders appear, with no guessed fallback modes", () => {
  assert.deepEqual(matchingDecoderSuggestions("unknown", catalog, bindings, ["RTTY"]).native, []);
  assert.deepEqual(matchingDecoderSuggestions(null, catalog, bindings, []).fldigi, []);
  const rtty = matchingDecoderSuggestions("2-FSK", catalog, [], ["RTTY"]);
  assert.deepEqual(rtty.fldigi, ["RTTY"]);
  assert.deepEqual(rtty.native, []);
  const cw = matchingDecoderSuggestions("OOK", catalog, bindings, []);
  assert.deepEqual(cw.native.map((item) => item.mode_id), ["cw"]);
});
test("pure compatibility cannot bypass unavailable or unregistered decoder variants", () => {
  const m = matchingDecoderSuggestions("BPSK", catalog, [], ["BPSK31"]);
  assert.deepEqual(m.fldigi, ["BPSK31"]);
  assert.equal(m.compatibleCount, 2);
});
