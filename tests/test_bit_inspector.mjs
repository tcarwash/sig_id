import assert from "node:assert/strict";
import test from "node:test";
import { bytesForBits, framingCandidates, createBitAnnotation } from "../sigid/web/bit-inspector.mjs";

test("bit view groups bytes with aligned printable ASCII and partial tails", () => {
  assert.deepEqual(bytesForBits("0110000100000001"), [
    { bit_start: 0, bit_count: 8, binary: "01100001", hex: "61", ascii: "a" },
    { bit_start: 8, bit_count: 8, binary: "00000001", hex: "01", ascii: "." },
  ]);
  assert.equal(bytesForBits("101")[0].ascii, "·");
  assert.deepEqual(bytesForBits("10x"), []);
});

test("recognizes an exact HDLC-like flag as a candidate, without claiming a mode", () => {
  assert.deepEqual(framingCandidates("100111111000"), [
    { bit_start: 2, bit_count: 8, label: "HDLC-like flag (0x7E)", evidence: "Exact 8-bit pattern; bit order and protocol are unverified." },
  ]);
  assert.deepEqual(framingCandidates(""), []);
});

test("operator annotation validates and retains range and interpretation", () => {
  assert.deepEqual(createBitAnnotation({ bit_start: 8, bit_count: 16, label: "likely header", hypothesis: "normal" }), {
    bit_start: 8, bit_count: 16, label: "likely header", color: "#f4bf75", hypothesis: "normal", source: "operator",
  });
  assert.throws(() => createBitAnnotation({ bit_start: -1, bit_count: 1, label: "x" }), TypeError);
  assert.throws(() => createBitAnnotation({ bit_start: 0, bit_count: 1, label: " " }), TypeError);
});
