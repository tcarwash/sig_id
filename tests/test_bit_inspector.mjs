import assert from "node:assert/strict";
import test from "node:test";
import {
  bytesForBits,
  bitstreamMetrics,
  transformBitstream,
  bitSelectionRange,
  framingCandidates,
  createBitAnnotation,
} from "../sigid/web/bit-inspector.mjs";

test("bit view groups bytes with aligned printable ASCII and partial tails", () => {
  assert.deepEqual(bytesForBits("0110000100000001"), [
    { bit_start: 0, bit_count: 8, binary: "01100001", hex: "61", ascii: "a", lsb_hex: "86", lsb_ascii: "." },
    { bit_start: 8, bit_count: 8, binary: "00000001", hex: "01", ascii: ".", lsb_hex: "80", lsb_ascii: "." },
  ]);
  assert.deepEqual(bytesForBits("101")[0], {
    bit_start: 0, bit_count: 3, binary: "101", hex: "·", ascii: "·", lsb_hex: "·", lsb_ascii: "·",
  });
  assert.deepEqual(bytesForBits("10x"), []);
});

test("byte alignment skips leading bits but keeps offsets in the original stream", () => {
  assert.deepEqual(bytesForBits("1100110010110101", 3), [
    { bit_start: 3, bit_count: 8, binary: "01100101", hex: "65", ascii: "e", lsb_hex: "A6", lsb_ascii: "." },
    { bit_start: 11, bit_count: 5, binary: "10101", hex: "·", ascii: "·", lsb_hex: "·", lsb_ascii: "·" },
  ]);
  assert.throws(() => bytesForBits("101", 8), RangeError);
  assert.throws(() => bytesForBits("101", 1.5), RangeError);
});

test("bitstream transforms expose polarity and differential/NRZI conventions", () => {
  assert.equal(transformBitstream("001011", "raw"), "001011");
  assert.equal(transformBitstream("001011", "inverted"), "110100");
  assert.equal(transformBitstream("001011", "differential-xor", 0), "001110");
  assert.equal(transformBitstream("001011", "differential-xor", 1), "101110");
  assert.equal(transformBitstream("001011", "differential-xnor", 0), "110001");
  assert.throws(() => transformBitstream("0102", "raw"), TypeError);
  assert.throws(() => transformBitstream("0101", "unknown"), RangeError);
});

test("bit range selection respects the full partial tail regardless of click order", () => {
  assert.deepEqual(bitSelectionRange({ bit_start: 8, bit_count: 2 }, { bit_start: 0, bit_count: 8 }), { start: 0, end: 10 });
  assert.deepEqual(bitSelectionRange({ bit_start: 0, bit_count: 8 }, { bit_start: 8, bit_count: 2 }), { start: 0, end: 10 });
});

test("bitstream metrics report descriptive entropy, balance, transitions, runs and bytes", () => {
  assert.deepEqual(bitstreamMetrics("00110110"), {
    bit_count: 8,
    one_fraction: 0.5,
    bit_entropy_bits: 1,
    transition_fraction: 4 / 7,
    longest_run: 2,
    byte_alignment: 0,
    byte_count: 1,
    byte_entropy_bits: 0,
    printable_byte_fraction: 1,
    lsb_printable_byte_fraction: 1,
  });
  assert.equal(bitstreamMetrics("00110110", 3).byte_alignment, 3);
  assert.equal(bitstreamMetrics("101").byte_entropy_bits, null);
  assert.equal(bitstreamMetrics("00000000").bit_entropy_bits, 0);
  assert.equal(bitstreamMetrics("00101110").printable_byte_fraction, 1);
  assert.equal(bitstreamMetrics("01100001").lsb_printable_byte_fraction, 0);
});

test("recognizes an exact HDLC-like flag as a candidate, without claiming a mode", () => {
  assert.deepEqual(framingCandidates("100111111000"), [
    { bit_start: 2, bit_count: 8, label: "HDLC-like flag · frame delimiter", evidence: "Common delimiter, not unique to one protocol; bit order and framing are unverified. Possible protocols: HDLC family; also used by protocols such as AX.25.", stencil_id: "hdlc-flag" },
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
