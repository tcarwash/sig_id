import assert from "node:assert/strict";
import test from "node:test";
import { PROTOCOL_STENCILS, matchProtocolStencils } from "../sigid/web/protocol-stencils.mjs";

test("built-in flag stencil reports a possible field and offset, not a protocol ID", () => {
  const hits = matchProtocolStencils("1100111111001");
  assert.deepEqual(hits, [{
    stencil_id: "hdlc-flag",
    name: "HDLC-like flag",
    field: "frame delimiter",
    protocols: ["HDLC family; also used by protocols such as AX.25"],
    bit_start: 3,
    bit_count: 8,
    note: "Common delimiter, not unique to one protocol; bit order and framing are unverified.",
  }]);
});

test("custom masks match required bits while ignoring wildcard positions", () => {
  const hits = matchProtocolStencils("00100010", [{ id: "masked", name: "masked", pattern: "?01?" }]);
  assert.deepEqual(hits.map(({ bit_start, bit_count }) => ({ bit_start, bit_count })), [
    { bit_start: 0, bit_count: 4 }, { bit_start: 4, bit_count: 4 },
  ]);
});

test("invalid input and malformed stencils are ignored and scan results are bounded", () => {
  assert.deepEqual(matchProtocolStencils("001x"), []);
  assert.deepEqual(matchProtocolStencils("111", [{ pattern: "bad?" }]), []);
  assert.equal(matchProtocolStencils("0000", [{ id: "zero", pattern: "0" }], { maxMatches: 2 }).length, 2);
  assert.equal(PROTOCOL_STENCILS.length, 1);
  assert.throws(() => matchProtocolStencils("0", [], { maxMatches: 0 }), RangeError);
});
