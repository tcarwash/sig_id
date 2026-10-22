import assert from "node:assert/strict";
import test from "node:test";
import { REVIEWED_MODE_CANDIDATES, narrowKnownModes } from "../sigid/web/mode-candidates.mjs";

test("observed modulation narrows the reviewed Fldigi mode subset with reasons", () => {
  const candidates = narrowKnownModes({ modulation: "MFSK" });
  assert.deepEqual(candidates.map((candidate) => candidate.mode), ["THOR8", "THOR16", "MFSK16", "MFSK32"]);
  assert.ok(candidates.every((candidate) => candidate.reasons[0] === "Matches observed MFSK modulation family"));
  assert.ok(candidates.every((candidate) => candidate.evidence === "Fldigi loopback validated"));
});

test("QPSK shortlist exposes the validated mode without implying complete catalog coverage", () => {
  assert.deepEqual(narrowKnownModes({ modulation: "QPSK" }).map((candidate) => candidate.mode), ["QPSK31"]);
  assert.equal(REVIEWED_MODE_CANDIDATES.length, 10);
});
