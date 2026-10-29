import test from "node:test";
import assert from "node:assert/strict";
import { decodeSelectionKey, mayPresentDecodeResult } from "../sigid/web/decode-context.mjs";

const context = (overrides = {}) => ({
  liveView: false, capture: {id: "capture-a"},
  selection: {sample_start: 10, sample_count: 100, frequency_low_hz: 900, frequency_high_hz: 1300},
  ...overrides,
});

test("decode ownership includes the exact source, time and frequency selection", () => {
  const initial = decodeSelectionKey(context());
  assert.equal(initial, decodeSelectionKey(context()));
  assert.notEqual(initial, decodeSelectionKey(context({capture: {id: "capture-b"}})));
  for (const field of ["sample_start", "sample_count", "frequency_low_hz", "frequency_high_hz"]) {
    const changed = context();
    changed.selection[field] += 1;
    assert.notEqual(initial, decodeSelectionKey(changed), field);
  }
  assert.equal(decodeSelectionKey(context({selection: null})), null);
  assert.equal(decodeSelectionKey(context({capture: null})), null);
  assert.notEqual(initial, decodeSelectionKey(context({liveView: true, liveSessionId: "capture-a"})));
});

test("late results cannot replace another selection, attempt or output view", () => {
  const key = decodeSelectionKey(context());
  const current = {selectionKey: key, runId: "run-a", outputTab: "text"};
  assert.equal(mayPresentDecodeResult({selectionKey: key, runId: "run-a"}, current), true);
  assert.equal(mayPresentDecodeResult({selectionKey: "other", runId: "run-a"}, current), false);
  assert.equal(mayPresentDecodeResult({selectionKey: key, runId: "run-b"}, current), false);
  assert.equal(mayPresentDecodeResult({selectionKey: key, runId: "run-a"}, {...current, outputTab: "raw"}), false);
  assert.equal(mayPresentDecodeResult({selectionKey: null, runId: "run-a"}, {...current, selectionKey: null}), false);
});
