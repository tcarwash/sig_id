import assert from "node:assert/strict";
import test from "node:test";

import { liveRowLimit, liveTimeWindow } from "../sigid/web/waterfall-window.mjs";

test("live waterfall expands its startup window until it reaches ten seconds", () => {
  assert.deepEqual(liveTimeWindow(48_000, 48_000), { startSample: 0, endSample: 48_000 });
  assert.deepEqual(liveTimeWindow(480_000, 48_000), { startSample: 0, endSample: 480_000 });
  assert.deepEqual(liveTimeWindow(528_000, 48_000), { startSample: 48_000, endSample: 528_000 });
});

test("live waterfall rejects invalid window inputs", () => {
  assert.throws(() => liveTimeWindow(-1, 48_000), RangeError);
  assert.throws(() => liveTimeWindow(10, 0), RangeError);
  assert.throws(() => liveTimeWindow(10, 48_000, 0), RangeError);
});

test("live waterfall requests enough FFT rows for ten seconds at supported sample rates", () => {
  assert.equal(liveRowLimit(48_000, 512), 939);
  assert.equal(liveRowLimit(192_000, 512), 3751);
  assert.throws(() => liveRowLimit(48_000, 0), RangeError);
});
