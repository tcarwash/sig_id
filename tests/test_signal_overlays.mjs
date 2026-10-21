import assert from "node:assert/strict";
import test from "node:test";
import {
  layoutSignalLabels,
  signalCaption,
  hitSignal,
} from "../sigid/web/signal-overlays.mjs";
test("labels retain fixed screen sizes, avoid collisions and prioritize selection", () => {
  const items = Array.from({ length: 12 }, (_, i) => ({
    text: `Signal ${i}`,
    x: 20,
    y: 10,
    selected: i === 11,
  }));
  const { placed, overflow } = layoutSignalLabels(items, 320, 200);
  assert.equal(placed[0].text, "Signal 11");
  assert.ok(overflow.length > 0);
  for (let i = 0; i < placed.length; i++) {
    const a = placed[i];
    assert.ok(a.x >= 0 && a.y + a.height <= 200 && a.x + a.width <= 320);
    for (const b of placed.slice(i + 1))
      assert.ok(
        a.y + a.height <= b.y ||
          b.y + b.height <= a.y ||
          a.x + a.width <= b.x ||
          b.x + b.width <= a.x,
      );
  }
});
test("labels distinguish spectral persistence from identification", () => {
  assert.match(
    signalCaption(
      {
        track_id: "session:0004",
        track_state: "persistent",
        modulation_hypothesis: "MFSK-like",
      },
      0,
    ),
    /^S4 .*candidate$/,
  );
  assert.match(
    signalCaption(
      {
        model_proposal: {
          available: true,
          abstained: false,
          method_id: "4fsk",
        },
      },
      0,
    ),
    /4fsk proposal/,
  );
  assert.doesNotMatch(
    signalCaption(
      {
        model_proposal: { available: true, abstained: true, method_id: "4fsk" },
      },
      0,
    ),
    /4fsk/,
  );
});
test("click hit testing selects the smallest overlapping signal and rejects out of range", () => {
  const wide = {
    sample_start: 100,
    sample_count: 200,
    frequency_low_hz: 500,
    frequency_high_hz: 1500,
  };
  const narrow = {
    sample_start: 120,
    sample_count: 100,
    frequency_low_hz: 900,
    frequency_high_hz: 1000,
  };
  assert.equal(hitSignal([wide, narrow], 150, 950), narrow);
  assert.equal(hitSignal([wide, narrow], 150, 700), wide);
  assert.equal(hitSignal([wide, narrow], 20, 950), null);
});
