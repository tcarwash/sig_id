import assert from "node:assert/strict";
import test from "node:test";
import {
  timeLensConfig, sampleToTimeLensRow, timeLensRowToSample, closestSpectrumRow,
} from "../sigid/web/time-lens.mjs";

const view = {timeStartSample: 0, timeEndSample: 60_000, sampleRate: 1000};

test("lens shows 60 seconds, linear mode stays at 30", () => {
  assert.equal(timeLensConfig("linear").seconds, 30);
  assert.equal(timeLensConfig("gentle").seconds, 60);
  assert.equal(timeLensConfig("strong").seconds, 60);
  assert.equal(timeLensConfig("unknown").seconds, 30);
});

for (const mode of ["linear", "gentle", "strong"]) {
  test(mode + " maps samples and pointer rows back with sub-sample precision", () => {
    for (const ageSeconds of [0, .01, .3, 1, 2, 5, 10, 15, 30, 45, 60]) {
      const sample = view.timeEndSample - ageSeconds * view.sampleRate;
      const row = sampleToTimeLensRow(sample, view, 800, mode);
      assert.ok(row >= -1e-9 && row <= 800 + 1e-9);
      assert.ok(Math.abs(timeLensRowToSample(row, view, 800, mode) - sample) < 1e-8);
    }
  });
}
test("last five seconds get more pixels, older history compresses", () => {
  const recent = view.timeEndSample - 5_000;
  const linearY = sampleToTimeLensRow(recent, view, 800, "linear");
  const gentleY = sampleToTimeLensRow(recent, view, 800, "gentle");
  const strongY = sampleToTimeLensRow(recent, view, 800, "strong");
  assert.ok(linearY < gentleY && gentleY < strongY);
  assert.ok(strongY > 200);
  assert.ok(sampleToTimeLensRow(view.timeEndSample - 50_000, view, 800, "strong") < 800);
});
test("nearest original spectrogram row is stable, handles bounds and gaps", () => {
  const samples = [10, 20, 30, 80, 90, 100];
  assert.equal(closestSpectrumRow(samples, 24), 1);
  assert.equal(closestSpectrumRow(samples, 28), 2);
  assert.equal(closestSpectrumRow(samples, -3), 0);
  assert.equal(closestSpectrumRow(samples, 200), 5);
  assert.equal(closestSpectrumRow(samples, 19, 2, 4), 2);
  assert.equal(closestSpectrumRow([], 5), -1);
});
test("invalid time bounds fail instead of creating NaN coordinates", () => {
  assert.throws(() => sampleToTimeLensRow(5, {...view, timeEndSample: 0}, 100), RangeError);
  assert.throws(() => timeLensRowToSample(5, view, 0), RangeError);
});
