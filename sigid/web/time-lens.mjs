/** Monotone time-to-pixel map for live waterfalls (newest samples at top).
 * This is presentation-only: absolute samples remain unchanged for selections,
 * CV evidence and stored decoder runs. */
export const TIME_LENS_MODES = Object.freeze({
  linear: Object.freeze({ seconds: 30, focusSeconds: null }),
  gentle: Object.freeze({ seconds: 60, focusSeconds: 8 }),
  strong: Object.freeze({ seconds: 60, focusSeconds: 1.5 }),
});

export function timeLensConfig(mode = "linear") {
  return TIME_LENS_MODES[mode] || TIME_LENS_MODES.linear;
}

function validate(view, height) {
  const span = view.timeEndSample - view.timeStartSample;
  if (!Number.isFinite(span) || span <= 0 || !Number.isFinite(height) || height <= 0)
    throw new RangeError("Invalid time lens bounds");
  return span;
}

/** y = 0 is Now, y = height is the oldest selected sample. */
export function sampleToTimeLensRow(sample, view, height, mode = "linear") {
  const span = validate(view, height);
  const age = (view.timeEndSample - sample) / span;
  const focus = timeLensConfig(mode).focusSeconds;
  if (!focus) return age * height;
  const focusFraction = focus * view.sampleRate / span;
  return Math.log1p(age / focusFraction) / Math.log1p(1 / focusFraction) * height;
}

/** Exact inverse of sampleToTimeLensRow, for pointer/drag and tooltips. */
export function timeLensRowToSample(row, view, height, mode = "linear") {
  const span = validate(view, height);
  const fraction = row / height;
  const focus = timeLensConfig(mode).focusSeconds;
  const age = !focus ? fraction
    : Math.expm1(fraction * Math.log1p(span / (focus * view.sampleRate)))
        * focus * view.sampleRate / span;
  return view.timeEndSample - age * span;
}

/**
 * Pick nearest real FFT row, not a made-up interpolation across a gap.
 * Caller supplies sorted original sample starts and absolute sample time.
 */
export function closestSpectrumRow(sampleStarts, sample, start = 0, end = sampleStarts.length) {
  if (!sampleStarts.length || start >= end) return -1;
  let lo = start, hi = end;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sampleStarts[mid] < sample) lo = mid + 1;
    else hi = mid;
  }
  if (lo >= end) return end - 1;
  if (lo <= start) return start;
  return Math.abs(sampleStarts[lo] - sample) < Math.abs(sample - sampleStarts[lo - 1])
    ? lo : lo - 1;
}
