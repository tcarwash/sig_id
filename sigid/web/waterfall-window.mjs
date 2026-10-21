/** Sample bounds for the live waterfall's trailing time window. */
export function liveTimeWindow(sampleCount, sampleRate, windowSeconds = 10) {
  if (!Number.isSafeInteger(sampleCount) || sampleCount < 0) {
    throw new RangeError("sampleCount must be a non-negative safe integer");
  }
  if (!Number.isSafeInteger(sampleRate) || sampleRate <= 0) {
    throw new RangeError("sampleRate must be a positive safe integer");
  }
  if (!Number.isFinite(windowSeconds) || windowSeconds <= 0) {
    throw new RangeError("windowSeconds must be positive and finite");
  }
  const endSample = sampleCount;
  const startSample = Math.max(0, endSample - Math.round(sampleRate * windowSeconds));
  return { startSample, endSample };
}

/** Number of FFT rows needed to represent the entire live time window. */
export function liveRowLimit(sampleRate, hopSize, windowSeconds = 10) {
  if (!Number.isSafeInteger(sampleRate) || sampleRate <= 0) {
    throw new RangeError("sampleRate must be a positive safe integer");
  }
  if (!Number.isSafeInteger(hopSize) || hopSize <= 0) {
    throw new RangeError("hopSize must be a positive safe integer");
  }
  if (!Number.isFinite(windowSeconds) || windowSeconds <= 0) {
    throw new RangeError("windowSeconds must be positive and finite");
  }
  return Math.ceil(sampleRate * windowSeconds / hopSize) + 1;
}

/** Relative time-axis labels for a scrolling live view. */
export function liveTimeLabels(visibleStartSample, sampleCount, sampleRate) {
  if (!Number.isSafeInteger(visibleStartSample) || visibleStartSample < 0) {
    throw new RangeError("visibleStartSample must be a non-negative safe integer");
  }
  if (!Number.isSafeInteger(sampleCount) || sampleCount < visibleStartSample) {
    throw new RangeError("sampleCount must be a safe integer at or after visibleStartSample");
  }
  if (!Number.isSafeInteger(sampleRate) || sampleRate <= 0) {
    throw new RangeError("sampleRate must be a positive safe integer");
  }
  const elapsedSeconds = (sampleCount - visibleStartSample) / sampleRate;
  return { start: `−${elapsedSeconds.toFixed(1)} s`, end: "Now" };
}
