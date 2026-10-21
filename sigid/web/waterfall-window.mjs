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
