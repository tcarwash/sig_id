// Stable source ownership for asynchronous decoder output.
export function decodeSelectionKey(context) {
  const source = context.liveView ? context.liveSessionId : context.capture?.id;
  const selection = context.selection;
  if (!source || !selection) return null;
  return JSON.stringify([context.liveView ? "live" : "capture", source,
    selection.sample_start, selection.sample_count,
    selection.frequency_low_hz ?? null, selection.frequency_high_hz ?? null]);
}

export function mayPresentDecodeResult(owner, current) {
  return Boolean(owner?.selectionKey && owner.runId
    && owner.selectionKey === current.selectionKey
    && owner.runId === current.runId && current.outputTab === "text");
}
