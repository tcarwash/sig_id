/** Place readable labels without covering each other; omitted items stay in Signals. */
export function layoutSignalLabels(items, width, height) {
  const placed = [];
  const overflow = [];
  for (const item of [...items].sort(
    (a, b) => Number(b.selected) - Number(a.selected),
  )) {
    const w = Math.min(210, Math.max(100, item.text.length * 6.2 + 16), width);
    const h = 24;
    const x = Math.max(0, Math.min(width - w, item.x));
    let y = Math.max(0, item.y);
    while (
      placed.some(
        (p) =>
          x < p.x + p.width + 4 &&
          x + w + 4 > p.x &&
          y < p.y + h + 4 &&
          y + h + 4 > p.y,
      )
    )
      y += h + 4;
    if (y + h > height) {
      overflow.push(item);
      continue;
    }
    placed.push({
      ...item,
      anchorX: item.x,
      anchorY: item.y,
      x,
      y,
      width: w,
      height: h,
    });
  }
  return { placed, overflow };
}

export function signalCaption(track, index) {
  const proposal = track.model_proposal;
  const identity =
    proposal?.available && !proposal.abstained
      ? `${proposal.method_id || proposal.family} proposal`
      : `${track.modulation_hypothesis || "Unknown"} candidate`;
  const shortId = track.track_id
    ? `S${Number(track.track_id.split(":").at(-1)) || index + 1}`
    : `S${index + 1}`;
  return `${shortId} · ${identity}`;
}

export function hitSignal(tracks, sample, frequency) {
  return (
    tracks
      .filter(
        (t) =>
          sample >= t.sample_start &&
          sample <= t.sample_start + t.sample_count &&
          frequency >= t.frequency_low_hz &&
          frequency <= t.frequency_high_hz,
      )
      .sort(
        (a, b) =>
          (a.frequency_high_hz - a.frequency_low_hz) * a.sample_count -
          (b.frequency_high_hz - b.frequency_low_hz) * b.sample_count,
      )[0] || null
  );
}
