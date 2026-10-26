/** Place readable labels without covering each other; omitted items stay in Boxes. */
export function layoutSignalLabels(items, width, height) {
  const placed = [];
  const overflow = [];
  for (const item of [...items].sort(
    (a, b) => Number(b.selected) - Number(a.selected),
  )) {
    const w = Math.min(280, Math.max(150, item.labelWidth || 230), width);
    const h = item.labelHeight || (item.text.includes("\n") ? 38 : 24);
    const boxRight = item.boxRight ?? item.x;
    const boxLeft = item.boxLeft ?? item.x;
    const rightX = boxRight + 8;
    const leftX = boxLeft - w - 8;
    // Keep labels beside their boxes, preferring the right and flipping at the edge.
    const preferredX = rightX + w <= width ? rightX : leftX >= 0 ? leftX : rightX;
    const x = Math.max(0, Math.min(width - w, preferredX));
    let y = Math.max(0, Math.min(height - h, item.y));
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
      anchorX: preferredX === leftX ? boxLeft : boxRight,
      anchorY: item.boxCenterY ?? item.y + h / 2,
      leaderToX: preferredX === leftX ? x + w : x,
      leaderToY: y + h / 2,
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
      : track.modulation_hypothesis || "Unknown signal";
  const candidateNumber = track.track_id
    ? Number(track.track_id.split(":").at(-1)) || index + 1
    : index + 1;
  return `Candidate ${candidateNumber} · ${identity}`;
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
