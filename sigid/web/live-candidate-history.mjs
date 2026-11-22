/** Keep only actionable live CV candidates in the browser's bounded view.
 * Saved recordings and decoder-run evidence remain untouched.
 */
export function pruneLiveCandidates(items, currentSample, rate, options = {}) {
  if (!Number.isFinite(currentSample) || !Number.isFinite(rate) || rate <= 0)
    return items.slice();
  const retainSeconds = options.retainSeconds ?? 75;
  const maxItems = options.maxItems ?? 240;
  const selectedTrackId = options.selectedTrackId || null;
  const selectedProposalId = options.selectedProposalId || null;
  const cutoff = currentSample - retainSeconds * rate;
  const keep = items.filter(item => {
    const selected = Boolean(selectedTrackId && item.track_id === selectedTrackId
      || selectedProposalId && item.cv_proposal_id === selectedProposalId);
    return selected || item.sample_start + item.sample_count >= cutoff;
  });
  if (keep.length <= maxItems) return keep;
  const pinned = keep.filter(item => selectedTrackId && item.track_id === selectedTrackId
    || selectedProposalId && item.cv_proposal_id === selectedProposalId);
  const ranked = keep.filter(item => !pinned.includes(item)).sort((a, b) =>
    (b.sample_start + b.sample_count) - (a.sample_start + a.sample_count));
  return [...pinned.slice(0, maxItems), ...ranked.slice(0, Math.max(0, maxItems - pinned.length))]
    .sort((a, b) => a.sample_start - b.sample_start);
}
