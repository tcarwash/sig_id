// Pure, receive-only presentation of persisted first-party decoder evidence.
// Keep this independent of particular modem implementations and browser state.
const resultStates = {
  decoded: "Decoded candidate — not operator verified",
  partial: "Partial candidate — not operator verified",
  no_decode: "Abstained — no reliable decode",
};

function shortText(value, limit = 140) {
  return typeof value === "string" ? value.slice(0, limit) : "";
}

export function summarizeDecoderResult(result) {
  if (!result || typeof result !== "object"
      || result.source !== "sigid-independent-implementation") return "";
  const status = resultStates[result.status] || "Unknown decoder status";
  const lines = [`First-party ${shortText(result.decoder_id, 48) || "decoder"} · ${status}`];
  if (result.version) lines.push(`Implementation: ${shortText(result.version, 80)}`);
  const evidence = result.evidence && typeof result.evidence === "object" ? result.evidence : {};
  const descriptors = [
    ["carrier_hz", "Audio carrier", "Hz"],
    ["estimated_wpm", "Estimated speed", "WPM"],
    ["mark_count", "Keyed marks", ""],
    ["mean_symbol_timing_fit", "Timing-fit heuristic", ""],
    ["unknown_token_fraction", "Unknown-token fraction", ""],
    ["envelope_on_off_ratio", "Envelope on/off ratio", ""],
  ];
  for (const [key, label, units] of descriptors) {
    if (typeof evidence[key] === "number" && Number.isFinite(evidence[key])) {
      lines.push(`${label}: ${evidence[key]}${units ? ` ${units}` : ""}`);
    }
  }
  const tokens = Array.isArray(result.tokens) ? result.tokens : [];
  if (tokens.length) {
    const preview = tokens.slice(0, 24).map((token) => {
      const pattern = shortText(token?.pattern, 16);
      const symbol = shortText(token?.text, 8);
      return pattern ? `${symbol}(${pattern})` : symbol;
    }).filter(Boolean).join(" ");
    lines.push(`Candidate tokens (${tokens.length}): ${preview}${tokens.length > 24 ? " …" : ""}`);
  }
  // Candidate frames and soft decisions remain distinct from validated mode identity.
  const frames = Array.isArray(result.frames) ? result.frames : [];
  if (frames.length) {
    lines.push(`Candidate binary frames: ${frames.length}`);
    for (const [index, frame] of frames.slice(0, 8).entries()) {
      const bits = Number.isInteger(frame?.bit_length) ? frame.bit_length : "?";
      const bytes = Number.isInteger(frame?.byte_count) ? frame.byte_count : "?";
      const checks = Array.isArray(frame?.integrity) ? frame.integrity : [];
      const checkSummary = checks.slice(0, 4).map((check) =>
        `${shortText(check?.kind, 16)} ${shortText(check?.algorithm, 48)}: ${shortText(check?.outcome, 16)}`
      ).join(" · ");
      lines.push(`Frame ${index + 1}: ${bytes} bytes, ${bits} bits · samples ${frame?.start_sample ?? "?"}–${frame?.end_sample ?? "?"}${checkSummary ? ` · ${checkSummary}` : " · integrity not checked"}`);
    }
    if (frames.length > 8) lines.push(`… ${frames.length - 8} more frame candidates`);
  }
  const spans = Array.isArray(result.soft_bits) ? result.soft_bits : [];
  if (spans.length) {
    const bitCount = spans.reduce((n, span) => n + (Number.isInteger(span?.bit_count) ? span.bit_count : 0), 0);
    const types = [...new Set(spans.map((span) => shortText(span?.encoding, 32)).filter(Boolean))];
    lines.push(`Soft decision evidence: ${bitCount} bits in ${spans.length} span(s) · ${types.join(", ")} (uncalibrated)`);
  }
  if (result.status === "no_decode" && evidence.unverified_pattern_preview) {
    lines.push(`Unverified interpretation (NOT a decode): ${shortText(evidence.unverified_pattern_preview)}`);
  }
  if (result.note) lines.push(`Decoder note: ${shortText(result.note, 300)}`);
  lines.push("Evidence is diagnostic, not a calibrated probability or confirmed identification.");
  return lines.join("\n");
}
