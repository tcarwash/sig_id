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
  const acquisition = result.live_acquisition;
  if (acquisition && typeof acquisition === "object") {
    const originalStart = acquisition.original_sample_start;
    const originalCount = acquisition.original_sample_count;
    const chosenStart = acquisition.selected_sample_start;
    const chosenCount = acquisition.selected_sample_count;
    if ([originalStart, originalCount, chosenStart, chosenCount].every(Number.isInteger)
        && originalStart >= 0 && originalCount > 0 && chosenStart >= originalStart
        && chosenCount > 0 && chosenStart + chosenCount <= originalStart + originalCount) {
      lines.push(`Live audio window: samples ${originalStart}–${originalStart + originalCount}`);
      lines.push(`Selected decoder window: samples ${chosenStart}–${chosenStart + chosenCount}`);
    }
    const carrierProposal = acquisition.carrier_refinement;
    if (carrierProposal && carrierProposal.strategy === "bpsk31-squared-analytic-carrier-v1"
        && Number.isFinite(carrierProposal.initial_carrier_hz)
        && Number.isFinite(carrierProposal.proposed_carrier_hz)) {
      lines.push(`BPSK31 suggested carrier refinement: ${carrierProposal.initial_carrier_hz.toFixed(2)} → ${carrierProposal.proposed_carrier_hz.toFixed(2)} Hz · signal hypothesis only`);
      lines.push("Carrier refinement is spectral evidence; the decoder must still establish varicode coherence before text is published.");
    }
    const attempts = Array.isArray(acquisition.attempts) ? acquisition.attempts : [];
    if (attempts.length) {
      lines.push(`Acquisition trials (${attempts.length}): ${attempts.slice(0, 3).map((trial) =>
        `${shortText(trial?.kind, 25)}${trial?.selected ? " [selected]" : ""} · ${shortText(trial?.status, 20)} · ${shortText(trial?.grade, 30)}`
      ).join(" | ")}`);
    }
    lines.push("Acquisition windows are proposals, not verified message boundaries.");
  }
  const stream = result.cw_stream;
  if (result.decoder_id === "cw" && stream && typeof stream === "object") {
    const delta = shortText(stream.delta_text, 100);
    const accumulated = shortText(stream.cumulative_text, 300);
    const edgeStream = stream.strategy === "continuous-key-transition-cw-v1";
    const label = edgeStream ? "Continuous CW (new key-transition words)"
      : "Incremental CW (new overlap-confirmed words)";
    lines.push(`${label}: ${delta || "none yet"}`);
    lines.push(`CW stream so far: ${accumulated || (edgeStream ? "awaiting complete word timing" : "awaiting corroboration")}`);
    if (Number.isInteger(stream.last_emitted_end_sample) && stream.last_emitted_end_sample >= 0) {
      lines.push(`Last emitted Morse source sample: ${stream.last_emitted_end_sample}`);
    }
    if (edgeStream && Number.isFinite(stream.carrier_hz)) {
      const start = Number.isFinite(stream.initial_carrier_hz)
        ? ` (initial ${stream.initial_carrier_hz.toFixed(1)} Hz)` : "";
      const retunes = Number.isInteger(stream.carrier_retune_count)
        ? ` · ${stream.carrier_retune_count} bounded retunes` : "";
      lines.push(`CW audio carrier: ${stream.carrier_hz.toFixed(1)} Hz${start}${retunes} · oscillator phase continuous, carrier identity unverified`);
    }
    if (edgeStream && stream.state_reset_reason) {
      lines.push(`CW stream initialized/reset: ${shortText(stream.state_reset_reason, 40)} · older partial symbols not reused`);
    }
    lines.push(`Stream status: ${shortText(stream.reason, 60)} · provisional words, no Morse checksum or operator verification`);
  }
  const modem = result.modem_stream;
  if (modem && typeof modem === "object"
      && modem.strategy === "overlap-deduplicated-modem-evidence-v1") {
    const mode = shortText(modem.mode, 24).toUpperCase();
    const label = modem.mode === "ax25" ? "CRC-checked new packet frames" : "new text tokens";
    lines.push(`${mode} incremental ${label}: ${shortText(modem.delta_text, 140) || "none"}`);
    lines.push(`Modem stream so far: ${shortText(modem.cumulative_text, 300) || "awaiting new evidence"}`);
    if (Number.isInteger(modem.new_items) && Number.isInteger(modem.suppressed_overlap_items)) {
      lines.push(`Modem event counts: ${modem.new_items} new · ${modem.suppressed_overlap_items} overlapping duplicates suppressed`);
    }
    if (Number.isInteger(modem.last_emitted_end_sample) && modem.last_emitted_end_sample >= 0) {
      lines.push(`Last emitted modem source sample: ${modem.last_emitted_end_sample}`);
    }
    lines.push("Incremental results preserve sample provenance; decoded text and modem identity remain unverified.");
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
