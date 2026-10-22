import { matchProtocolStencils } from "./protocol-stencils.mjs";

export function printableAscii(byte) {
  return byte >= 32 && byte <= 126 ? String.fromCharCode(byte) : ".";
}

export function bytesForBits(bits, alignment = 0) {
  if (typeof bits !== "string" || !/^[01]*$/.test(bits)) return [];
  if (!Number.isInteger(alignment) || alignment < 0 || alignment > 7) throw new RangeError("byte alignment must be an integer from 0 to 7");
  const rows = [];
  for (let start = alignment; start < bits.length; start += 8) {
    const binary = bits.slice(start, start + 8);
    if (binary.length !== 8) {
      rows.push({ bit_start: start, bit_count: binary.length, binary, hex: "·", ascii: "·", lsb_hex: "·", lsb_ascii: "·" });
      continue;
    }
    const value = Number.parseInt(binary, 2);
    const lsbValue = Number.parseInt([...binary].reverse().join(""), 2);
    rows.push({
      bit_start: start,
      bit_count: binary.length,
      binary,
      hex: value.toString(16).padStart(2, "0").toUpperCase(),
      ascii: printableAscii(value),
      lsb_hex: lsbValue.toString(16).padStart(2, "0").toUpperCase(),
      lsb_ascii: printableAscii(lsbValue),
    });
  }
  return rows;
}

export function bitSelectionRange(first, second) {
  for (const cell of [first, second]) {
    if (!cell || !Number.isInteger(cell.bit_start) || cell.bit_start < 0
        || !Number.isInteger(cell.bit_count) || cell.bit_count < 1) {
      throw new TypeError("bit selections require non-empty bit ranges");
    }
  }
  return {
    start: Math.min(first.bit_start, second.bit_start),
    end: Math.max(first.bit_start + first.bit_count, second.bit_start + second.bit_count),
  };
}

export function transformBitstream(bits, transform = "raw", initialState = 0) {
  if (typeof bits !== "string" || !/^[01]*$/.test(bits)) throw new TypeError("bits must be a binary string");
  const supported = new Set(["raw", "inverted", "differential-xor", "differential-xnor"]);
  if (!supported.has(transform)) throw new RangeError("unsupported bitstream transform");
  if (initialState !== 0 && initialState !== 1) throw new RangeError("initial state must be 0 or 1");
  if (transform === "raw") return bits;
  if (transform === "inverted") return [...bits].map((bit) => bit === "0" ? "1" : "0").join("");

  let previous = initialState;
  let output = "";
  for (const bit of bits) {
    const current = Number(bit);
    const decoded = current ^ previous;
    output += String(transform === "differential-xnor" ? decoded ^ 1 : decoded);
    previous = current;
  }
  return output;
}

function shannonEntropy(counts, total) {
  if (!total) return null;
  let entropy = 0;
  for (const count of counts.values()) {
    if (!count) continue;
    const probability = count / total;
    entropy -= probability * Math.log2(probability);
  }
  return entropy;
}

export function bitstreamMetrics(bits, alignment = 0) {
  if (typeof bits !== "string" || !/^[01]*$/.test(bits)) throw new TypeError("bits must be a binary string");
  if (!Number.isInteger(alignment) || alignment < 0 || alignment > 7) throw new RangeError("byte alignment must be an integer from 0 to 7");

  let ones = 0;
  let transitions = 0;
  let longestRun = 0;
  let currentRun = 0;
  let previous = null;
  for (const bit of bits) {
    if (bit === "1") ones += 1;
    if (bit === previous) currentRun += 1;
    else {
      if (previous !== null) transitions += 1;
      currentRun = 1;
      previous = bit;
    }
    longestRun = Math.max(longestRun, currentRun);
  }

  const byteRows = bytesForBits(bits, alignment).filter((row) => row.bit_count === 8);
  const byteCounts = new Map();
  let printableBytes = 0;
  let lsbPrintableBytes = 0;
  for (const row of byteRows) {
    byteCounts.set(row.hex, (byteCounts.get(row.hex) || 0) + 1);
    const value = Number.parseInt(row.hex, 16);
    if (value >= 32 && value <= 126) printableBytes += 1;
    const lsbValue = Number.parseInt(row.lsb_hex, 16);
    if (lsbValue >= 32 && lsbValue <= 126) lsbPrintableBytes += 1;
  }
  const oneFraction = bits.length ? ones / bits.length : null;
  const bitEntropy = oneFraction === null ? null : shannonEntropy(new Map([[0, bits.length - ones], [1, ones]]), bits.length);
  return {
    bit_count: bits.length,
    one_fraction: oneFraction,
    bit_entropy_bits: bitEntropy,
    transition_fraction: bits.length > 1 ? transitions / (bits.length - 1) : null,
    longest_run: longestRun,
    byte_alignment: alignment,
    byte_count: byteRows.length,
    byte_entropy_bits: shannonEntropy(byteCounts, byteRows.length),
    printable_byte_fraction: byteRows.length ? printableBytes / byteRows.length : null,
    lsb_printable_byte_fraction: byteRows.length ? lsbPrintableBytes / byteRows.length : null,
  };
}

export function framingCandidates(bits, customStencils = []) {
  return matchProtocolStencils(bits, undefined, { maxMatches: 256 }).map((match) => ({
    bit_start: match.bit_start,
    bit_count: match.bit_count,
    label: `${match.name} · ${match.field}`,
    evidence: `${match.note} Possible protocols: ${match.protocols.join(", ") || "unassigned"}.`,
    stencil_id: match.stencil_id,
  })).concat(matchProtocolStencils(bits, customStencils).map((match) => ({
    bit_start: match.bit_start,
    bit_count: match.bit_count,
    label: `${match.name} · ${match.field}`,
    evidence: `${match.note} Possible protocols: ${match.protocols.join(", ") || "unassigned"}.`,
    stencil_id: match.stencil_id,
    source: "operator",
  })));
}

export function createBitAnnotation({ bit_start, bit_count, label, color = "#f4bf75", hypothesis = "bits" }) {
  if (!Number.isInteger(bit_start) || bit_start < 0 || !Number.isInteger(bit_count) || bit_count < 1) throw new TypeError("A positive bit range is required");
  if (typeof label !== "string" || !label.trim()) throw new TypeError("An operator interpretation is required");
  return { bit_start, bit_count, label: label.trim(), color, hypothesis, source: "operator" };
}
