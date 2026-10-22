import { matchProtocolStencils } from "./protocol-stencils.mjs";

export function printableAscii(byte) {
  return byte >= 32 && byte <= 126 ? String.fromCharCode(byte) : ".";
}

export function bytesForBits(bits) {
  if (typeof bits !== "string" || !/^[01]*$/.test(bits)) return [];
  const rows = [];
  for (let start = 0; start < bits.length; start += 8) {
    const binary = bits.slice(start, start + 8);
    const value = Number.parseInt(binary, 2);
    rows.push({ bit_start: start, bit_count: binary.length, binary, hex: value.toString(16).padStart(2, "0").toUpperCase(), ascii: binary.length === 8 ? printableAscii(value) : "·" });
  }
  return rows;
}

export function framingCandidates(bits) {
  return matchProtocolStencils(bits).map((match) => ({
    bit_start: match.bit_start,
    bit_count: match.bit_count,
    label: `${match.name} · ${match.field}`,
    evidence: `${match.note} Possible protocols: ${match.protocols.join(", ") || "unassigned"}.`,
    stencil_id: match.stencil_id,
  }));
}

export function createBitAnnotation({ bit_start, bit_count, label, color = "#f4bf75", hypothesis = "bits" }) {
  if (!Number.isInteger(bit_start) || bit_start < 0 || !Number.isInteger(bit_count) || bit_count < 1) throw new TypeError("A positive bit range is required");
  if (typeof label !== "string" || !label.trim()) throw new TypeError("An operator interpretation is required");
  return { bit_start, bit_count, label: label.trim(), color, hypothesis, source: "operator" };
}
