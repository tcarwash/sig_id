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
  if (typeof bits !== "string") return [];
  const candidates = [];
  // HDLC uses the 0x7E flag; bit order and framing are not inferred here.
  for (let at = bits.indexOf("01111110"); at >= 0; at = bits.indexOf("01111110", at + 1)) {
    candidates.push({ bit_start: at, bit_count: 8, label: "HDLC-like flag (0x7E)", evidence: "Exact 8-bit pattern; bit order and protocol are unverified." });
  }
  return candidates;
}

export function createBitAnnotation({ bit_start, bit_count, label, color = "#f4bf75", hypothesis = "bits" }) {
  if (!Number.isInteger(bit_start) || bit_start < 0 || !Number.isInteger(bit_count) || bit_count < 1) throw new TypeError("A positive bit range is required");
  if (typeof label !== "string" || !label.trim()) throw new TypeError("An operator interpretation is required");
  return { bit_start, bit_count, label: label.trim(), color, hypothesis, source: "operator" };
}
