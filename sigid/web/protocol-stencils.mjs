/**
 * A stencil is evidence for a possible field/framing element, never a protocol ID.
 * `pattern` uses 0/1 for required bits and `?` for ignored bits.
 */
export const PROTOCOL_STENCILS = Object.freeze([
  Object.freeze({
    id: "hdlc-flag",
    name: "HDLC-like flag",
    pattern: "01111110",
    field: "frame delimiter",
    protocols: ["HDLC family; also used by protocols such as AX.25"],
    note: "Common delimiter, not unique to one protocol; bit order and framing are unverified.",
  }),
]);

export function matchProtocolStencils(bits, stencils = PROTOCOL_STENCILS, { maxMatches = 256 } = {}) {
  if (typeof bits !== "string" || !/^[01]*$/.test(bits)) return [];
  if (!Number.isInteger(maxMatches) || maxMatches < 1) throw new RangeError("maxMatches must be a positive integer");
  const matches = [];
  for (const stencil of stencils) {
    const pattern = stencil?.pattern;
    if (typeof pattern !== "string" || !/^[01?]+$/.test(pattern)) continue;
    for (let offset = 0; offset + pattern.length <= bits.length; offset += 1) {
      let fits = true;
      for (let index = 0; index < pattern.length; index += 1) {
        if (pattern[index] !== "?" && pattern[index] !== bits[offset + index]) { fits = false; break; }
      }
      if (!fits) continue;
      matches.push({
        stencil_id: String(stencil.id || "unnamed"),
        name: String(stencil.name || "Unnamed stencil"),
        field: String(stencil.field || "pattern"),
        protocols: Array.isArray(stencil.protocols) ? [...stencil.protocols] : [],
        bit_start: offset,
        bit_count: pattern.length,
        note: String(stencil.note || "Pattern match only; interpretation is unverified."),
      });
      if (matches.length >= maxMatches) return matches;
    }
  }
  return matches;
}
