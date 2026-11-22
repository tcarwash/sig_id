// Map a classifier's modulation hypothesis to selectable decoder modes.
// Compatibility is never proof of protocol identity. No automatic decoding.
function normalize(value) {
  return typeof value === "string" ? value.toLocaleLowerCase().replace(/[^a-z0-9]/g, "") : "";
}

export function matchingDecoderSuggestions(method, catalogModes = [], nativeBindings = [], fldigiModes = []) {
  const query = normalize(method);
  if (!query) return {native: [], fldigi: [], compatibleCount: 0};
  const compatible = catalogModes.filter((mode) => [mode.method, mode.display_name, mode.variant]
    .some((value) => normalize(value) === query));
  if (!compatible.length) return {native: [], fldigi: [], compatibleCount: 0};
  const ids = new Set(compatible.map((mode) => normalize(mode.id)).filter(Boolean));
  const names = new Set(compatible.map((mode) => normalize(mode.display_name)).filter(Boolean));
  const native = nativeBindings.filter((mode) =>
    typeof mode.backend_id === "string" && typeof mode.mode_id === "string"
    && (ids.has(normalize(mode.mode_id)) || names.has(normalize(mode.display_name))));
  const fldigiLookup = new Map(fldigiModes.map((name) => [normalize(name), name]));
  const fldigi = [...new Set(compatible.map((mode) => fldigiLookup.get(normalize(mode.display_name)))
    .filter(Boolean))];
  return {native, fldigi, compatibleCount: compatible.length};
}
