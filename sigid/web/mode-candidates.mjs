// Reviewed initial shortlist drawn from Fldigi modes with end-to-end
// loopback evidence. This intentionally does not imply full catalog coverage.
export const REVIEWED_MODE_CANDIDATES = Object.freeze([
  { mode: "BPSK31", modulation: "BPSK", evidence: "Fldigi loopback validated" },
  { mode: "BPSK63", modulation: "BPSK", evidence: "Fldigi loopback validated" },
  { mode: "BPSK63F", modulation: "BPSK", evidence: "Fldigi loopback validated" },
  { mode: "BPSK125", modulation: "BPSK", evidence: "Fldigi loopback validated" },
  { mode: "BPSK250", modulation: "BPSK", evidence: "Fldigi loopback validated" },
  { mode: "BPSK500", modulation: "BPSK", evidence: "Fldigi loopback validated" },
  { mode: "BPSK1000", modulation: "BPSK", evidence: "Fldigi loopback validated" },
  { mode: "QPSK31", modulation: "QPSK", evidence: "Fldigi loopback validated" },
  { mode: "QPSK63", modulation: "QPSK", evidence: "Fldigi loopback validated" },
  { mode: "QPSK125", modulation: "QPSK", evidence: "Fldigi loopback validated" },
  { mode: "QPSK250", modulation: "QPSK", evidence: "Fldigi loopback validated" },
  { mode: "QPSK500", modulation: "QPSK", evidence: "Fldigi loopback validated" },
  { mode: "8PSK125", modulation: "8PSK", evidence: "Fldigi loopback validated" },
  { mode: "8PSK250", modulation: "8PSK", evidence: "Fldigi loopback validated" },
  { mode: "8PSK500", modulation: "8PSK", evidence: "Fldigi loopback validated" },
  { mode: "RTTY", modulation: "2-FSK", evidence: "Fldigi loopback validated" },
  { mode: "THOR8", modulation: "MFSK", evidence: "Fldigi loopback validated" },
  { mode: "THOR16", modulation: "MFSK", evidence: "Fldigi loopback validated" },
  { mode: "MFSK16", modulation: "MFSK", evidence: "Fldigi loopback validated" },
  { mode: "MFSK32", modulation: "MFSK", evidence: "Fldigi loopback validated" },
  { mode: "OLIVIA-8/250", modulation: "MFSK", evidence: "Fldigi loopback validated" },
  { mode: "MT63-500S", modulation: "OFDM", evidence: "Fldigi loopback validated" },
  { mode: "MT63-1KS", modulation: "OFDM", evidence: "Fldigi loopback validated" },
]);

export function narrowKnownModes({ modulation, stencils = [], modes = REVIEWED_MODE_CANDIDATES } = {}) {
  const matches = modes.filter((item) => !modulation || item.modulation === modulation);
  return matches.map((item) => ({
    ...item,
    reasons: [modulation ? `Matches observed ${modulation} modulation family` : "No modulation family filter applied"],
    related_stencils: stencils.filter((stencil) => item.protocolFamilies?.some((family) => stencil.protocols?.includes(family))).map((stencil) => stencil.stencil_id),
  }));
}
