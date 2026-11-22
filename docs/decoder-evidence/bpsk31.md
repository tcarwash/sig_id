# BPSK31 — evidence dossier (experimental only)

- Owned mode ID: bpsk31
- Implementation/version: sigid/libsigdecode/bpsk31.py, bpsk31-text-v1
- Status: experimental_implemented; never independently validated on air
- Primary protocol source: [ARRL PSK31 specification](https://www.arrl.org/psk31-spec)
  (31.25 baud BPSK; phase reversal = 0, no reversal = 1; "00" separates
  self-synchronizing varicode codewords, transmitted left-bit-first;
  BPSK has no FEC/CRC); printable Varicode values checked against the
  [ARRL-published Varicode table](https://www.arrl.org/files/file/Technology/tis/info/pdf/x9907003.pdf)
- In-project API: [decoder contract](../DECODER_API_SPEC.md)
- Deterministic tests: [test_decode_bpsk31.py](../../tests/test_decode_bpsk31.py)

## Coverage and limitations

Single selected audio carrier, mono audio only; no automatic
multi-carrier search, no sideband selection, no streaming/IQ input.
Carrier accuracy must be within +/-3 Hz (squaring estimator bound check).
Symbol timing: 16 steps/symbol grid search, re-estimated per 128-symbol
block with a +/-2-step continuity constraint; short trailing blocks
(<32 symbols) inherit timing. Fast clock drift outside these bounds
abstains. Only printable ASCII varicode (codes 32-126) is embedded;
control characters (including CR/LF) decode as unmapped "?" placeholders
and cap status at partial — a documented coverage gap. The decoder uses a
conservative 0.997 phase-coherence threshold to reject unrelated signals;
this trades sensitivity for fewer false-positive text decodes. Unmapped
codewords fail closed ("?", at most partial).

The printable mapping is checked against the published ARRL Varicode table.
The test suite asserts a fixed table fingerprint plus representative codes,
and its deterministic test waveform uses the same bit ordering and framing
conventions described by the protocol source.

## Independently generated golden vectors

Test-only synthesizer (`synth_bpsk31` in the test file): text encoded via
the decoder's published table, differential phase track, raised-cosine
complex blend at reversal boundaries (amplitude dips on reversals only),
single audio carrier at 8000 Hz / 1000 Hz. Round-trip text
"cq cq de ab1cd 73 " (letters, digits, spaces): transmitted bit-string
SHA-256 `039bb912b804adede67f338b5dba8f3705f6cef82080e2a172c5791fc0259794`
(32768 samples). Controls: polarity inversion (x-1), absolute phase
offset 0.7 rad, and 0.37-symbol leading-silence shift all decode to the
identical text with absolute source sample offsets on every token.

## Negative and boundary controls

Silence, seeded noise, steady carrier, +6 Hz carrier offset (explicit CFO
bound path), +200 Hz wrong carrier, alternating-bit overlong framing with
no "00", mid-stream garbage resync (decodes around it as partial), and a
45-baud two-tone FSK signal must all abstain (no_decode, empty text).
Missing carrier and sub-0.5 s windows abstain. No FEC/CRC exists per
protocol; evidence always reports `none_defined_by_protocol_no_fec_crc`,
and coherence values are documented heuristics, not probabilities.

## Reviewed real recordings

None. No on-air BPSK31 capture, transcript, CER/WER, or false-positive
rate is established. Status stays experimental until independently
hand-transcribed real captures are reviewed.

## Reproducibility and limitations

- Test commands: `python -m pytest -q tests/test_decode_bpsk31.py` and
  `python -m pytest -q tests/test_decoder_contract_docs.py`.
- Mode is registered as experimental; it is not exposed in the current UI
  menu, and no real-capture performance is claimed.
- `decoder-work-items.json` state for bpsk31 is `experimental_implemented`.
- Known failure classes: carrier error beyond +/-3 Hz, fast drift,
  multi-station passbands and control-character text.
- No external-decoder comparison performed; none is ground truth.
