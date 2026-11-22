# libsigdecode: independently implemented in-project decoders

**Goal:** decode ham-radio signals from Signal ID's own audio or IQ, without
installing OnAir, WSJT-X, Fldigi or another SDR receiver. Preserve external
decoders as optional backend adapters for cross-checks and unsupported modes.

## Source and licensing policy

Write independent implementations from public protocol specifications and
independently created test vectors. Do not copy, mechanically translate or
import OnAir/Fldigi/WSJT-X decoder source into this library. A black-box
decoder can be used for comparison, but its output is not guaranteed truth.
Record per-mode specifications and test-vector provenance. Numerics libraries
such as NumPy/SciPy are DSP dependencies, not external decoding engines.
Determine Signal ID's distribution license explicitly before release.
Copied copyrighted expression requires its own licensing compliance review.

## Implemented now

- sigid/libsigdecode/core.py: bounded mono DecodeWindow, explicit mode
  registry, structured DecodedToken sample offsets, LocalDecodeResult.
- sigid/libsigdecode/cw.py: original experimental Morse decoder. Given an
  audio carrier, it isolates an envelope, extracts marks/spaces, estimates WPM,
  and interprets the international Morse alphabet. It returns text,
  character timings and heuristic timing-fit measures. The fit score is
  NOT a calibrated probability. This is not a multi-station skimmer.
- sigid/local_decoder_backend.py: thin adapter, backend name 'sigdecode',
  selected WAV decoding and persisted decode-run records through the existing
  Signal ID API. No external decoder application or radio control is involved.
- sigid/libsigdecode/cli.py: reproducible offline JSON decoding.

From the UI: select a saved audio signal, open Decode, choose
'Signal ID built-in decoders · CW (Morse)', then decode the selected window.
The action reads 'Decode with Signal ID' for the in-project backend;
external decoder selection remains available from the same menu.

CLI:

    python -m sigid.libsigdecode.cli recording.wav --mode cw --carrier-hz 1100 --start-seconds 0 --duration-seconds 20

Input is mono PCM16 WAV, maximum 120 seconds. The caller chooses the CW
carrier, rather than assuming a CV/model guess is already correct. Actual
accuracy on hand-transcribed on-air recordings remains to be measured.

## Mode roadmap (not a claim that these are implemented)

| Priority | Family | Initial decoders |
|---|---|---|
| First | Morse/beacons | CW, QRSS, DFCW |
| High | Traditional | RTTY/Baudot, BPSK31, QPSK31, Hellschreiber |
| High | Packet | Bell 202, AX.25/APRS, HF packet |
| High | Weak signal | FT8, FT4, WSPR, JT65, JT9, JS8 |
| Medium | Keyboard/FEC | Olivia, Contestia, THOR, DominoEX, MFSK, MT63 |
| Medium | Images | SSTV, digital SSTV |
| Later | Specialized/voice | PI4, FSK441, MSK144, Q65, M17, FreeDV |

For modes requiring complex IQ, use a separate typed IQ input; SSB audio
cannot recover information lost in prior demodulation.

Each mode must have: a public specification, independent source, deterministic
golden vectors, exact time/frequency provenance, explicitly typed partial
results, false-positive and corruption tests, independent real-world
ground truth, a bounded CPU budget, and optional black-box comparisons with
reference implementations. FEC/CRC validity should be reported explicitly
where available. Decoders must never block audio capture or transmit.

Next slices: improve CW against reviewed IC-7300 Morse recordings; add
RTTY/ITA2 framing atop native 2-FSK symbol analysis; PSK31/varicode;
AX.25/NRZI/HDLC/CRC; and eventually native slot-aligned FT8/LDPC decoding.
External OnAir/WSJT-X/Fldigi adapters continue to work independently.

## Cautious decode semantics

The CW timing-fit score is a diagnostic, not a probability. The first
archived IC-7300 replay produced gibberish on a narrow selected band;
its mean timing fit was only 0.23–0.29. The CW decoder now abstains on
that audio, retaining candidate marks in structured metadata instead of
publishing the text as a successful decode. The true mode and text in
that audio have not been independently established. Real Morse accuracy
and false-decode rates still require reviewed recordings.
