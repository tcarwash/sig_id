# RTTY — evidence dossier (experimental only, not on-air validated)

- Owned mode ID: rtty
- Implementation/version: sigid/libsigdecode/rtty.py, `rtty-ita2-fsk-v2`
- State: experimental implementation registered in
  `sigid/libsigdecode/__init__.py`; synthetic vectors pass, but it has never
  been independently validated on air.
- Primary published protocol specification and exact revision (reviewed
  directly during integration):
  [ITU-T S.1, International Telegraph Alphabet No. 2 (1993-03)](https://www.itu.int/rec/T-REC-S.1-199303-I/)
  — Table 1 and clauses 3.1–3.2, 4.1–4.5, and Table 2. S.1 standardises
  the alphabet, not a complete on-air receiver
  (no baud/shift tolerance, sync, or interference behaviour).
- Framing/profile supplement: [ARRL, Getting Started in Digital
  Communications — Part 3: RTTY](https://www.arrl.org/files/file/Technology/tis/info/pdf/59241.pdf)
  lists 45, 50, 57, and 75 baud for Baudot, 170 Hz as the standard shift,
  and Baudot stop lengths of 1.4, 1.5, or 2 bits. The implementation adds
  57-baud and 1.4-stop support. The same source describes 110 baud as HF
  ASCII, which remains outside this five-bit ITA2 decoder.
- Protocol fields not yet specified: carrier/timing tolerance and
  competing-signal behavior remain heuristic; the published alphabet and
  profile references do not specify a complete receiver.
- Source/provenance and any third-party license audit: independently authored
  from the published alphabet plus standard amateur RTTY framing practice.
  No code, tables, or parameters copied/translated from OnAir, Fldigi,
  WSJT-X, or any other decoder application. NumPy/SciPy are DSP dependencies
  only, not decoding engines.
- Input (audio vs IQ, sample rate, carrier, timing, variant): bounded mono
  audio only (`DecodeWindow`, max 120 s enforced by core). Operator-supplied
  carrier required; mark = carrier + shift/2, space = carrier − shift/2.
  Supported baud {45.45, 50, 57, 75, 100}, shift {170, 200, 425, 850} Hz,
  stop bits {1, 1.4, 1.5, 2} (searched, winner recorded). All other
  baud/shift/stop combinations raise `ValueError` at settings construction.
- Output (text, tokens, packet, image, CRC/FEC status): ITA2 text with
  per-character tokens on the ORIGINAL capture timeline
  (`start_sample`/`end_sample` in source indexes, pattern `L|F:bbbbb` with
  data bits in transmission order). confidence_hint is a framing/tone-fit
  heuristic, never a calibrated probability. ITA2 has no FEC or CRC, so
  `bit_and_frame_integrity` is reported as `ita2_has_no_fec_or_crc`.

## Implemented assumptions (exact)

1. Idle/quiet line state is mark; start bit is space (one unit); 5 data bits
   LSB-first; stop is mark for 1, 1.4, 1.5, or 2 units (classic amateur default
   45.45 baud / 170 Hz / 1.5 stop).
2. Normal polarity: mark tone (higher audio frequency) = 1, space = 0.
   Inverted polarity is tried as an explicit hypothesis; a near-tie between
   polarity/stop hypotheses (both ≥ 0.70 valid-stop fraction, gap < 0.05)
   abstains as ambiguous rather than guessing.
3. Initial shift state is letters; FIGS/LTRS shift mid-stream; FIGS+J (bell)
   is counted in `bell_count`, never printed; NUL/BLANK ignored; CR → `\r`,
   LF → `\n`. Figure assignments for F/G/H are left unassigned as S.1
   clause 4.2 requires; national/private alternatives are not guessed.
4. The ITA2 code table in the module was checked against S.1 Table 1/2.
   The code integer uses the first transmitted unit as the least significant
   bit, as required by the standard's note that code element 1 is transmitted
   first. The figures for F/G/H (combinations 6/7/8) are not internationally
   defined by clause 4.2 and are therefore left unassigned.
5. A window must show BOTH tones with data-like alternation (decisive
   fraction ≥ 0.30, each tone dominant ≥ 0.08 of the window) or the decoder
   abstains — silence, white noise, a single keyed tone, and idle mark-only
   diddle all abstain. `decoded` additionally requires valid-stop fraction
   ≥ 0.85 and ≥ 6 emitted characters; `partial` requires ≥ 0.60 and ≥ 3
   characters; otherwise `no_decode` with empty text.
6. Receive-only: pure array computation, no hardware, tuning, transmit,
   subprocess, network, or file I/O. 120-second core cap plus an 8,000,000
   input-sample cap bound DSP memory and work.
   Token indexes are `sample_start + decimated_index × decimation_factor`
   (within a few ms when decimated); exact at factor 1.

## Unsupported variants (explicit abstain or settings error)

- Baud rates other than 45.45/50/57/75/100 (including 110-baud ASCII and 300-baud HF variants):
  `ValueError`. Shifts other than 170/200/425/850 Hz: `ValueError`. Stop
  bits outside {1, 1.4, 1.5, 2}: `ValueError`.
- Missing carrier (`carrier_hz=None`): `no_decode` — no automatic
  multi-carrier search. Tones outside the audio passband or unresolvable at
  the window rate: `no_decode`.
- Multiple/co-channel RTTY carriers in one window, bit-sync drift within a
  window beyond the single static grid search, and signals shorter than two
  8-bit characters: `no_decode`.
- Non-Baudot alphabets (7-bit ASCII/AMTOR/Pactor/ Olivia etc.) are out of
  scope for this slice.

## Independently generated golden vectors

`tests/test_decode_rtty.py` synthesizes continuous-phase 2-FSK from a
separately written ITA2 encoder and checks text at 45.45 baud / 170 Hz,
50 baud / 425 Hz, 57 baud / 170 Hz, 75 baud / 200 Hz, and 100 baud / 850 Hz.
The tests cover every combination of those five baud rates and four shifts,
normal and inverted polarity, 1/1.4/1.5/2 stop bits, letters-shift J, and
48 kHz input decimation with original-sample token offsets. These are deterministic
generated inputs, not stored audio captures; no waveform SHA-256 is recorded.

## Negative and boundary controls

Executed synthetic abstention controls cover silence, seeded noise, a single
tone, wrong carrier, and a fixed memory limit. The contrast gate
(mark/space dominance), framing-validity gate (start-space/stop-mark/data
decisiveness per character), and polarity/stop ambiguity rule remain
heuristic; separate corrupted-frame and competing-station tests are still
needed.

## Reviewed real recordings

None. No on-air capture has been reviewed; no transcript ground truth, no
CER/WER, no SNR calibration. Real-world accuracy is entirely unmeasured.

## Reproducibility and limitations

- Validation boundary: code table reviewed against S.1; deterministic
  synthetic round-trip and abstention tests pass. No external black-box
  comparison or on-air evidence. Status stays experimental.
- Integration review corrected erroneous letters/figures assignments found
  in the initial worker draft and prevents a partial trailing frame from
  receiving an artificially clipped stop-bit score. Independent review also
  caught a letters-shift J/BEL confusion; BEL is now counted only in figures
  shift, and a J round-trip covers the correction.
- Known limits: single static timing grid per window (no per-character
  clock tracking or drift compensation); single carrier; national/private
  alternatives for figures F/G/H are deliberately unsupported; BEL/NUL
  produce no tokens; missed FIGS/LTRS
  mid-text can mis-case subsequent characters; the 8M sample cap may reject
  long windows at high sample rates.
- Unresolved shared dependency: work queue lists `depends_on:
  ["shared_fsk_timing"]`. This slice implements a MODE-PRIVATE timing/grid
  search inside `rtty.py` and proposes — but does not create — a shared
  `shared_fsk_timing` primitive (common noncoherent FSK envelope/bit-timing
  helper with generic test vectors) owned by the integrator. No shared
  file was modified. If the integrator later provides it, this decoder
  should be refactored onto it and re-gated.
- Reproduction commands: `python -m pytest -q tests/test_decode_rtty.py
  tests/test_decoder_conformance_rtty.py`; the pinned default and 57-baud
  suites are synthetic only.
- Owner/reviewer and next acceptance gate: coordinator-owned. Gate: (1)
  shared-FSK decision from integrator, (2) independently hand-transcribed on-air captures with provenance, and
  (3) a separate input/result contract for 110-baud ASCII before adding that
  alphabet. No benchmarked/validated claim is supported yet.
