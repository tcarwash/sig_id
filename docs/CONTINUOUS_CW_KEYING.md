# Continuous CW key-transition decoder (experimental)

This document describes the **receive-only** CW streaming lane introduced by
`feature/continuous-cw-keying`. It is intended for live signal-ID *evidence*,
not confirmed Morse transcription, callsign identification, or mode verification.

## Why another CW lane?

The original first-party CW decoder (`sigid/libsigdecode/cw.py`) estimates
on/off keying in a bounded audio crop. The subsequent
`CWTokenStream` cross-checks complete words across overlapping polls, but it
cannot carry partly received symbols through a crop boundary. The new lane
keeps DSP and Morse timing state across polls and reads **each absolute source
sample at most once**, even when submitted audio windows overlap.

## Flow and isolation

1. The live detector proposes a *persistent, narrow* candidate (≤80 Hz).
   The classifier must supply an explicit CW/Morse hypothesis (an accepted
   mode or a ranked top candidate, even if classification abstains). Merely
   seeing a narrow carrier does **not** enable continuous Morse decoding.
   Ranked hypotheses remain unverified and cannot become mode labels.
2. The regular `NativeLiveRunner` admission limits still apply, including at
   most one in-flight decode child, 16 CW jobs/minute maximum and a
   3,000-ms/minute CPU budget.
3. The dedicated **decoder process** holds a bounded `CWKeyingStream` for
   each (`session_id`, `track_id`) key. The audio callback performs no DSP.
   At most 12 streams are retained; oldest streams are evicted.
4. The stream uses a phase-continuous complex oscillator, fourth-order
   45-Hz lowpass, a ~500-Hz complex envelope, a three-sample smoother,
   adaptive on/off threshold, and a 10-ms transition debounce. Filter,
   oscillator phase, envelope remainder, and unfinished Morse marks persist
   across bounded carrier corrections (up to 18 Hz per poll); larger jumps
   reset to protect against unrelated nearby signals. See
   [carrier drift validation](CW_CARRIER_DRIFT_VALIDATION.md).
5. Mark durations provide a provisional WPM/dot estimate (5–80 WPM).
   Complete, bounded words must have clean leading silence, an observed
   terminal word gap, valid alphabet patterns, and conservative timing fit.
   A **minimum of two plausible words totaling at least five letters** is
   required before the *first* emission; subsequent words can emit singly.
6. `cw_stream` evidence preserves the original sample positions, incremental
   delta, cumulative transcript, strategy name, no-duplicate sample count,
   and `confirmed_identity: false`. The UI distinguishes key-transition
   output from older overlap-corroborated words.

The existing `CWTokenStream` remains active for narrow/uncertain candidates
that did **not** qualify for this specific key-transition lane. Other decoder
modes keep their pre-existing multi-mode trials. The standalone/offline CW
decoder remains untouched.

## Verification

Run from a worktree with the existing dev environment:

```bash
PYTHONPATH=. /storage5/dev/sig_id/.venv/bin/python -m pytest -q \
  tests/test_cw_keying_stream.py tests/test_cw_token_stream.py \
  tests/test_native_live_decode.py tests/test_native_weak_acquisition.py
node --test tests/test_decoder_result.mjs
```

The focused tests cover overlapping polls, no reprocessed samples, initial
two-word admission, source discontinuity, per-session isolation, monotonic
sample spans, noise-only abstention, spawned process integration, budget gates,
legacy-path compatibility, and UI evidence.

## Constraints / known limitations

- All positive CW evidence is **synthetic**, using the pinned Signal Lab
  `CQ TEST` fixture at 12 kHz. It is **not** an on-air validation result.
- Narrow filtering assumes a reasonably stable detected audio carrier
  (8-Hz tolerance across polls). Frequency tracking/AFC and simultaneous
  multiple CW stations are not implemented.
- Morse timing adaptation and noise floor tracking are heuristics and may
  abstain or produce false text at very low SNR, during QRM, with unusually
  large speed variation, or with poor detector track continuity.
- A stream begun mid-word deliberately discards truncated leading words.
- The first two-word gate suppresses short noise coincidences but can delay
  isolated one-word transmissions. There is no checksum in Morse, so even
  coherent text is **unverified**.
- The UI and API may display a running transcript, but should never turn it
  into a confident signal-ID label without independent corroboration.

## Next experiments

1. Run a wider pinned multi-seed SNR/QRM/speed sweep with false-word
   counts, clean-word exact match, timing, and CPU percentiles. Compare
   against the prior overlap-confirmed decoder under identical captures.
2. Exercise the lane from a separate live preview with genuine receive
   audio, deliberate retunes, playback restarts and overlapping tracks.
3. Add an optional carrier tracking loop with drift measurements, not
   aggressive retuning that destroys timing continuity.
4. Investigate soft symbol reliability, WPM drift, Farnsworth spacing,
   and bounded joint detection/decoding assistance under weak-signal QRM.

No transmitter, CAT/PTT, or radio firmware operations are part of this feature.

## Preliminary synthetic sweep (2026-11-22)

Signal Lab `CQ TEST` repeating message, 12 kHz audio, reference SNR
bandwidth 2,500 Hz, sparse traffic, 16-second burst, source seeds 21–26.
Receive-only tests used 8-second windows every 3 seconds and retained
all DSP state in between. This is a **small exploratory sweep**, not an
estimate of field performance.

| SNR | Seeds containing correct `CQ TEST` | Seeds with an incorrect emitted word |
| --- | ---: | ---: |
| 0 dB | 6/6 | 0/6 |
| −5 dB | 6/6 | 1/6 |
| −10 dB | 0/6 | 0/6 |

Eight 30-second AWGN-only controls and RTTY, BPSK31 and AX.25 test
transmissions produced no CW word outputs. The initial two-word admission
gate was added after an earlier −10 dB version had emitted a dubious
single word; the table above is **after** that change.

These counts indicate useful receive sensitivity on this particular
synthetic fixture, not general precision or recall. A valid Morse word
can still be misrecognized, especially at −5 dB. The remaining priority
is a blind live-radio capture benchmark, QRM controls, and CPU percentiles.


A later [provisional native-trial handoff](PROVISIONAL_NATIVE_DECODER_TRIALS.md)
uses detector persistence and uncertain model rankings to start safe
CPU-bounded trials earlier; it does not change the Morse demodulator itself.
