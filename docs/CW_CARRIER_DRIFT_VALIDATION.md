# CW carrier drift and false-output checks (experimental)

**Status:** synthetic receive-only bench and API integration tests passed.
No independent over-the-air receiver/SDR comparison has been performed.
The decoder's words are always tentative, and carrier tracking is not a
verified mode or station identity.

## Why the earlier implementation lost state

The one-process live CW worker keyed demodulator state by session/track, but
discarded it whenever the current audio carrier estimate changed more than
8 Hz from its originally assigned center. That also reprocessed the entire
overlapping 8-second ring window. In a slow frequency drift, the mode detector
can keep the same track ID while the decoder repeatedly resets and emits
the same word again with changed source provenance.

## Changes

- A phase-continuous numerically controlled oscillator (NCO) now stores its
  phase **at the next absolute sample**. Every feed mixes only unprocessed
  source audio. A small carrier correction changes the NCO's increment but
  never reinitializes its filter, envelope, Morse timing or incomplete word.
- Accepted per-call carrier corrections are at most 18 Hz. Larger jumps
  instantiate a **new stream** so unrelated adjacent audio tones are not
  accidentally joined. The stream remains bounded to 12 tracks in the
  worker. This is a defensive guard, not a claim that 18 Hz is a universal
  oscillator-drift threshold.
- `cw_stream` evidence includes `initial_carrier_hz`, `carrier_hz`,
  `carrier_retune_count`, `last_carrier_step_hz`,
  `mixer_phase_continuous`, and `state_reset_reason`. Worker resets are
  tagged `new-stream`, `carrier-jump`, `sample-rate-changed`, or
  `source-discontinuity` rather than being silently hidden.

## Bench methodology

`tools/bench_cw_carrier_tracking.py` renders 30-second, 12-kHz sparse
Signal Lab CW fixtures containing repeated “CQ TEST,” then analytically
mixes the *entire mono signal including the AWGN* with a linear frequency
offset of up to 0, 45 or 90 Hz over the first 20 seconds. This preserves
a wideband approximately white noise control but is **not** a hardware
radio drift measurement.

Each worker receives rolling 8-second windows at 8, 11, 14, 17, 20, 23,
26 and 29 seconds. “Fixed” always requests the nominal center.
“Following” requests the known instantaneous test center at each poll;
it deliberately uses **oracle carrier values**. The third, **detected**
strategy instead computes a new 4,096-bin waterfall at each poll, calls the
project's actual heuristic `detect_tracks` function, and selects the strongest
single-tone, narrow, >=14-dB-SNR peak within 85 Hz of the last detected
carrier. If there is no peak, it holds the preceding center. This choice
uses only received-audio evidence and the prior estimate; it does not
use the oracle truth. Neither method proves a verified mode identity.

To rerun:

```bash
PYTHONPATH=. python -m tools.bench_cw_carrier_tracking \
  --rate 12000 --seeds 6 --seed-start 71062 --snrs 0,-5 --drifts 0,45,90 \
  --output /tmp/sigid-cw-carrier-tracking.json
PYTHONPATH=. python -m tools.bench_cw_false_positives \
  --seeds 8 --seed-start 400 --snrs=-5,0 \
  --output /tmp/sigid-cw-false-controls.json
```

### Before/after: same 4 seeds, +45 Hz drift, following mode

| Synthetic condition | Baseline | With phase-continuous retune |
| --- | ---: | ---: |
| 0 dB, exact phrase present | 4/4 | 4/4 |
| 0 dB, words output | 37 | 32 |
| 0 dB, incorrect words | 0 | 0 |
| 0 dB, source-seconds processed (4 cases) | 156 | 116 |
| −5 dB, exact phrase present | 4/4 | 4/4 |
| −5 dB, words output | 31 | 30 |
| −5 dB, incorrect words | 4 | 4 |
| −5 dB, source-seconds processed (4 cases) | 156 | 116 |

Expected four 29-second unique-source streams = 116 seconds.
The baseline spent 40 extra source-seconds re-decoding overlap and
emitted redundant words in the strong-SNR drift case. The updated
worker preserves one-pass provenance.

### More demanding +90 Hz synthetic drift, 6 seeds

| SNR | Carrier strategy | Exact “CQ TEST” present | Total words | Incorrect words |
| --- | --- | ---: | ---: | ---: |
| 0 dB | fixed at original | 6/6 | 32 | 2 |
| 0 dB | following test center | 6/6 | 48 | 0 |
| −5 dB | fixed at original | 6/6 | 30 | 2 |
| −5 dB | following test center | 6/6 | 45 | 5 |

### Carrier obtained from the actual heuristic detector, 6 seeds

| SNR | Synthetic drift | Exact “CQ TEST” present | Total emitted words | Incorrect words |
| --- | --- | ---: | ---: | ---: |
| 0 dB | 0 Hz | 6/6 | 48 | 0 |
| 0 dB | +45 Hz | 6/6 | 48 | 0 |
| 0 dB | +90 Hz | 6/6 | 48 | 0 |
| −5 dB | 0 Hz | 6/6 | 46 | 4 |
| −5 dB | +45 Hz | 6/6 | 48 | 1 |
| −5 dB | +90 Hz | 6/6 | 48 | 1 |

For these synthetic fixtures, the detector produced narrow, single-tone
peaks whose centers lagged the instantaneous true carrier by roughly 15–20 Hz
in the +90-Hz condition. The decoder followed the measured centers without
losing its incomplete Morse words. These observations are **spectral
algorithm plus decoder tests**, not independent live SDR/RF verification.
The tested drift is positive-going; opposite direction still requires tests.

### Actual 48-kHz capture-rate spot check

The benchmark also ran at the preview's normal **48-kHz** sample rate,
using two independent seeds (71062–71063), +45 and +90 Hz drift, and
0/−5 dB SNR. Carrier estimates again came from the actual heuristic
spectral detector, not oracle truth. All **8/8** cases contained the
expected `CQ TEST` phrase; the detector-following strategy emitted one
incorrect word total (in a −5-dB, +45-Hz case), with zero forced stream
resets. Each two-seed detector test had 12 positive polls in the expected
active burst region; later silent polls correctly had no tone candidate.
This is a small sample-size smoke test, not an accuracy estimate.

Run with `--rate 48000 --seeds 2 --snrs 0,-5 --drifts 45,90`.

The improvement in 0-dB recovered words is real **for these generated
seeds**; at −5 dB, following also increases incorrect words. Do not
interpret exact-phrase-present as accurate complete transcription or
independent frequency acquisition. Note that these 90-Hz cases were run
with the updated decoder; no 90-Hz pre-change comparison was recorded.

## False-word negative controls

`tools/bench_cw_false_positives.py` ran 64 synthetic 30-second cases,
each in a fresh stream with the same overlapping poll schedule:

| Negative control | Cases | Cases emitting CW words |
| --- | ---: | ---: |
| Gaussian white noise | 8 | 0 |
| Random-duration, non-Morse on/off audio carrier | 8 | 0 |
| RTTY at −5 dB and 0 dB | 16 | 0 |
| BPSK31 at −5 dB and 0 dB | 16 | 0 |
| AX.25 at −5 dB and 0 dB | 16 | 0 |
| **Total** | **64** | **0** |

Zero outputs in a small selected synthetic control set does **not**
establish a field false-positive rate or upper bound for adversarial
QRM. Other weaknesses include sporadic incorrect weak-CW letters,
start-of-burst truncation, sideband offsets and unrelated keyed tones
that happen to match valid Morse timing.

## Pipeline-level verification

A FastAPI/ProcessPool CV integration test uses an abstaining classifier
with a clear ranked CW proposal, two overlapping PCM windows and a
modelled CV center update of +13.5 Hz. It asserts that the second
decoder call advances by only 3 seconds of new audio, preserves stream
identity, records one retune and leaves `confirmed_identity=false`.
A separate test checks jumps greater than 18 Hz force a worker reset.
No RF was transmitted or flashed.

## Follow-on experiments

- Estimate the carrier *from the actual received audio* (robust peak
  tracking on CW key-down intervals), rather than using oracle center.
- Measure opposite-sense drift, step interference, adjacent CW signals,
  edge-cropped callsigns and varying WPM/frequency error.
- Validate with independently captured SDR/audio recordings whose
  transmitted content and center frequency are known, and report both
  miss/false word rates and latency.
