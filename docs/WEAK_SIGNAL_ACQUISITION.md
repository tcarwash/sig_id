# Weak-signal and longer-CW acquisition: synthetic evaluation

This receive-only milestone addresses arbitrary live audio boundaries without changing the audio callback, radio controls, ML labels, or optional external decoders. It is **overlapping-window acquisition**, not yet a stateful CW stream decoder.

## Acquisition changes

**RTTY remains four seconds per mode.** For segmentation, the two 170-Hz-shift mark/space tones (carrier ±85 Hz) must each exceed an off-tone Welch PSD median by at least 6 dB and must be within 12 dB of each other. This gate prevents a single CW-like keyed carrier from triggering RTTY segmentation. A strong 5:1 full-band envelope contrast uses the existing burst finder. If that fails, a narrow two-tone envelope with 30-ms hops, 3.3:1 contrast, and at least 120 ms of silence between bursts may suggest a complete interior message. At most two bounded segments are tried, then the original complete four-second crop if needed. None of these tests establish a verified FSK protocol identity; the underlying RTTY modem still assumes 45.45 baud and 170-Hz shift.

**CW-first candidates may use eight seconds of the existing ten-second live ring.** Both the full eight seconds and the latest four seconds are evaluated by the same single isolated decoder worker. The selection compares coherent status, Morse timing fit, and unknown-token fraction rather than assuming longer is better. A CW second-choice fallback does not expand capture length. Other decoders are still limited to four seconds even when they run alongside CW.

Native results retain original and chosen absolute sample start/count, attempted sample spans, trial status, the selected trial marker, and an explicit unverified identity flag. They remain independently comparable with manually selected Fldigi/OnAir/WSJT-X result cards. **No decoded text automatically becomes a confirmed identity or annotation.**

Existing worker admission limits remain: one process, zero queued jobs, five starts/minute, 3,000 ms rolling worker CPU/minute, minimum three seconds between starts, and a 24-second same-track cooldown. These are admission limits, not a hard timeout for an already-running decoder.

## Reproducible synthetic evaluation

~~~bash
PYTHONPATH=. python tools/bench_weak_acquisition.py \
  --sample-rate 48000 --seeds 6 --seed-start 71062 \
  --snrs=-10,-5,0,10 \
  --output /tmp/weak-acquisition-48khz.json
~~~

Each independent seed synthesizes RTTY and CW with repeated intact pinned messages, 2.5-kHz SNR reference noise, no deliberate QRM, and separate pure-noise intervals. RTTY tests an arbitrary four-second crop starting 350 ms into its first message. CW tests eight seconds against the latest four-second slice **ending at the same time**, and also records the initial four-second slice beginning at the same time as the eight-second window. Each case includes WAV/fixture SHA256, exact original and selected sample bounds, expected text, decoded text, wall/CPU times, and unverified evidence grade.

Grounded report: [48-kHz, six-seed results](../reports/live-weak-acquisition-48khz-2026-11-22.json). Results below count **exact expected payloads present** (RTTY exact matching; CW expected message as substring because CW is not packet-framed). Incorrect readable text is separately counted, not silently treated as correct.

| Mode / SNR | Original recent 4s exact | Updated exact | Incorrect coherent, unverified | Noise-only false output | Median worker CPU |
|---|---:|---:|---:|---:|---:|
| RTTY −10 dB | 0/6 | 0/6 | 0/6 | 0/6 | 570 ms |
| RTTY −5 dB | 0/6 | **2/6** | 2/6 | 0/6 | 331 ms |
| RTTY 0 dB | 0/6 | **6/6** | 0/6 | 0/6 | 304 ms |
| RTTY +10 dB | 0/6 | **6/6** | 0/6 | 0/6 | 307 ms |
| CW −10 dB | 0/6 | 0/6 | 0/6 | 0/6 | 60 ms |
| CW −5 dB | 2/6 | 2/6 | 2/6 | 0/6 | 59 ms |
| CW 0 dB | 6/6 | 6/6 | 0/6 | 0/6 | 59 ms |
| CW +10 dB | 6/6 | 6/6 | 0/6 | 0/6 | 59 ms |

The old RTTY approach frequently fails when a whole four-second crop contains truncated adjacent transmissions even though one complete shorter message is present. Narrowband acquisition recovers two of six −5-dB RTTY cases and all tested 0/+10-dB cases. The others at −5 dB include wrong but *readable* outputs (for example, VCQ TEST or IQ TEST) and abstentions. Because ITA2 lacks a CRC, these must remain unverified.

The original latest-four-second CW window already contains a complete repeated message in many cases. Eight-second lookback improves contextual evidence and provides a second interpretation; it **does not improve the aggregate CW exact-message count** in this six-seed experiment. At −10 dB both modes abstain. Noise-only audio yielded no graded text in these 48 controls, but this is far too small to establish a real-world false-positive rate.

These are synthetic fixtures, with one modem active at a time, not held-out on-air validation or a statistically calibrated receiver performance study. The earlier [signal-chain SNR benchmark](SIGNAL_CHAIN_SNR_BENCHMARK.md) separately evaluates actual YOLO and narrow-tone detection; these numbers concern native modem acquisition only.

## Verification

~~~bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_native_weak_acquisition.py tests/test_native_acquisition.py \
  tests/test_native_live_decode.py tests/test_native_live_api.py

node --test tests/test_decoder_result.mjs
~~~

The tests cover both sample rates, exact RTTY recovery at −5 dB, conservative weak-tone evidence, negative single-tone and noise tests, the eight-second CW-first ring bound, a secondary CW candidate remaining four seconds, and exact persisted sample provenance from an actual spawned process.

## Remaining experiments

Actual stateful CW key-transition decoding across detection polls is still needed to reconstruct longer messages without resending broken partial prefixes. Other next steps include RTTY soft-symbol acquisition across fading and clock/carrier offsets, multi-signal interference and mismatched shifts, negative data from reviewed real recordings, and better presentation of uncertain per-character evidence. None are completed by this milestone.
