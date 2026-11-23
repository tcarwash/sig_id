# Incremental CW word assembly across live polls

This receive-only milestone improves Signal ID's live identification and annotation workbench by tracking CW character evidence across overlapping audio windows. It never turns Morse-looking text into an operator-verified mode identity.

## What streams

A persistent, narrow (up to 80 Hz), CW-first detector track may request up to **16 native worker jobs/minute** rather than waiting 24 seconds between attempts. The minimum start gap remains 3 seconds, the one isolated child process never queues work, and all modems share the same **3000 ms/minute rolling worker CPU admission limit**. The previous 5 starts/minute and track cooldown still apply to non-CW, wide, and uncertain tracks.

The same established CW decoder receives up to eight seconds from the already-present ten-second live analysis ring, comparing a recent four-second slice when indicated. Other native modes remain capped at four seconds. This does not add a new causal audio filter or change the recording callback.

New *CWTokenStream* state is scoped to session ID, detector track ID, sample rate and carrier frequency (8 Hz tolerance). It uses absolute character sample positions and Morse patterns to identify **complete words reproduced in two overlapping decoder windows**, accepting no more than 75 ms position difference. It rejects window-edge fragments, low-fit characters, unknown Morse symbols, and implausible partial words. It requires at least five corroborated characters before the first publication.

Only newly corroborated words are emitted. A repeated window produces an empty delta, not duplicate text. Source gaps, frequency or rate changes, session changes and track resets prevent accidental joining of unrelated CW. Result state and transcript history are bounded.

Saved native decoder results retain their original window text, tokens and acquisition provenance, and additionally expose **cw_stream** with delta, cumulative text, source word sample bounds, and explicit *confirmed_identity=false / not_verified=true*. The run's live output text is the new delta. The existing decoder comparison workbench can show the delta and accumulated transcript separately from raw decoder output. No auto-labels, callsign claims or training truth are generated.

This is **stateful word/token accumulation across live polls**, not a newly implemented sample-continuous key-down/up demodulator. A prototype causal keying filter failed accuracy tests and was excluded.

## Controlled SNR evaluation

~~~bash
PYTHONPATH=. python tools/bench_cw_token_stream.py \
  --rate 48000 --snrs=-10,-5,0,10,25 \
  --seeds 6 --seed-start 71062 \
  --output /tmp/cw-token-stream-48khz.json
~~~

Each 45-second Signal Lab scene has one CW station, repeated pinned CQ TEST, 2.5-kHz reference-noise bandwidth and no intentional QRM. The test uses overlapping eight-second windows ending at 8, 11, 14, 17 and 20 seconds, then five background-only windows ending at 32, 35, 38, 41 and 44 seconds. WAV and fixture hashes, source sample coordinates, raw/window text, stream deltas, noise outputs and CPU times are recorded. The test does not open audio devices or transmit RF.

[Machine-readable 48-kHz results](../reports/live-cw-token-stream-48khz-2026-11-22.json):

| Nominal SNR | Exact phrase in assembled output | Incorrect emitted words | Duplicate source sample intervals | Noise-only emitted words |
|---|---:|---:|---:|---:|
| +25 dB | 5/6 | 0 | 0 | 0 |
| +10 dB | 6/6 | 0 | 0 | 0 |
| 0 dB | 6/6 | 0 | 0 | 0 |
| -5 dB | 1/6 | 0 | 0 | 0 |
| -10 dB | 0/6 | 0 | 0 | 0 |

Median **total native CW worker CPU for ten positive and negative polls combined**, not a single request, is about 0.40–0.42 seconds per synthetic case. This does not measure end-to-end HTTP p95 or behavior under many simultaneous modes. The corpus is deliberately small and uses one repeated phrase; it does not establish held-out on-air accuracy, general false-positive rates, or gains in raw weak-SNR demodulation.

The measured improvement is **temporal continuity, de-duplication and conservative abstention**. Some weak signals produce no stream output even when a single window yields plausible text. All incremental words remain unverified.

## Regression and safety

~~~bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest -q \
  tests/test_cw_token_stream.py tests/test_native_live_decode.py \
  tests/test_native_live_api.py tests/test_native_weak_acquisition.py

node --test tests/test_decoder_result.mjs
~~~

Coverage includes absolute-sample matching, duplicate-window abstention, invalid symbols and edge fragments, session/carrier/rate/gap isolation, noise-only output, distinct CW/other rate limits, preserved decoder raw evidence, live delta run persistence and real spawned worker execution.

## Next experiments

A genuinely **sample-continuous CW key-transition decoder** remains open: streaming bandpass/downmix filter state, adaptive on/off threshold, online dot/dash timing, partial symbol carry-over and verified source continuity. The first naive causal-filter attempt produced wrong characters and failed 48-kHz test audio, so it was not shipped. Future evaluation should include independently keyed and reviewed CW, mixed QRM, drifting carrier, unique messages, varied WPM and non-synthetic negative controls.
