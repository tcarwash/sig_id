# Automatic native decoding during live Signal ID monitoring

Signal ID can now attempt **receive-only first-party modem decodes while live CV or narrow-tone detection is running**, without requiring a saved capture. Decoder output is independent classification evidence, not a label that silently replaces the ML classifier.

In **Settings → Receive settings**, enable **Auto-try native decoders (CPU limited)**. Enabled by default for the live workbench; uncheck to disable all automatic native decode jobs. The separate **Auto-try validated decoders** option continues to control the existing external isolated Fldigi mode sweeps, not the native pipeline.

## What triggers a trial

The browser's existing live detection polls carry `auto_native_decode=true` when enabled. A detector must produce a **persistent track**; provisional and historical candidates do not trigger work. The service extracts up to four seconds of the exact current, absolute-sample-indexed mono input around the detector region and tries at most two native mode hypotheses.

Classifier method/family suggestions narrow the mode shortlist when available; **classifier abstention does not forbid a trial**. The fallback shortlist is based on audio frequency and occupied bandwidth and is explicitly an *unverified hypothesis*. The first-party library currently offers CW, RTTY, BPSK31 and AX.25. AX.25 uses its present fixed Bell 202 1200/2200-Hz tone pair.

All modem DSP executes on a dedicated **single-worker child process**, not in the audio callback or the HTTP detection request. It owns no sound hardware, CAT/PTT, network decoder service or transmit action. It is fully separate from Fldigi, OnAir, and WSJT-X.

## Admission and performance limits

The following default admission controls are applied **per server**:

| Control | Value |
|---|---:|
| Native decoder process workers | 1 |
| Concurrent/queued decoder jobs | 1 running / 0 queued |
| Maximum candidates per attempt | 2 |
| PCM input window | Up to 4 seconds |
| Starts per rolling minute | 5 |
| Decoder process CPU budget | 3,000 ms per rolling minute |
| Minimum time between starts | 3 seconds |
| Same detector track cooldown | 24 seconds |
| Recent performance/result history | Bounded to 40 events |

An overloaded or busy worker **skips** candidates instead of building a latency backlog. The budget limits *new admissions* based on measured process CPU; it is not a real-time preemptive deadline for a decoder already running. Because only one child process is active, a pathologically slow decoder can still use one core while finishing. Stronger process-level timeouts remain a follow-up. Mode algorithms have separate regression and conformance vectors.

The **Native live decode** row in Receive settings shows jobs/minute, CPU milliseconds/minute, estimated single-core load percentage, and per-mode median/p95 latency. The results section shows only coherent text or frames, with an explicit unverified qualifier.

Additional machine-readable metrics:

    GET /api/live/native-decode/status?session_id=<active-live-session-id>

Fields: `attempts_last_minute`, `cpu_ms_last_minute`, `cpu_pct_single_core_last_minute`, `max_attempts_per_minute`, `max_cpu_ms_per_minute`, `busy`, `counts`, `latency_by_mode`, `recent`. Counts include skips due to busy, cooldown and budget exhaustion. CPU load is the isolated **native decoder process's** CPU, not the entire app, CV inference, or waterfall rendering. Compare UI latency separately.

## Classification assistance, not identity assignment

A native decoder's candidate output is graded conservatively:

- **frame-integrity-passed:** a decoded frame with an explicitly passed CRC/checksum integrity check. This is stronger evidence for that decoded payload, **not a reviewed mode identity**.
- **coherent-unverified:** a decoded text string that meets minimal length/printability/character variety criteria. These heuristics reject some garbage but are not calibrated correctness scores.
- **none:** abstained, partial, failed, or incoherent result. No text is promoted to the UI.

The decoder's mode suggestion and recovered text are attached to the **same detector track**, kept apart from the classifier's own proposals, and displayed in the live waterfall label and separate native results section. No manual annotation, label, protocol identity, or model training truth is changed automatically. All attempts—including failures and abstentions—are persisted as structured decoder runs with exact sample-window provenance for later comparison. They can be collected for review only through the usual deliberate workflow.

Unrelated noise, overlapping modems, bad acquisition cropping, and negative SNR can produce no decode even when detection succeeds. A valid FCS does not imply the radio or human sender's identity.

## Verification and evaluation

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
      tests/test_native_live_decode.py tests/test_native_live_api.py

The tests cover pinned positive CW/RTTY/BPSK31/AX.25 waveforms, an actual spawned worker process, CV persistence, a classifier-disabled successful RTTY decode, opt-out, deduplication, invalid/stale detector tracks, one-at-a-time admission, rolling CPU budget, result persistence and unverified evidence semantics.

## Initial synthetic benchmark (not a field performance claim)

Run `PYTHONPATH=. python tools/bench_native_live_decode.py --iterations 14 --output /tmp/native-live-report.json` from the checkout. This creates its own temporary test database, feeds a pinned RTTY fixture into a fake CV detector, and compares identical in-process FastAPI detection requests with native auto-decoding disabled/enabled. It neither opens a soundcard nor touches the live radio/preview instance.

The first 14-request measurement on `tyler-ldesktop` gave **0.99 ms median / 15.14 ms p95** with native work disabled, versus **1.53 ms median / 4.41 ms p95** with native work enabled. The difference in medians is about **0.54 ms/request**; the p95 distributions are too small/noisy to support a general speedup claim. RTTY consumed **405 ms** decode time (about **411 ms** total child CPU including its BPSK31 alternative), representing **0.68% of one core over a minute** for that attempt. All 13 concurrent requests while the worker was busy skipped additional modem jobs, preventing backlogs. Earlier direct decoder measurements were approximately CW 6–28 ms, RTTY 385–441 ms, BPSK31 5–7 ms, and AX.25 2–8 ms per fixture, depending on sample rate. Model inference, UI rendering, longer live recordings and noise/interference can change these results.

For field performance, compare the same Signal Lab recording with the switch off/on and look at `/api/live/native-decode/status`. The synthetic SNR and ground truth are available from Signal Lab's JSON manifest. New native results should not increase latency of the audio callback; real-world p95 waterfall/update times and memory should be benchmarked independently of the isolated decoder process.

This first phase is **browser-poll triggered**: if the browser is closed, the server does not continuously run native decoders in unattended mode. Unattended integration and adaptive per-mode CPU budgets are future extensions.
