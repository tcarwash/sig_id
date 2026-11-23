# Early native decode probes on provisional live tracks

Experimental receive-only addition to sig_id, following
`docs/CONTINUOUS_CW_KEYING.md`. It addresses a real handoff problem:
the live heuristic tracker requires three distinct observations, while
a useful short CW burst may last fewer polls. A classifer that abstains
can still provide *relative* mode rankings, which previously were
discarded by `candidate_modes`.

## Revised admission policy

This affects **automatic native decoder trials only**. No detector confidence
or model ranking becomes a verified signal identity. The UI opt-in toggle
`Auto-try native decoders (CPU limited)` remains authoritative.

A persistent track still follows the existing decode policy. For a
`provisional` track with a session-scoped ID and valid audio geometry:

- At least 1.5 seconds of available source audio, fully within the
  current ring buffer, and 20–3000 Hz audio bounds.
- Strong detector evidence: peak spectral SNR >= 14 dB **or**
  CV detector confidence >= 0.85. These are conservative routing
  thresholds, **not calibrated probabilities**.
- For immediate *CW* admission (the first observation may qualify):
  occupied width <= 80 Hz, plus an accepted CW/Morse prediction
  **or** a clearly leading ranked CW candidate.
- For other modem families: at least two distinct observations and
  at least two seconds of audio; other families retain the original
  at-most-four-second decoder window.

A ranked, abstained hypothesis is eligible **only for trial ordering**
if the top `ranked_candidates[0].relative_model_score` is finite,
>= 0.55 and leads the second score by >= 0.15. Recognized mode IDs are
CW, RTTY, BPSK31 and AX.25. An ambiguous ranking is ignored and
the original spectral fallback remains. An absent classifier is
not a reason to suppress persistent-track trials.

The current CV and heuristic track registries are unchanged; an early
probe never increments `track_observations`, forces a persistent state,
or changes classifier acceptance.

## Budgets and isolation

All trials still run in the single existing decode child process and
never block audio callbacks. A new **3 provisional admissions per rolling
minute** budget applies in addition to:

- 5 general trials/minute, or 16 for narrow CW streams;
- 3000 ms decoder process CPU/minute;
- at most one running worker job and a 3-second global job gap;
- per-track cooldown (2.8 sec for narrow CW, 24 sec otherwise).

These are admission limits, not preemption guarantees. Resource usage
is reported in `/api/live/native-decode/status` as
`provisional_probes_last_minute`,
`max_provisional_probes_per_minute`, and
`counts.admitted_provisional` /
`counts.skipped_provisional_budget`.

The live detection objects expose `native_decode_admission` status and
the decoder run records `track_state_at_admission`,
`track_observations`, and `provisional_probe_kind` as provenance.
The existing `cw_stream` result and `live_assistance` payloads remain
`confirmed_identity: false`.

## Verification

```bash
PYTHONPATH=. /storage5/dev/sig_id/.venv/bin/python -m pytest -q \
  tests/test_native_live_decode.py tests/test_native_live_api.py \
  tests/test_native_weak_acquisition.py tests/test_cw_keying_stream.py \
  tests/test_tracking.py
```

The API integration test sends a real Signal Lab CW waveform through
the FastAPI route using a deliberately abstaining ranked classifier,
and verifies that the first provisional observation queues a worker
and obtains tentative CW output, without creating an accepted label.
Other tests exercise quota enforcement, abstention routing, mode
ranking, wrong-mode controls, and no-worker behavior when disabled.

## Next measurement

Run sustained independent CV and heuristic polling against the
**separate** port-8014 preview using the existing Signal Lab monitor.
Measure detector hit counts, how many tracks reach persistent state,
how many early probes are admitted, decoder output, and rolling CPU usage.
Do not change the port-8013 preview or main checkout, and do not call
an abstention a positive decode.

## Live Signal Lab monitor run — 2026-11-22

The updated side preview was restarted at 127.0.0.1:8014 and scanned
the pre-existing `sigid_signal_lab.monitor` source (48 kHz, stereo,
`scan_only` capture). The 100-second fixture loop contained CW at 260 Hz,
RTTY at 740 Hz, BPSK31 at 2520 Hz and AX.25 at 1200 Hz; it was not
changed or re-transmitted over RF. The separate port-8013 preview stayed
healthy. The 26 API polls used the standard heuristic track pipeline
with narrow-tone thresholds and 1.65 seconds between polls.

| Observed result | Value |
| --- | --- |
| Tracks on first 15 polls | none (source/fixture phase) |
| First CW-like track | provisional, observation 1 |
| First trial | observation 2, `provisional_modem` |
| Track became persistent | observation 3 |
| Admitted native jobs | 5 total, 1 provisional |
| Coherent native events | 2 (`CQ TEST`, then `CQ`) |
| Rolling native CPU | 327.64 ms / 3000 ms cap |
| CW worker latency | median 51.28 ms, p95 96.79 ms (5 attempts) |
| RTTY worker latency | median 21.41 ms (3 attempts) |
| Verified mode identity | none; tentative output only |

The classifier's top suggestion moved between Morse and BPSK31 and its
acceptance remained unverified. Because the first observation did *not*
satisfy the strong ranked-CW gate, it correctly did **not** trigger a
one-poll CW probe. The second observation qualified for the general
provisional modem probe; its output abstained. Following persistence,
continuous CW-keying became eligible, and two later polls produced
unverified but coherent Morse text. The first three CW attempts did not
produce published text; they are not counted as successes.

The measured raw monitor log is stored outside Git in
`/storage5/dev/sig_id-cw-preview-data/provisional-trial-monitor.json`.
This is a controlled synthetic audio loop going through the real capture
stack, **not** independent over-the-air RF field validation. CV-specific
admission has FastAPI integration coverage but was not exercised by this
particular live run.
