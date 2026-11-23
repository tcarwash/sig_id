# First-party live modem event integration: RTTY, BPSK31 and AX.25

**Experimental / receive-only.** This is an addition to the project's
independent `sigid/libsigdecode` decoders. It does **not** import or
run Fldigi, OnAir, WSJT-X, CAT/PTT or any transmitter.

## Existing decoders, now live event streams

| Mode | First-party decoder | Live publication | What counts as evidence |
| --- | --- | --- | --- |
| RTTY / 45.45 Baudot | `rtty.py` | New absolute-sample-indexed text tokens | Coherent but unverified ITA2 framing; no CRC |
| BPSK31 | `bpsk31.py` | New absolute-sample-indexed varicode tokens | Coherent but unverified text; no CRC/FEC |
| AX.25 Bell 202 1200 | `ax25.py` | New absolute-sample-indexed frames | X.25 CRC/FCS check passed, not a reviewed station or mode ID |
| CW | `cw_keying_stream.py` | Prior continuous key-transition words | Unverified Morse timing; separate from modem stream |

The decoder algorithms still run on bounded mono windows in an isolated
one-worker process. Their waveform demodulation is **not** yet a true
bit-by-bit sample-continuous demodulator. The new
`sigid/modem_live_stream.py` assembles the decoder **events**, not bits,
across live polls. Each mode/track/session has bounded state; overlapping
sample-time evidence is not repeatedly inserted into the transcript.
An identical message repeated later, at different source samples, remains
a new event, separated by a newline for keyboard modes when a silent gap
is observed. Each decoder run retains its raw original results and the new
`modem_stream` object showing the delta, suppression and cumulative text.

### BPSK31 coarse-to-fine carrier acquisition

CV boxes and spectral detectors return approximate frequency regions,
while the existing BPSK31 decoder needs a carrier close to the correct
center (normally within ±3 Hz, sometimes more precisely for a complete
first varicode character). The native worker now tries one conservative
spectral carrier proposal for BPSK31 using the narrowband analytic signal
squared, where a BPSK phase-reversal carrier produces a narrow line at
twice its frequency error. The sign is retained, searching at most ±35 Hz.
A proposal requires a strong peak-to-median ratio (>=18) and a meaningful
correction (>=3.5 Hz). It does **not** itself identify BPSK31; keyed CW can
also produce a square-law line. Every candidate must still pass the
original varicode decoder's symbol coherence and codeword checks.

For example, the pinned 8-kHz fixture with its true 1000-Hz carrier but
an initial 1011-Hz detector center yields a proposed center around
999.932 Hz and a spectral ratio around 86. This is useful frequency
evidence, but the existing demodulator still rejects the incomplete
leading varicode symbol in that case; no text is falsely published.
The caller sees the proposal in persisted
`decoder_result.live_acquisition.carrier_refinement`. Pure AWGN, silence,
RTTY and AX.25 negative controls generate no proposal. A keyed CW signal
can generate a proposal, but its BPSK31 decoding still abstains.
**PSK31 acquisition/first-symbol robustness is not complete.**

The stream only publishes decoded RTTY/PSK31 characters when the independent
mode decoder already passed its conservative text gate. AX.25 requires a
valid structural UI frame **and independently passed FCS**. Bad CRC, noise,
insufficient confidence and repeat windows publish nothing; they remain
visible in saved evidence as abstentions.

The rolling streams are capped at 20 active mode/track associations and
160 recent items per stream, with 1024 characters of cumulative output.
Sessions and sufficiently large carrier changes reset association.
This caps display/evidence accumulation but not capture duration.

## Admission and CPU

The existing opt-in `Auto-try native decoders (CPU limited)` is unchanged.
The path has one isolated worker and zero queued backlog, never touching
capture callbacks or transmit controls.

- Existing normal trials: up to 5 per rolling minute and a 24-second
  same-track cooldown.
- **Modem follow-up lane:** for *persistent* tracks whose primary mode is
  RTTY/BPSK31/AX.25 and whose model accepted that mode or ranked it clearly
  ahead of alternatives, up to **10 total native starts per rolling minute**
  and a 5.8-second same-track cooldown. This is not an extra worker or ten
  trials *per mode*. All other candidates retain the normal limit.
- Existing CW lane: up to 16 starts/minute; existing source-aware Morse
  stream remains independent.
- Shared: 3000 ms measured decoder process CPU/minute, at most one
  running job, >=3 seconds between starts, at most 2 modem hypotheses/job,
  no queue backlog, and max 3 provisional trials/minute.

These are admission budgets, not preemption. RTTY decoding is CPU-heavy;
a session can hit the shared 3000-ms cap before reaching 10 starts.

The former 0.4-second minimum source crop inadvertently rejected real short
AX.25 packets before demodulation. The first AX.25 hypothesis now permits
>=0.125-second crops, matching the decoder's own lower bound. Other modes
keep their previous 0.4-second minimum. This does not bypass persistent
track criteria or frame-integrity checks.

## API/UI provenance

`/api/live/native-decode/status` gains
`max_modem_followup_attempts_per_minute`,
`modem_stream_tracks`, `modem_stream_units`,
`modem_stream_frames`, and
`counts.admitted_modem_followup`. UI shows new modem text tokens,
CRC-passed frames and per-mode latencies alongside the shared CPU budget.

Persisted `decoder_result.modem_stream` contains `delta_text`,
`delta_tokens` / `delta_frames`, `cumulative_text`,
`last_emitted_end_sample`, `suppressed_overlap_items`,
`confirmed_identity: false` and `not_verified: true`.
Original `decoder_result.tokens` and `decoder_result.frames` remain
available for forensic review. A CRC-passed packet is stronger integrity
evidence, but neither a modem classifier label nor authenticated sender.

## Regression coverage

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_modem_live_stream.py tests/test_native_live_decode.py \
  tests/test_native_live_api.py
node --test tests/test_decoder_result.mjs
```

- Pinned independent decoder fixtures recover `CQ TEST`, `cq de` and
  `HELLO WORLD` when the selected carrier is correct.
- The BPSK31 carrier estimator is checked against deliberately offset
  positive audio, keyed-CW cross-mode false proposals, and noise/RTTY/AX.25
  negatives. A frequency proposal alone never yields text.
- Running the actual native worker three times per mode against the same
  source interval, then a later true repeat, yields **one first event,
  no duplicate, one later repeated event**.
- Frame events require FCS-passed AX.25 evidence; failed/missing integrity,
  wrong decoder identity, out-of-window sample references and unrelated
  sessions cannot generate valid events.
- A real FastAPI CV-to-worker test exercises the previously rejected
  sub-400-ms AX.25 frame and persists its CRC-passed result without
  calling it verified identity.
- Existing CW, negative-signal, audio and scheduler regression tests
  are expected to remain unchanged.

## First side-preview monitor check (known limitations)

The updated isolated port-8014 process was used for one **scan-only,
receive-only** monitoring session against the pre-existing synthetic
`sigid_signal_lab.monitor` loop, using both CV and heuristic live API
polls. Over 30 polls and approximately 100 seconds of synthetic audio,
8 native jobs were admitted: 2 BPSK31 and 6 CW. The BPSK31 jobs abstained,
and the monitor did not admit an RTTY or AX.25 job from its current
CV/heuristic detector boxes. Total native decoder CPU reported was about
479 ms (well inside the shared budget). The test was run before adding
the bounded BPSK31 carrier-refinement proposal.

This is **not** successful live validation for all three modems. It shows
that the modem decoder/API path works for deterministic selected regions,
but detector-side frequency grouping and eligibility remain incomplete
for the multi-mode Signal Lab. In particular, the real CV boxes suggested
BPSK31 around 2528–2532 Hz when the Signal Lab placed it at 2520 Hz.
The correct-by-spec decoder abstained. The carrier-refinement addition
reports that discrepancy; it does not relax varicode coherence/CRC gates.
Raw side-preview metadata for this exercise is outside Git at
`/storage5/dev/sig_id-cw-preview-data/live-other-modems.json`.

**Next pipeline target:** a bounded multi-tone geometry and mode-shortlist
check for RTTY and Bell 202 AX.25, plus symbol-level BPSK31 acquisition
robustness. Do not equate a fixture/CRC test with demonstrated unprompted
real-world mode discovery.

## Roadmap for the *other* modes

The machine-readable `docs/decoder-work-items.json` catalog remains
authoritative and distinguishes planned from experimental implemented modes.

1. **APRS payload:** parse a restricted subset only after AX.25 FCS passes;
   never confuse APRS syntax recognition with a new RF demodulator.
2. **FT8/FT4/WSPR:** plan independent UTC-slot synchronization, symbol
   detection, interleaving/FEC and CRC (where applicable). Test against
   independently generated golden messages before live auto-admission.
   The existing classifier's "FT8-like" output is not decoded FT8.
3. **HF keyboard families:** add mode variants such as PSK63/QPSK31,
   Olivia, Contestia, THOR, DominoEX, and MFSK using per-mode test vectors
   and separable timing/FEC primitives.
4. **Packet extensions:** broader AX.25 UI types, multiple frames per crop,
   300-baud HF packet, and 9600-baud direct FSK only after measured
   reception and standards-grounded framing tests.

Do not claim all these modes are implemented. Keep independent decoder
work, sources and evidence in the existing API-spec / agent playbook
framework. The next practical work is real/crowded audio validation and
improving true symbol-state streaming for the keyboard modes.
