# Crowded-signal benchmark: 40m FT8 development fixture

This benchmark is designed to validate the **generic audio signal detection
chain**, not to turn the detector into an FT8-only implementation. It uses
crowded 40m FT8 as a useful real-world case because many independent narrow
signals coexist in a single audio passband.

## Conservative high-resolution refinement

The optional `sigid.frequency_refinement.refine_frequency_region` stage
accepts a CV-selected time/frequency region. It analyzes at most 16 seconds,
resamples high-rate input to 6 kHz, and uses a 4096-point Hann-window STFT
(**about 1.46 Hz resolution** at 6 kHz). Each time frame is whitened against
nearby frequency bins to suppress broadband receiver noise; candidate
sub-bands are built from persistent *excess* spectral occupancy.

A split is suggested **only** when the audio provides at least two broad
enough, spectrally separated occupied components. The analyzer deliberately
abstains on noise, one narrow carrier, one hopping signal, and sparse
two-tone FSK. It cannot reliably separate completely overlapping emissions
or prove that separate frequencies represent separate stations.

For a live capture, enable **Receive settings → Live signal analysis →
Preview fine-frequency splits**. An opt-in query argument
`include_frequency_refinement=true` is also available on:

- `GET /api/live/cv-detections`
- `POST /api/captures/{capture_id}/cv-detections`

When a suggestion exists, the UI lists approximate sub-band frequencies and
states **unverified; original box retained**. It never silently generates new
track IDs, saves training labels, or starts a decoder. The live API limits
analysis to eight highest-scoring boxes per request; saved-capture
refinement does likewise. The option defaults to **off**.

## Offline benchmark

Use the read-only CLI on a local PCM16 WAV and a JSON CV proposal export:

```bash
sig-id-benchmark-crowded /path/to/40m-recording.wav \
  --cv-json /path/to/cv-boxes.json \
  --output /path/to/report.json
```

It also works without installation:

```bash
python -m sigid.crowded_benchmark /path/to/40m-recording.wav \
  --cv-json /path/to/cv-boxes.json
```

For the archived 40m sample in this project, the observed (unreviewed)
CV proposals are stored as
[`reports/40m-ft8-cv-boxes-2026-11-21.json`](../reports/40m-ft8-cv-boxes-2026-11-21.json).
Run the same CLI against the locally retained WAV to reproduce the
three-region no-split result. The recording itself is intentionally not
committed.

The `cv-boxes.json` file can be a list or an object with a `detections`
array. Each entry needs `sample_start`, `sample_count`,
`frequency_low_hz`, and `frequency_high_hz`, or
`start_seconds`/`end_seconds` instead of the sample fields.
The benchmark stores input WAV SHA-256, source filename, region geometry,
split/abstention decisions, and any supplied validation results. It does
**not** upload audio, modify input, make decoder calls, or add database labels.

To obtain actual accuracy metrics, independently review signal boundaries
(including time) and supply `--reference-json reviewed-bands.json` with the
same box schema. The report then gives original-vs-refined one-to-one matches,
missed references, unmatched proposals, precision, and recall **within the
provided CV boxes**. The matcher requires at least half the shorter time
interval to overlap and at least 0.25 frequency IoU. It is not a substitute
for measuring end-to-end CV missed detections.

## Initial benchmark results (2026-11-21)

**Controlled synthetic overlap**: Two uncoded 8-tone hopping signals with
6.25 Hz tone spacing, one beginning at 1000 Hz and the other at 1075 Hz,
inside a single broad 980–1140 Hz parent CV box. The fine-resolution stage
proposed two separate sub-bands. Against the known synthetic bounds,
one-to-one recall improved from 1/2 (one broad box) to 2/2. These signals
are deliberately **not claimed to be valid FT8 packets**.

**Real 40m capture**: The preserved recording
`live-7d7ed981-d896-4cc0-9e9c-6c91499c98df.wav`
is approximately 599.72 seconds at 48 kHz, with WAV SHA-256
`4ab6af5e586fb1704ef7ae90fa923cf5a8d0d7ae87d3b1819cfa35cf83550d63`.
Three CV regions observed around 225 seconds were replayed:

| Original audio band | Refiner result | Suggestions |
|---|---|---:|
| 860–992 Hz | One resolved spectral component | 0 |
| 1719–1830 Hz | One resolved spectral component | 0 |
| 2125–2247 Hz | One resolved spectral component | 0 |

The new **on-demand frequency-transition timing inspector** was also
replayed against the same three original CV regions without modifying the
underlying capture:

| Audio band | Observed tone-grid candidate | Clock candidate |
|---|---:|---|
| 860–992 Hz | ~6 Hz | Inconclusive |
| 1719–1830 Hz | ~6 Hz | **6.25 baud / 160 ms** |
| 2125–2247 Hz | ~9 Hz | Inconclusive |

The middle region therefore supplies independent FT8-compatible timing
evidence despite its earlier PSK/BPSK31 model hypothesis. The other
measurements do not establish their modulation family, and tone-grid
candidates without a coherent clock should be treated cautiously.
**No independent FT8 decode or reviewed station identity exists for these
three regions.**

The refiner preserved all original CV boxes. **No independent manual or
WSJT-X-derived reference boxes were supplied**, so there is no valid
real-world recall, precision, or confirmed station count for this sample.
This negative result is still useful: the stage did not force fake splits
on three broad but already plausibly single-source regions. The original
audio remains local and is **not** committed to the repository.

## Next validation steps

1. Record or export time-aligned decoder reference intervals (prefer
   actual received-message evidence, not inferred ML labels) across multiple
   15-second slots. Distinguish missed CV boxes from incorrect refinement.
2. Measure both crowded and quiet segments, weak adjacent stations,
   overlapping station frequencies, drift, FSK tone alternation, and
   receiver gain changes. Report false splits as seriously as missed splits.
3. Only after those results justify it, consider promoting refined bands
   into separately tracked proposals. Keep operator confirmation and
   decoder-derived identity separate.

Tests: `tests/test_frequency_refinement.py`,
`tests/test_crowded_benchmark.py`, and
`tests/test_frequency_refinement_api.py`.
