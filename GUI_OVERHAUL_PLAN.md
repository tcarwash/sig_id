# Waterfall-centered GUI overhaul

Follow-up: [Annotation workflow simplification](ANNOTATION_WORKFLOW_PLAN.md)
records the planned window-level draw → label → refine → save workflow.

Status: core overhaul implemented and verified in Chromium at desktop/mobile sizes.
This includes layout, interactive overlays, binary live display, fixed history,
freeze controls, raw-result provenance and saved-label revision/history review.
Fldigi output remains shared or tentative until its audio route provides defensible
per-track attribution; independent decoder workers remain a 1.0 backend task.

## Product goal

Make the waterfall the main place to find signals, understand what is known about
them, select a decoder, and label recordings. Show the next useful action for the
selected signal instead of displaying every tool simultaneously.

The current interface stacks source controls, synthetic import, selection labeling,
Fldigi output, five demodulator forms, candidates and saved regions down one page.
The waterfall already has track and region outlines, but most useful information
and actions require leaving it. Selecting a track also scrolls the page to a form.

## Default workspace

Use the full browser viewport, with three persistent areas:

1. A compact top toolbar: source selector, Start/Stop or Play/Pause, Open recording,
   and a single status indicator. Put audio device, sample rate, decoder connection,
   model selection and display preferences under Settings.
2. A large waterfall taking the remaining space. Keep frequency horizontal and
   time vertical, with the newest audio at the bottom. Preserve the fixed 0–3 kHz
   default frequency range and avoid drag-to-scale axes.
3. A shallow, collapsible output dock showing the selected signal's decoded text,
   image or raw-symbol results. Expand it when more detail is needed.

A right-side inspector opens on selection and closes with Escape. The waterfall
remains visible while editing or decoding. On narrow screens, use a bottom sheet.
Dataset browsing and synthetic generation/import belong in a Library drawer opened
from the toolbar, rather than occupying permanent receive-workspace space.

```text
 Signal ID   [Radio input ▾]  [Start / Stop]  [Open]       [Library] [Settings]
 ┌────────────────────────────────────────────────────────┬──────────────┐
 │ −30 s                                                  │ Selected     │
 │        ┌─ S1 · MFSK candidate ──┐                       │ signal       │
 │        │      signal trace     │   ┌─ S2 · Unknown ─┐   │              │
 │        │                       │   │ signal trace  │   │ Best ID      │
 │        └───────────────────────┘   └───────────────┘   │ Evidence     │
 │                                                        │ Decode       │
 │ Now                                                    │ Label / Save │
 └──────0 Hz────────1000────────2000────────3000 Hz─────────┴──────────────┘
 Selected output · S1 / trial THOR16 · tentative              [Expand] [×]
 Received text or raw symbols…
```

The inspector is closed initially. The wireframe's signal names and states are
illustrative, not claims about current automatic identification coverage.

## Information directly on the waterfall

| Element | Always visible | On hover or selection |
| --- | --- | --- |
| Signal extent | Thin frequency/time outline and stable short track ID | Exact bounds and duration |
| Identification | Best justified label, such as “MFSK candidate” or “Unknown” | Family → method → protocol → variant; alternatives and evidence |
| Decoder | Small Trying / Output / Stopped badge when applicable | Actual backend, selected mode, trial status and routing status |
| Measurements | Optional compact frequency for active tracks | Center frequency, occupied bandwidth, estimated SNR, tone count/spacing, drift and rate estimates when available |
| Dataset annotation | Small saved-label marker | Accepted label, source, revision and annotation bounds |
| Received output | Short preview for the selected or pinned track, if associated | Full output in the dock; image thumbnail when available |
| Cursor | Crosshair while inspecting | Frequency, time, sample position and spectrum level |

Use uncertainty in the text: “MFSK candidate,” “THOR16 trial,” or “User labeled.”
Do not display a model score as a calibrated probability of protocol correctness.
Unknown, rejected and unavailable measurements must remain explicit.

Keep track persistence, identification evidence, decoder activity and accepted
dataset labels as separate states. Repeated spectral detections alone must never
produce a confirmed-mode badge. Arbitrary Fldigi characters do not confirm a mode.

### Keep dense displays readable

- Show one short label per visible track; expand details for the selected track.
- Prefer selected, pinned and active-decoder labels when space is limited.
- Move colliding labels into a narrow annotation gutter with leader lines; show an
  overflow count for crowded groups. Preserve access through the signal switcher.
- Use thin outlines and small opaque label backplates instead of large filled boxes.
- Give the selected track one clear accent. Use icons, text and line styles in
  addition to color to distinguish candidate, reviewed and decoding states.
- Clip overlays to visible bounds and avoid letting labels cover axis ticks.
- Render labels in CSS-pixel coordinates so font size does not depend on FFT size,
  sample rate or the number of waterfall rows.

## Interaction model

- Click a track or its label to select it and open its inspector. Selection itself
  never changes the decoder mode or saves a dataset label.
- Drag empty waterfall space to select a manual time/frequency region. Show its
  bounds and a small action bar: Inspect, Decode, Label.
- Use one native mode dropdown and one explicit Decode action in the inspector.
  Keep automatic trials as an explicit opt-in under receive settings.
- Combine raw analysis into one “Raw symbols” tool with a family/method dropdown.
  Show only the selected demodulator's rate, tone/carrier and timing fields. Put
  detailed search limits, soft metrics and ambiguity hypotheses under Advanced.
- Label opens a compact form with optional family, method, protocol and variant.
  Suggestions remain proposals until the user saves them. Partial labels are valid.
- Click a saved annotation to review it; present revision information in the
  inspector rather than repeating a saved-regions table below the waterfall.
- Escape closes transient controls; Tab reaches track labels and all actions.
  Hover information must also be available on keyboard focus and touch selection.

## Output and decoder association

The current Fldigi backend is externally audio-routed and shares a receive buffer.
A trial's track ID expresses intended association, not proof that the backend heard
only that track. A redraw must not imply simultaneous independent decoding.

Initially, display unassociated text as “Fldigi output · MODE” in the dock. Show
trial-associated output with a tentative badge. Add anchored previews only when
output records contain a defensible capture/session, track, mode and receive-time
association. Preserve timestamps as receive timestamps unless sample timing is known.
Clear or partition output when changing source/session so old text cannot appear
attached to a new signal. Images follow the same association rules.

## Live history and smoothness

Target a fixed 30-second visible history, with the newest row at the bottom.
Separate display history from the existing ten-second analysis window. The first
layout checkpoint may retain ten seconds until the display transport is upgraded.

Replace complete waterfall redraws every second with incremental, batched spectrum
rows. Target visual updates every 100–200 ms while candidate analysis retains its
own bounded cadence. Do not repeatedly calculate and transfer the entire history
to obtain smoother motion.

Use absolute sample positions and session IDs for row deduplication, overlay
placement and selection. Keep startup time scale fixed, leaving the unavailable
history blank rather than stretching the first second over the entire canvas.
Show discontinuities as gaps, not interpolated signal. Bound the display cache to
60 seconds and expose dropped-row/overrun status through the status indicator.

“Freeze view” stops visual following while recording continues; show “Recording ·
view frozen” prominently. Resume returns to the newest rows. Use a separate Stop
action to finish capture. A frozen selection retains absolute bounds. Until live
annotation storage is implemented, label saving remains available after stopping
and saving the capture; the UI must say so beside the action.

## Implementation sequence and tests

### 1. Simplify the shell — first usable overhaul

Commit: `feat(ui): center the workspace on the waterfall`

- Build viewport layout, compact source toolbar, selection inspector and output dock.
- Move synthetic import and dataset lists into Library; move configuration to Settings.
- Replace the five simultaneously visible raw-demodulator forms with one tool selector.
- Reuse current API operations and preserve existing exports and manual labels.
- Remove scroll-to-form behavior when selecting a track.

Verify WAV loading, live start/stop, manual region saving, decoder mode selection,
synthetic import and every raw-demodulator export through the reorganized controls.
Inspect rendered layouts at desktop and narrow widths. No receive workflow should
require scrolling through unrelated analysis forms.

### 2. Put useful signal information on the waterfall

Commit: `feat(ui): add interactive signal labels and evidence overlays`

- Build a shared projection/hit-testing layer and stable selected-track state.
- Add readable track labels, status badges, hover/focus details and annotation markers.
- Add collision handling, gutter placement and keyboard-accessible track selection.
- Connect selection to the inspector and output dock.

Test coordinate mapping against known sample/frequency bounds, overlapping tracks,
offscreen edges, resizing, device-pixel ratios and different FFT resolutions.
Use dense synthetic scenes for rendered inspection. Test that clicking selects,
dragging marks a region, and neither action silently changes modes or saves labels.

### 3. Make the rolling waterfall behave like an instrument

Commit: `feat(ui): stream waterfall rows with fixed history and freeze controls`

- Introduce incremental bounded spectrum transport and a 30-second display history.
- Decouple spectrum updates from track/model analysis and avoid overlapping polls.
- Add fixed startup scale, freeze/follow, session resets and discontinuity markers.

Use a deterministic timed tone/burst stream to verify scrolling, gaps and overlays
against absolute sample positions. Test late/out-of-order updates, source changes,
freeze/resume and capture stop while requests are pending. Measure transfer size,
render latency and memory at 48/96/192 kHz over a sustained run on a named machine.

### 4. Integrate decoding and dataset review

Commit: `feat(ui): unify signal output and labeling workflows`

- Introduce explicit decoder-output association records where backend evidence permits.
- Add safe waterfall previews, expanded output, image viewing and raw-symbol diagnostics.
- Add Library review navigation and full annotation editing/revision workflows as
  backend support becomes available.
- Keep unsupported or unavailable operations visibly disabled with a short reason.

Test session/mode changes and concurrent tracks for stale or misattributed text.
Verify model abstention, tentative output and accepted labels remain distinct.
Exercise partial labeling and exports end to end, including restart persistence.

## Acceptance for the overhaul

- Source, Start/Stop, waterfall and selected output are reachable without page scrolling.
- A user can find, select, inspect and label a signal without losing its waterfall context.
- All visible signals retain an accessible identity even when labels collide.
- Frequency/time orientation and selection sample bounds stay correct at every layout size.
- Candidate identity is visibly provisional until supported by actual evidence or review.
- Existing capture, decoding, dataset import and raw-output export capabilities remain usable.
- UI checks, relevant backend tests and rendered inspection accompany each checkpoint.
- Continue the 1.0 release plan after the GUI checkpoints, as requested.
