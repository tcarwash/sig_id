# Evidence-guided identification chain (first integration slice)

The intended receive path is:

1. **Audio/WAV → spectrogram.** The sample timeline remains authoritative.
2. **Class-agnostic CV detector → time/frequency region.** A YOLO bounding box
   supplies sample start/end and audio-frequency low/high bounds, not a mode.
3. **Bounded audio crop → ROI-aware classifier.** The current trained model
   extracts spectral/envelope features from a two-second crop (when the model
   supports ROI features, it applies an expanded bandpass).
4. **Family/expert ML → *ranked* mode hypotheses.** Both flat and hierarchical
   model bundles now retain the top alternative modes. The hierarchical model
   scores other families' experts too; the joint score is a relative ranking,
   **not** a real-world calibrated protocol probability. Existing model
   abstention/novelty handling remains authoritative.
5. **Independent audio evidence.** The `audio-roi-spectral-v1` helper measures
   a bounded selected interval: frequency resolution, excess-power occupied
   bandwidth, spectral flatness, peaks, possible tone spacing, and a burst
   activity fraction when temporal variation supports it. Peaks alone do not
   establish modulation, and no RF-domain feature is inferred from audio.
6. **Conservative decoder shortlist.** Up to three modes are ordered using the
   ranked hypotheses and the existing family fallbacks, but automatically
   tried modes are still limited to the project's loopback-tested set:
   `RTTY`, `THOR16`, `BPSK31`. A higher-scoring candidate does not become a
   confirmed identity. Unknown and competing hypotheses remain visible.
7. **Decoder evidence + review.** Isolated replay and output association stay
   unchanged. Text is tentative, and labels require their existing evidence
   and review policy.

## Where the first slice is wired

- Saved capture `POST /api/captures/{id}/classify` returns
  `ranked_candidates`, `dsp_evidence`, and `decoder_shortlist`.
  Classifier evidence attached to a saved region keeps the same fields.
- `GET /api/live/cv-detections` now associates CV boxes with
  **persistent, session-scoped CV track IDs**. Separately configured heuristic
  tracks retain their own IDs. On explicit `auto_trial=true`, accepted model
  proposals can schedule a bounded, receive-only isolated decoder trial with
  detector/classifier provenance.
- `GET /api/live/analysis` still supports the heuristic fallback, including
  optional trials when CV is unavailable or disabled.
- The browser reconnects to an existing server-owned live capture after a
  refresh, restoring the live waterfall and Stop and review.
- `FldigiTrackTrialRequest.candidate_modes` is optional and every entry is
  checked against the loopback-tested allowlist at the queue boundary.
- Opt-in `include_frequency_refinement=true` adds fine-resolution, frequency-
  only DSP split **previews** to live and saved CV detections. These are
  proposals, not new tracks, automatic labels, or decoder results. See
  [Crowded-signal benchmark](CROWDED_AUDIO_BENCHMARK.md) for reproducible
  synthetic and archived 40m real-audio checks.

CV and heuristic detectors share the *isolated receive-only decoder queue*,
but not their track registries. CV remains optional; the heuristic path is
available when the model is unavailable.

## Inspecting CV geometry and independent symbol timing

The waterfall's **Debug CV bounds** switch renders a lightly shaded
time/frequency rectangle with a midpoint guide. The compact label includes
the CV-reported audio-frequency bounds and the detector score. These are the
detector's proposed bounds, not a measured carrier or confirmed identity.
The normal overlay remains available when debugging is off.

Each CV result lists its top three **relative, uncalibrated** hierarchical
mode scores. On-demand **Inspect DSP** uses a bounded excerpt (at most 16 s)
from the already-captured audio to measure independent, narrowband
instantaneous-frequency transitions. The helper
`sigid/timing_evidence.py` reports a *possible* tone grid and a candidate
frequency-change clock when it can establish sufficient periodicity. It
abstains on stationary tones, too-short crops, broad/crowded ROIs, and
ambiguous timing. Neither the classifier scores nor these measurements imply
a validated modulation identity, protocol, frame sync, or decode.

Read-only API endpoints:

- `GET /api/live/dsp-evidence?session_id=...&sample_start=...&sample_count=...&frequency_low_hz=...&frequency_high_hz=...`
- `GET /api/captures/{capture_id}/dsp-evidence` with the same geometry
  parameters (except `session_id`).

The endpoints never start or stop capture, save labels, or invoke a decoder.
Live audio that has aged outside the bounded ring returns HTTP 409; use a
newer track or the saved WAV. Tone-transition evidence specifically targets
frequency-stepped modulation. PSK, ASK, phase-only, and many other emissions
will require separate estimators rather than forced FSK-like results.

In the locally retained 40m FT8 example, the 1719–1830 Hz box produced
a **6.25 baud / 160 ms** transition-clock candidate with an approximately
6 Hz tone-grid candidate. Adjacent boxes were inconclusive for timing.
This is an FT8-*compatible measurement*, not an independent FT8 decode.
See [Crowded audio benchmark](CROWDED_AUDIO_BENCHMARK.md).

## Tests and next gates

```bash
python -m pytest -q tests/test_identification.py tests/test_classifier_api.py tests/test_tracking.py tests/test_api.py tests/test_frequency_refinement.py tests/test_crowded_benchmark.py tests/test_frequency_refinement_api.py tests/test_timing_evidence.py tests/test_timing_evidence_api.py
```

Before widening automatic identification, benchmark the whole chain on
held-out real captures: false detections/hour, correct *family* top-3,
mode top-3, latency, abstention quality, decode-supported identifications,
and explicit unknown retention. Test crowded/overlapping passbands and
partial transmissions. Candidate tests using our own synthetic signals
cannot establish independent on-air accuracy.

Next engineering slices:

- Improve CV track association for simultaneous same-frequency bursts and
  interrupted receptions; benchmark real on-air tracking failure cases.
- Aggregate ranked model alternatives across multiple independent crops,
  with uncertainty rather than committing to one cropped sample.
- Add independently measured audio symbol-clock and cadence features,
  then protocol-specific sync verification without treating inference
  as proof.
- Grow validated decoder adapters and expand their receive-only mode
  catalog only after controlled loopbacks and independent RF checks.

## Upstream inspiration and provenance

The staged evidence, modulation measurements, and validation philosophy
were informed by [OnAir](https://github.com/adswill/OnAir), particularly
its `docs/IDENTIFIER.md` and `docs/SIGNAL-LAB.md`. This first slice is
written independently in Python and does not include OnAir source code.
Review licensing before ever importing/reusing upstream implementation.
