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
5. **Independent audio evidence.** The \`audio-roi-spectral-v1\` helper measures
   a bounded selected interval: frequency resolution, excess-power occupied
   bandwidth, spectral flatness, peaks, possible tone spacing, and a burst
   activity fraction when temporal variation supports it. Peaks alone do not
   establish modulation, and no RF-domain feature is inferred from audio.
6. **Conservative decoder shortlist.** Up to three modes are ordered using the
   ranked hypotheses and the existing family fallbacks, but automatically
   tried modes are still limited to the project's loopback-tested set:
   \`RTTY\`, \`THOR16\`, \`BPSK31\`. A higher-scoring candidate does not become a
   confirmed identity. Unknown and competing hypotheses remain visible.
7. **Decoder evidence + review.** Isolated replay and output association stay
   unchanged. Text is tentative, and labels require their existing evidence
   and review policy.

## Where the first slice is wired

- Saved capture \`POST /api/captures/{id}/classify\` returns
  \`ranked_candidates\`, \`dsp_evidence\`, and \`decoder_shortlist\`.
  Classifier evidence attached to a saved region keeps the same fields.
- \`GET /api/live/cv-detections\` exposes the fields per classified CV box.
- \`GET /api/live/analysis\` exposes them on ranked heuristic tracks; an
  explicitly enabled \`auto_trial\` forwards the model-ordered shortlist
  to the existing isolated live trial queue.
- \`FldigiTrackTrialRequest.candidate_modes\` is optional and every entry is
  checked against the loopback-tested allowlist at the queue boundary.

These two live discovery paths are **not yet unified**: CV box output does not
automatically become a persistent live track with its own decode worker.
This slice does not replace the existing audio detector with CV when the CV
model is unavailable.

## Tests and next gates

\`\`\`bash
python -m pytest -q tests/test_identification.py tests/test_classifier_api.py tests/test_tracking.py tests/test_api.py
\`\`\`

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
its \`docs/IDENTIFIER.md\` and \`docs/SIGNAL-LAB.md\`. This first slice is
written independently in Python and does not include OnAir source code.
Review licensing before ever importing/reusing upstream implementation.
