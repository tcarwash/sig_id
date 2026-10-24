# Optional computer-vision signal detection plan

## Goal

Add an optional computer-vision (CV) detector that scans full spectrogram windows, proposes signal time/frequency boxes, and passes reviewed detections through the existing classifier and decoder workflow. The existing classifier and manual selection workflow remain the default and continue to work when CV is not installed or configured.

The detector answers **where is signal activity?** The classifier and optional embedding retrieval answer **what might it be?** Decoder selection and execution remain a separate step tied to the exact audio range.

## Current project foundations

- Full-capture spectrograms and a live waterfall are available.
- The live audio analysis buffer is bounded to 10 seconds; the browser waterfall can display 30 seconds.
- Manual regions use sample-indexed time bounds and frequency bounds.
- Scene coverage reviews, annotation revisions, and geometry corrections provide a basis for reviewed training labels.
- A trained modulation-method classifier can propose labels for saved regions and abstain; proposals require review.
- Saved regions can be replayed through the selected decoder using their exact audio sample bounds.
- The current classifier is optional through `SIG_ID_MODEL_PATH`.

## Architecture

```text
10-second spectrogram window
  -> optional signal detector
  -> provisional time/frequency boxes
  -> inspect or correct boxes
  -> exact sample-indexed audio region
  -> existing classifier
       -> confident proposal: review / choose decoder
       -> uncertain or abstained: embedding neighbors / unknown label
  -> explicitly selected or supported decoder
  -> saved decode result linked to region and source
```

The initial detector should be **class-agnostic**: it finds signal-shaped regions but does not claim a protocol identity. This keeps detection focused and allows newly encountered signal types to enter the existing unknown/review workflow without adding YOLO classes for every catalog entry.

## Implementation phases

The first implementation slice is in place: optional class-agnostic YOLO inference
over overlapping 10-second spectrogram windows, confidence-ranked cross-window
duplicate suppression with supporting-window lineage, capture-level proposals and overlays, human
review into existing regions, lineage-aware dataset export and training commands,
evaluation on held-out capture groups, optional ROCm Compose support, and an
embedding index for reviewed regions. The base classifier/manual workflow remains
available without the CV extra. The implementation commits are `37cd4f6` (detector
and ROCm container), `d5dfc0b` (proposal review evidence), and `a421f39` (reviewed
region embedding retrieval), followed by evaluation, packaging, and waterfall
geometry refinements.

The first slice reuses the existing selection and geometry correction tools to
inspect and adjust proposed boxes. A held-out downstream evaluator now measures
detection-to-classifier behavior and limited oracle-assisted bit recovery on direct
synthetic captures. Live YOLO scanning, independent real-capture quality evidence,
classifier-selected decoding, and protocol interoperability validation remain
follow-up work.

### 1. Define the detection contract

- Use 10-second spectrogram windows as the initial inference unit.
- Specify and version the spectrogram preprocessing: sample rate, FFT/window and hop, frequency range, intensity scaling, image dimensions, and time/frequency orientation.
- Preserve the native 1024x256 spectrogram shape through rectangular YOLO training batches and inference to avoid square-padding mismatch.
- Return boxes in normalized image coordinates plus confidence and model version. Convert them to authoritative sample and frequency bounds using the window metadata.
- Inference responses now preserve normalized XYXY coordinates, image dimensions, and a content-hash model version alongside converted sample/frequency bounds. Saving a reviewed detector proposal retains that mapping evidence.
- Keep detector proposals distinct from reviewed annotations and classifier predictions.
- Define behavior for clipped boxes at window edges, duplicate boxes across adjacent windows, overlapping signals, and empty/no-signal windows.

### 2. Build the training-data path

- Use existing accepted signal-region annotations as detector boxes and fresh,
  complete scene reviews as evidence for no-signal windows. Explicit controlled
  noise-only fixtures export as negative windows and are excluded from signal
  embedding references. Users can draw and correct regions with the existing
  selection and geometry-editing controls; those reviewed regions feed the CV
  export path.
- Store box geometry in sample and frequency coordinates, not display pixels. Preserve capture/session lineage and annotation revision history.
- A capture-wide review queue now steps through the exporter's non-overlapping 10-second windows over the detector's 0–3 kHz band. It skips fresh complete reviews, revisits missing or stale coverage, advances after each saved complete/partial review, and leaves uncertain windows out of export. Accepted signal boxes remain separate annotations.
- The review UI and training guide now make the export consequence explicit: a complete window with no accepted boxes becomes a negative example. Reviewers should save all signal boxes first, mark uncertain coverage partial, and rerun the queue after label or geometry edits stale overlapping reviews.
- Deterministic synthetic imports now derive signal-frequency geometry from their validated generator manifests. The CV exporter can recover missing geometry on older imports only when the sidecar hash matches the stored capture provenance.
- Synthetic scene imports now transform frequency bounds with applied frequency offset and drift, including folding at Nyquist, so scene-derived detector boxes match the rendered signal.
- A CV-only deterministic source-corpus command now adds OOK, 4-ASK and PPM captures with validated manifests and accepted regions. It feeds the existing scene builder without changing the six known-modulation classifier classes; its generated artifacts remain isolated from the user's live data when following the documented `--data-dir` recipe.
- Export the reviewed data to the detector's training format. Generate images and labels from the same versioned spectrogram preprocessing used at inference.
- Split by source capture or recording session before generating windows or augmentations. Keep all derived windows from a source in one split to avoid leakage, and stratify reviewed positive, no-signal, mixed, and unreviewed groups across partitions when possible so validation and test cover both detection classes. Export reserves at least three lineages for each of validation and test, and training rejects smaller held-out splits.
- Include representative real captures as they become available, alongside synthetic scenes. Track their source and review status separately.

### 3. Train and evaluate offline

- Add a reproducible training command, configuration, and versioned model artifact.
- Start with class-agnostic boxes. Evaluate class-labeled detection only if measurements show a clear benefit.
- Report box precision/recall at defined overlap thresholds, temporal/frequency localization error, and false detections per unit of audio. The trainer evaluates the held-out split at the same rectangular inference shape, reports center/bounds error in seconds and hertz for IoU 0.50 matches, sweeps confidence thresholds on validation, selects the default operating point from validation IoU 0.50 F1, then reports test metrics at that selected threshold.
- Break results down by signal strength/SNR, signal type, overlap, drift, noise, and boundary truncation. The exporter records available source/scene conditions, and the trainer reports held-out IoU 0.50 detection metrics by source kind, overlap, tagged noise/SNR, drift, and boundary truncation, plus signal-type box recall. Per-type false-positive attribution is omitted because this class-agnostic model does not predict signal identity. Real-capture SNR and other unrecorded conditions remain unavailable rather than inferred.
- Compare against the existing heuristic detector where both apply. The held-out flow evaluator now reports class-agnostic box precision/recall for both YOLO and the spectral peak-clustering baseline on identical reviewed windows.
- Separately evaluate the full downstream flow: detection recall, classifier coverage/accuracy on detected regions, unknown/abstention behavior, and successful decoder outcomes. Detector confidence alone must not establish identity.
- A held-out detector-to-classifier evaluator records detector matches, classifier coverage/accuracy, abstentions, and accepted known labels on unmatched proposals. It also reports limited bit recovery for matched YOLO crops from direct synthetic captures, using oracle demodulator parameters and resolving BPSK polarity/QPSK rotation against the known payload. This does not measure classifier-driven mode selection, scene mixtures, real recordings, protocol framing, or interoperability.
- An isolated 140-group synthetic bootstrap dataset completed export, one-epoch ROCm training, validation-selected thresholding, held-out evaluation, and runtime artifact loading. The smoke model detected 0/20 validation and 0/18 held-out signal boxes, so this verifies the training path only and is not a useful detector. See [`reports/cv-synthetic-bootstrap-2026-10-23.json`](reports/cv-synthetic-bootstrap-2026-10-23.json); representative real-capture training and quality evidence remain open.
- A ten-epoch follow-up on the stratified version of that corpus reached 15/18 held-out box recall at IoU 0.50 but also produced 13 false detections across 3 reviewed negative windows (about 698/hour for the short synthetic clips). This is evidence that the trainer learns controlled bursts, but the false-alarm rate remains too high and does not establish useful or real-capture performance; the report records the model/runtime hashes and metrics.
- A new burst-in-noise synthetic follow-up used 120 independently generated signal captures to create 112 lineage-held-out scenes, including overlapping bursts, over fresh noise backgrounds. A 10-epoch RX 6700 XT ROCm run reached 27/36 held-out box recall at IoU 0.50, with 12 false detections over 17 negative 10-second windows (about 121/hour). It learns some signal families better than others (zero recall for held-out BPSK and QPSK in this run), and is not suitable for unattended detection. This remains controlled synthetic evidence and used host Ultralytics 8.3.180 rather than the project pin 8.4.184. See [`reports/cv-synthetic-bootstrap-2026-10-23.json`](reports/cv-synthetic-bootstrap-2026-10-23.json).
- The exact pinned ROCm container also completed 10-epoch training and project-wrapper inference on the same synthetic dataset: Ultralytics 8.4.184, PyTorch 2.13.0+rocm10.1.0, RX 6700 XT. Held-out IoU 0.50 recall was 19/36 (52.8%), with 16 false boxes on signal-containing windows (about 161/hour), and zero on 17 pure-noise windows. It missed all held-out BPSK and QPSK boxes. A held-out positive spectrogram loaded through `YoloSignalDetector` returned a box on ROCm. MIOpen emitted optional gfx1031 database/kernel warnings, while training and inference completed. This verifies the pinned container path; it does not establish useful detector quality. The artifact and run remain under `/tmp`; details and hashes are in the report.
- A 100-epoch run in the same pinned ROCm container and on the same lineage split improved held-out IoU 0.50 recall to 29/36 (80.6%) with 29/31 precision (93.5%). At the validation-selected confidence 0.50 it produced 2 false boxes over 357.28 seconds of held-out audio (20.15/hour), with zero false boxes in 17 noise-only windows. It detected 2/6 BPSK and 1/2 QPSK boxes, so those weaker families remain a focus. Mean detector latency was 12.47 ms per window. This suggests the 10-epoch run was undertrained, but the small synthetic test and lack of reviewed real captures still prevent any production-quality claim. Model SHA256, thresholds, condition breakdowns, and runtime measurements are in the report; weights and run logs remain under `/tmp`.
- A larger corpus with 120 original FSK/PSK sources plus 20 each of OOK, 4-ASK and PPM yielded 136 lineage groups and 590 captures with 375 positive and 215 negative reviewed windows. A 100-epoch run in the pinned ROCm image matched 63/69 held-out boxes at IoU 0.50 (91.3% recall, 98.4% precision), with one false box over 710.32 seconds (5.07/hour) and no false boxes in 33 pure-noise windows. Held-out recall was 11/12 OOK, 9/10 4-ASK and 2/2 PPM; QPSK remained lower at 4/6. This shows the new CV-only source path exports and trains across the added visual families without changing classifier classes. It remains synthetic-only; the noise sample is too short for a reliable low false-alarm claim. The expanded dataset, model hash and run metrics are in the report; weights and logs remain under `/tmp`.
- An exploratory ROCm scan of five locally saved recordings (480 seconds total) produced 11 proposals against eight existing accepted regions. None reached IoU 0.50; the best overlaps were below the threshold. Those labels cover only selected regions, so this is not a complete recall or false-alarm evaluation. The result does show that the synthetic-trained detector needs independently reviewed real captures before live inference or unattended use. Aggregate evidence, model hash, and limitations are in [`reports/cv-real-capture-exploration-2026-10-23.json`](reports/cv-real-capture-exploration-2026-10-23.json); capture identities, paths, and audio remain local.

### 4. Add optional packaging and inference

- Keep CV libraries out of base runtime dependencies. Add an explicit optional install extra for inference; isolate training dependencies in a separate extra or training environment if that avoids burdening deployment.
- Load the detector only when CV is enabled and a model artifact is configured. Report a clear unavailable/disabled state when dependencies or weights are absent.
- Optional Compose inference can use the CPU-capable base image or an AMD ROCm/PyTorch image with explicit `/dev/kfd` and `/dev/dri` pass-through.
- Container paths were built and exercised locally: the default image omits Ultralytics, the optional CPU image uses CPU-only PyTorch, and the pinned ROCm image completed an Ultralytics model forward pass on the RX 6700 XT. The ROCm test used the host's render GID 110; Compose and `.env.example` use that default while still allowing host-specific overrides. MIOpen emitted missing optional gfx1031 kernel/database warnings during the successful forward pass.
- Keep inference failures isolated so the classifier, manual selection, capture, and decoding workflows remain usable.
- Begin with batch detection on completed captures. Training evaluation now records model size, per-window latency/throughput, process RSS, and accelerator memory when exposed by PyTorch. Real-capture and hardware measurements are still required before enabling periodic live inference.
- Keep model weights and writable CV data in the configured persistent data directory for Compose deployments; do not bake user-trained weights into the base image.

### 5. Connect detections to the existing workflow

- Render boxes as provisional overlays on the waterfall and let users accept, reject, or adjust them. Users can dismiss a proposal from the current scan; this scan-local dismissal is not a persisted negative label or detector training example. Persistent reviewed negative feedback remains future work.
- Convert accepted boxes into the existing exact time/frequency selection contract and fetch/replay the matching audio range.
- Run the existing classifier on each accepted region. Keep classifier proposals reviewable and preserve its abstention behavior.
- When the classifier is uncertain, offer embedding-based nearest examples and an unknown label. The initial UI exposes candidate retrieval for an active selection and points to it after classifier abstention; retrieval remains an explicit user action. Treat embeddings as candidate retrieval, not proof of identity.
- Offer decoder choices based on supported classifier results, but keep decoder execution explicit initially. Saved regions now retain the server-generated classifier proposal (including abstentions, model hash, and exact audio sample range); saved-capture Fldigi replays snapshot that evidence and retain a validated link to the active region when replay bounds match its saved geometry.
- Deduplicate boxes from overlapping windows and track a detection across windows without losing its source-window lineage. Batch capture scanning now uses a 5-second stride, suppresses cross-window boxes at 0.30 2D IoU, and retains supporting window metadata on the surviving proposal; temporal tracking beyond one completed-capture scan remains future work.

### 6. Opt-in rollout and acceptance gates

- Add an explicit CV enable/configuration setting; default it off.
- Keep manual selection available and visually distinguish detector proposals, reviewed boxes, classifier proposals, and confirmed user labels.
- Enable completed-capture batch inference first. Add live inference only after resource use, duplicate handling, and latency are acceptable.
- Before calling the feature useful, demonstrate held-out detection quality on captures split by source/session, then validate the detector-to-classifier path and audio-range-to-decoder mapping independently.
- Preserve the boundary between synthetic validation, independent recording validation, hardware/RF validation, and decoder interoperability evidence.

## Initial acceptance criteria

- The base application installs and runs without CV dependencies or a CV model.
- ROCm is an opt-in Compose override with device pass-through; actual GPU support still depends on the host driver and GPU compatibility with the chosen AMD image.
- Similarity search uses a model-hash-bound index of reviewed regions and clearly labels results as candidates rather than identifications.
- With CV enabled, a 10-second window produces boxes that map back to the correct sample and frequency ranges.
- Users can correct or reject boxes, and reviewed labels export reproducibly with preprocessing and source metadata.
- Evaluation uses source/session-held-out data and reports detection misses and false detections, including empty/no-signal windows.
- A detector proposal can enter the existing classifier flow; classifier abstention leads to candidate retrieval or unknown handling rather than a forced known identity.
- An explicitly chosen decoder receives the exact audio region associated with the box, and its result remains linked to that region.
- Disabling CV or encountering a CV inference error does not interrupt existing manual selection, classifier, or decoding behavior.

## Open decisions for implementation

- Which spectrogram scaling and normalization best preserve weak and strong signals across recordings?
- Should inference run only after capture completion initially, or also on a user-requested live window?
- What minimum real-capture coverage is required before enabling automatic decoder suggestions?
- Should the first detector artifact run through a PyTorch-based runtime or be exported to a lighter inference format after training?
