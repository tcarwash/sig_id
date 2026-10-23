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
over 10-second spectrogram windows, capture-level proposals and overlays, human
review into existing regions, lineage-aware dataset export and training commands,
evaluation on held-out capture groups, optional ROCm Compose support, and an
embedding index for reviewed regions. The base classifier/manual workflow remains
available without the CV extra. The implementation commits are `37cd4f6` (detector
and ROCm container), `d5dfc0b` (proposal review evidence), and `a421f39` (reviewed
region embedding retrieval).

The first slice does not yet provide a dedicated arbitrary box editor, live YOLO
scanning, overlapping-window tracking, real-capture quality evidence, or
end-to-end detector-to-decoder evaluation. Those remain explicit follow-up work.

### 1. Define the detection contract

- Use 10-second spectrogram windows as the initial inference unit.
- Specify and version the spectrogram preprocessing: sample rate, FFT/window and hop, frequency range, intensity scaling, image dimensions, and time/frequency orientation.
- Preserve the native 1024x256 spectrogram shape through rectangular YOLO training batches and inference to avoid square-padding mismatch.
- Return boxes in normalized image coordinates plus confidence and model version. Convert them to authoritative sample and frequency bounds using the window metadata.
- Keep detector proposals distinct from reviewed annotations and classifier predictions.
- Define behavior for clipped boxes at window edges, duplicate boxes across adjacent windows, overlapping signals, and empty/no-signal windows.

### 2. Build the training-data path

- Use existing accepted signal-region annotations as detector boxes and fresh,
  complete scene reviews as evidence for no-signal windows. Explicit controlled
  noise-only fixtures export as negative windows and are excluded from signal
  embedding references. A dedicated arbitrary
  box editor and annotation-focused revision workflow remain follow-up work.
- Store box geometry in sample and frequency coordinates, not display pixels. Preserve capture/session lineage and annotation revision history.
- Export the reviewed data to the detector's training format. Generate images and labels from the same versioned spectrogram preprocessing used at inference.
- Split by source capture or recording session before generating windows or augmentations. Keep all derived windows from a source in one split to avoid leakage.
- Include representative real captures as they become available, alongside synthetic scenes. Track their source and review status separately.

### 3. Train and evaluate offline

- Add a reproducible training command, configuration, and versioned model artifact.
- Start with class-agnostic boxes. Evaluate class-labeled detection only if measurements show a clear benefit.
- Report box precision/recall at defined overlap thresholds, temporal/frequency localization error, and false detections per unit of audio. The trainer now evaluates the held-out split at the same rectangular inference shape and reports center/bounds error in seconds and hertz for IoU 0.50 matches.
- Break results down by signal strength/SNR, signal type, overlap, drift, noise, and boundary truncation. The initial evaluator reports held-out box metrics and false detections per audio hour; condition-level breakdowns remain follow-up work.
- Compare against the existing heuristic detector where both apply.
- Separately evaluate the full downstream flow: detection recall, classifier coverage/accuracy on detected regions, unknown/abstention behavior, and successful decoder outcomes. Detector confidence alone must not establish identity.

### 4. Add optional packaging and inference

- Keep CV libraries out of base runtime dependencies. Add an explicit optional install extra for inference; isolate training dependencies in a separate extra or training environment if that avoids burdening deployment.
- Load the detector only when CV is enabled and a model artifact is configured. Report a clear unavailable/disabled state when dependencies or weights are absent.
- Optional Compose inference can use the CPU-capable base image or an AMD ROCm/PyTorch image with explicit `/dev/kfd` and `/dev/dri` pass-through.
- Keep inference failures isolated so the classifier, manual selection, capture, and decoding workflows remain usable.
- Begin with batch detection on completed captures. Measure model size, memory, CPU/GPU use, and per-window latency before enabling periodic live inference.
- Keep model weights and writable CV data in the configured persistent data directory for Compose deployments; do not bake user-trained weights into the base image.

### 5. Connect detections to the existing workflow

- Render boxes as provisional overlays on the waterfall and let users accept, reject, or adjust them.
- Convert accepted boxes into the existing exact time/frequency selection contract and fetch/replay the matching audio range.
- Run the existing classifier on each accepted region. Keep classifier proposals reviewable and preserve its abstention behavior.
- When the classifier is uncertain, offer embedding-based nearest examples and an unknown label. The initial UI exposes candidate retrieval for an active selection and points to it after classifier abstention; retrieval remains an explicit user action. Treat embeddings as candidate retrieval, not proof of identity.
- Offer decoder choices based on supported classifier results, but keep decoder execution explicit initially. Store the exact region, classifier evidence, selected decoder, and decode output together.
- Deduplicate boxes from overlapping windows and track a detection across windows without losing its source-window lineage.

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
