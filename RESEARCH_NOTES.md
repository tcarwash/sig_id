# Signal Identification Research Notes

Living project notebook for the eventual methods/results write-up. These notes
record design decisions, data provenance, measured outcomes, failures, and open
questions. They are not a claim that the detector is ready for unattended use.

**Last updated:** 2026-10-25  
**Scope:** CV signal-region detection and its relationship to the two-second
classifier.  
**Update practice:** append dated findings as runs finish; preserve poor or
inconclusive results alongside successful ones. Label statements as measured,
observed, or hypothesized, and keep synthetic, recorded-audio, live, and RF
evidence distinct.

## Research objective

The CV model proposes **where** signal activity appears in a spectrogram. It is
class-agnostic; it does not identify a modulation or protocol. The existing
classifier addresses **what** a selected signal may be, using two-second crops.
CV training and inference use the full ten-second window because labels are
time/frequency bounding boxes. A CV proposal remains provisional until reviewed.

The project is investigating whether a detector trained on controlled synthetic
signals, realistic recorded-noise backgrounds, and fully reviewed real captures
can find weak signals with a tolerable false-alarm rate on new recordings.

## Current method

### CV window and image

- Each CV example is a complete, non-overlapping ten-second window.
- The detector band is 0–3,000 Hz. Training and inference share
  `sig-id-live-waterfall-v2` preprocessing: 1024 × 640 landscape RGB image,
  4096-sample FFT, 512-sample hop, live-waterfall dB scaling, and newest time at
  the top.
- Boxes are stored from sample-time and frequency bounds, then converted to
  normalized image coordinates. Accepted signal regions generate positive
  boxes; detector-negative examples have no boxes.
- Full-window coverage is required. A short final remainder, an unreviewed
  window, or a window with unsupported accepted geometry is not exported.

### Real recordings and noise

- A real recording with no annotation is not automatically treated as noise.
  For a real pure-negative example, the source must be a saved ten-second review
  sample with fresh, complete human detection coverage and no active, unresolved
  signal regions. Explicitly rejected regions can represent reviewed false
  alarms.
- A segment cut from a longer recording does not inherit a sibling sample's
  labels or review. The saved ten-second sample is the canonical reviewed unit.
- Fully reviewed signal annotations can provide positive boxes. Their labels
  belong to their own capture/window; they are not copied to another recording
  just because it contains the same audio interval.
- The scene builder can use five disjoint two-second segments from eligible
  reviewed ten-second recordings to make a ten-second recorded-noise background.
  Source segments and component captures are retained in provenance. A raw noise
  source is eligible as a direct CV negative only under the same ten-second
  human-review rule.
- Classifier training has a separate noise path: a fully reviewed real noise
  sample can be added as an explicit noise-only training row unless its source
  metadata sets `exclude_training`. Keep this separate from the CV scene
  background use when deciding whether a recording is allowed into training.
- Recorded backgrounds are meant to add realistic noise texture to synthetic
  signal scenes. Uniform synthetic noise remains a different source category.
  Unreviewed real windows must never be promoted to negatives by assumption.

### Synthetic scenes and leakage control

- Signal inputs are accepted, manifest-backed synthetic captures with usable
  frequency bounds. One source is assigned to at most one generated scene.
- Each ten-second scene has independently seeded background noise. Individual
  signal levels are targeted between **−40 and −15 dB SNR** relative to aligned
  in-band noise during the active interval; the builder records target and
  measured values.
- `--overlap-rate` is a legacy option name: it controls the fraction of scenes
  containing two signals. The intended two-signal examples have separate time
  intervals so their boxes do not overlap. The placement code now places both
  intervals as a pair and leaves at least 0.5 seconds between newly generated
  signals to reduce spectrogram-window smearing.
- CV splits are made by source lineage, not by individual image. Derived scene
  dependencies and recording sessions are grouped so examples sharing a signal
  lineage cannot be split across train, validation, and test. This can reduce
  the number of independent groups substantially even when there are many image
  files.
- Rebuild from source captures when combining datasets. Do not concatenate
  exported YOLO folders without deduplicating capture hashes and reassigning
  lineages; duplicate or related captures across partitions can inflate
  validation results.

## Findings and run ledger

### Earlier controlled-synthetic experiments

The maintained CV plan and `reports/cv-synthetic-bootstrap-2026-10-23.json`
record several stages. A ten-epoch burst-in-noise run found 27/36 held-out boxes
at IoU 0.50 but produced 12 false detections on 17 negative ten-second windows.
A pinned 100-epoch run on a broader 180-source corpus later found 63/69 held-out
boxes at IoU 0.50 with one false box over 710.32 seconds of held-out audio. These
are controlled-synthetic results only. The small negative-audio exposure does
not establish a reliable false-alarm rate on real recordings.

### V4 export and training

`data/cv-seed/export_v4` contains 963 images: 302 positive and 631 negative.
The split is 654 train, 149 validation, and 130 test images across 554 recorded
lineage groups. The report lists 915 synthetic and 18 live images. Thus, V4 has
many more independent groups and validation images than the newer export below,
but little live data relative to synthetic data.

The V4 training run used that export, `yolo26n.pt`, 100 epochs, batch 8, seed
4192526, and device 0. The saved model is `data/cv-seed/signal-v4.pt`. Its
Ultralytics validation mAP@0.50 peaked at 0.909 (epoch 82); final mAP@0.50 was
0.878. The initial custom held-out evaluation was invalidated by the RGB/BGR
mismatch described below and incorrectly selected confidence 0.75.

Investigation found an RGB/BGR inference mismatch. Waterfall and PIL images are
RGB, while Ultralytics' numpy-array prediction API expects BGR and converts to
RGB internally. Both the custom confidence evaluator and live detector passed
RGB arrays directly, so predictions and threshold calibration used swapped red
and blue channels. The detector and evaluator now convert RGB arrays to
contiguous BGR before prediction. Re-evaluation on the ROCm GPU with corrected
code selected a 0.50 operating threshold from validation. At IoU 0.50,
validation had 54 TP, 8 FP, and 10 FN (precision 0.871, recall 0.844);
held-out test had 44 TP, 5 FP, and 8 FN (precision 0.898, recall 0.846). At the
former 0.75 threshold, validation recall was 0.703 and test recall was 0.731.
The model sidecar was updated to threshold 0.50 and its validation/test
summaries were replaced with the corrected evaluations. The trainer's built-in
metrics load files through its dataset pipeline and are not affected by this
specific numpy inference path.

### Expanded export that produced poor validation results

The report at `data/cv-runs/20261025T155333Z/dataset/export-report.json` has 867
images: 460 positive and 407 negative. It has 852 train images, only 7 validation
images, and 8 test images, across 31 lineages. The report lists 818 synthetic and
49 live images. Training completed 100 epochs. Its best validation mAP@0.50 was
0.167 at epoch 68; final mAP@0.50 was 0.069, with final recall 0.111. The live
review screenshot also showed many provisional candidate boxes on apparent
noise. This is a poor result, not evidence that the detector improved.

The export at `data/cv-runs/20261025T172552Z/dataset` has the same report counts
as the `20261025T155333Z` export. Treat these as likely repeated data until
capture hashes and image manifests establish otherwise; do not count their
matching images as added independent training evidence.

**Interpretation:** raw image count is not enough. The 867-image export had only
31 groups and 7 validation images. Its failure could involve limited independent
validation coverage, synthetic-to-real mismatch, label quality, or training and
inference differences. These are hypotheses; the current evidence does not
isolate the cause.

### New scene build and export

The run at `data/cv-runs/20261025T184754Z` contains 123 mixed scenes: 92 with one
signal and 31 with two signals, plus 24 noise-only examples. The builder first
stopped at scene 93 because its original placement algorithm could not fit a
second signal around a fixed first-signal position. The algorithm was corrected
to place pairs jointly, and the existing indexed build was resumed. The first 92
already saved scenes were non-overlapping when the earlier algorithm succeeded,
but may have less than the new 0.5-second guard gap; newly generated pairs use
the guard gap. The export completed with 1,113 images: 583 positive and 530
negative (1,064 synthetic and 49 live), split into 1,098 train, 7 validation,
and 8 test images across 31 lineages (21 train, 5 validation, 5 test). Each
held-out split clears the current three-lineage minimum, but has few images and
should be interpreted cautiously. No model has yet been trained from this run.
The direct ROCm training command is recorded in `TRAINING.md`.

## Current conclusions

1. The dataset must preserve the distinction between a real reviewed empty
   window, a synthetic-noise fixture, and real noise used as a synthetic-scene
   background.
2. More training images helped only if they also add independent, correctly
   labeled lineages and realistic conditions. Validation and test splits need
   enough independent reviewed recordings to support meaningful conclusions.
3. The poor latest model should not be promoted based on a high image count or
   training completion. Its validation metrics and visual false candidates are
   inconsistent with useful detection quality.
4. Current evidence is a mix of controlled synthetic results and limited real
   examples. We do not yet have a sufficiently broad independent real-recording
   benchmark for a production-quality claim.

## Open questions and next measurements

- Compare V4's corrected custom held-out results with its built-in Ultralytics
  metrics, and verify its live overlays on representative real recordings.
- Finish the resumed scene export and inspect its report: counts by source,
  positives/negatives per split, independent lineage counts, and source overlap.
- Review validation and test images/labels visually, especially faint signals
  and noise-only examples. Check model predictions on the same examples.
- Compare capture hashes, source-component lineages, and preprocessing before
  combining V4 with newer datasets. Re-export one merged source pool and split
  by lineage.
- Evaluate false detections per hour on substantially more independent,
  human-reviewed real noise recordings; report synthetic and live results
  separately.
- Confirm live preprocessing and the training export remain identical, then
  assess confidence calibration and choose an operating point using validation
  only.
- Investigate whether the current weak model reflects data composition, label
  quality, thresholding, model configuration, or a training/inference mismatch.

## Decoder-assisted mode identification (2026-10-25)

- Added an explicit auto-identify action for an exact selected live or saved
  audio interval. It uses a current classifier family/method proposal to
  prioritize compatible Fldigi modes, then tests up to six catalog-supported
  receive modes in isolated Fldigi replays. Without a usable proposal it falls
  back to a short list of established PSK, RTTY, THOR, and MFSK candidates.
- The first isolated replay enables Fldigi receive-side RSID, which can detect
  the mode identifier transmitted by a remote station (often called TXID at
  the transmitting end). It scans the selected audio passband. RSID may be
  absent from a clip; in that case the workflow compares candidate text output
  using a lightweight readability score and displays the best result for human
  review.
- The isolated profile enables `RECEIVERSID` only for this action and forces
  `TRANSMITRSID` off. The user's running Fldigi profile and radio are not
  changed. Candidate mode scores are tentative evidence and never write a
  signal label.
- This is an implemented workflow, not yet an evaluated decoder benchmark.
  Measure mode ranking and RSID recovery on known, RF-free loopback clips and
  representative saved recordings before treating ranking quality as reliable.

## Independent live CV and classifier controls (2026-10-25)

- Live settings now have separate CV-box and classifier-suggestion toggles;
  both default on. With both on, CV proposes the displayed bounds and the
  classifier optionally adds a method/family suggestion to each box. Turning
  off classifier scoring leaves CV geometry visible and does not affect CV
  detections.
- With CV off and classifier scoring on, the app displays the existing
  heuristic track candidates and their classifier proposals. These bounds are
  explicitly provisional and come from the spectral heuristic, not the
  classifier. Turning both off pauses both candidate paths.
- Classifier scoring is also independently switchable for CV detections. The
  existing opt-in automatic Fldigi trial still uses persistent heuristic
  tracks when enabled. Manual and saved-capture classification are unaffected.

## Live review capture modes (2026-10-25)

- The live input now has three modes. **Record** preserves the full session and
  continues creating ten-second review samples. **Scan only** keeps audio in
  the bounded in-memory live buffer so the app can run its live CV and
  classifier views, then discards it when stopped. **Unattended** does not
  retain the full session: it evaluates successive ten-second windows with the
  active CV detector and saves only windows with CV proposals plus a
  configurable periodic sample when CV has no proposal.
- Unattended clips are marked as pending review queue items. Detection counts,
  confidence summaries, and classifier method proposals are stored as triage
  metadata; classifier labels are not treated as verified annotations. Periodic
  no-detection clips are only noise candidates. A person must review the full
  ten-second window before it can qualify as a real noise-only training
  example.
- Live and saved-capture CV boxes now receive classifier proposals from
  centered two-second crops inside the detected interval and frequency bounds.
  The highest-confidence 32 boxes are scored; results and abstentions are shown
  with each box, remain provisional, and do not create saved annotations.
- Live CV detections are retained for the active capture session and merged
  across overlapping inference polls using configurable time/frequency overlap
  (10–80%, default 30%, persisted in browser settings). Same-frequency boxes
  can also bridge gaps up to three seconds, matching the live poll cadence. A
  repeat expands the retained box to the union of observed time/frequency
  bounds and can raise its detector score or improve its classifier proposal.
  Boxes scroll out when their audio leaves the live waterfall history.
  Saved-capture scan behavior is unchanged. Live behavior still needs
  browser/device validation.
- **Validation boundary:** code compilation, JavaScript syntax, and diff
  whitespace checks pass. No physical audio device, live model inference, or
  multi-hour unattended capture has been exercised in this environment.

## Incremental in-app model updates (2026-10-25)

- The app now has a paired **Train both models from new reviews** workflow. It
  requires both active checkpoints and a fully reviewed, labeled live capture
  not represented in the active classifier's reviewed-training manifest.
- The classifier update replays its original synthetic train/validation split
  plus reviewed real crops, initializing existing family and mode outputs from
  the active checkpoint. Newly introduced taxonomy outputs start with fresh
  weights. Calibration continues to use the held-out split and OOD corpus when
  its source can be resolved.
- The CV update exports the configured reviewed source stores plus the app data,
  then fine-tunes from the active detector checkpoint. Existing capture groups
  are re-exported and remain lineage-split by the standard exporter.
- The shared classifier trainer now forms its first stage from modulation
  family where the method is recognizable (for example, 8-GFSK joins MFSK;
  BPSK31 joins PSK), while preserving the catalog family separately for label
  integrity. Its second-stage expert predicts trained mode IDs and carries
  display name, protocol, and variant metadata. Inference can show the
  family-to-mode candidate path even when validation thresholds require it to
  abstain. This does not add an unreviewed mode such as FT8 to a model: that
  mode must appear in accepted training labels.
- Both artifacts are staged as a timestamped release and validated before one
  active-model JSON pointer is atomically replaced. A failed model step leaves
  the active pair untouched. The in-app action complements the full pipeline,
  which continues to build synthetic/collected-data base models.
- A failed update exposed a metadata shape mismatch: `reviewed_real_training`
  reports an integer example count, while row identities are stored in the
  hashed `reviewed-training-examples.json` sidecar. The job now verifies and
  reads that sidecar to distinguish newly reviewed examples before CV training.
- The training status API now reports overall progress and epoch callbacks for
  the classifier and CV stages. The app displays a percentage bar alongside
  the current stage.
- Verification so far is static Python/JavaScript syntax and diff checks; the
  corrected training job, progress display, and model quality have not yet been
  exercised.

## Source files and evidence

- Detector and preprocessing: [`sigid/cv_detector.py`](sigid/cv_detector.py)
- Scene generation and recorded-noise provenance: [`sigid/cv_scene_corpus.py`](sigid/cv_scene_corpus.py)
- Review requirements, export rules, and reports: [`sigid/cv_dataset.py`](sigid/cv_dataset.py)
- Lineage-based splitting: [`sigid/splits.py`](sigid/splits.py)
- Training and evaluation: [`sigid/cv_training.py`](sigid/cv_training.py)
- Workflow and historical CV results: [`CV_FLOW_PLAN.md`](CV_FLOW_PLAN.md)
- Reproduction commands: [`TRAINING.md`](TRAINING.md)
