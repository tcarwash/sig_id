# Training a new Signal ID model pair with synthetic-v4

The *model pair* consists of a class-agnostic YOLO spectrogram detector and a
hierarchical family/mode classifier. Training artifacts live outside Git;
source code and reproducible recipes belong in Git. GitHub Releases should
contain only independently approved model bundles, not private raw recordings.

## 1. Frozen training/validation/test boundary

The base CV export already has reviewed, lineage-grouped training,
validation and test images. The v4 source contains only images assigned to
**train**. Use the composition tool to add all v4 examples to training while
retaining each base validation/test image and its exact SHA-256.

```sh
export SIGID_BASE_CV_DATASET=/path/to/base/cv-dataset
export SIGID_V4_TRAIN_SUPPLEMENT=/path/to/v4/train-supplement
export SIGID_OUTPUT_ROOT=/path/to/new-experiment
mkdir -p "$SIGID_OUTPUT_ROOT"

python -m tools.compose_cv_training_dataset \
  --base "$SIGID_BASE_CV_DATASET" \
  --supplement "$SIGID_V4_TRAIN_SUPPLEMENT" \
  --output "$SIGID_OUTPUT_ROOT/cv-dataset"
```

This tool refuses to replace an existing dataset, verifies every input image
and label checksum, checks source-level split collisions, and preserves
held-out capture SHA, lineage-group assignment, pixels and labels. It creates
hard links where possible and copies files when inputs live on another
filesystem. **Do not edit any linked source/export files.** Rebuild with a
new destination when inputs change.

For the November 24 staged input, the resulting split was:

| Split | Base | After v4 |
| --- | ---: | ---: |
| Train | 819 | **1,207** |
| Validation | 159 | **159** |
| Test | 161 | **161** |
| Positive train | 340 | **520** |
| Negative train | 479 | **687** |

## 2. New hierarchical classifier

The 154 remapped Fldigi sources and 22 additional, independently generated
CW/RTTY/BPSK31 sources can be provided through `--fldigi-data-dir`. This
is a separate input channel from CV spectrogram images and does not simply
read the CV `windows.jsonl` export.

The real reviewed-source training loader normally collects reviewed windows
as new training examples; that would be dangerous after freezing a new
independent real-world holdout. **Always** pass
`--exclude-reviewed-sha-file` pointing at a versioned frozen holdout
exclusion manifest. It blocks all same-session SHA-256 captures, not merely
the exact held-out WAV, and records the exclusion count.

```sh
python -m sigid.hierarchical_training \
  --split "$SIGID_CLASSIFIER_SPLITS" \
  --data-dir "$SIGID_CLASSIFIER_DATA_DIR" \
  --ood-corpus "$SIGID_CLASSIFIER_OOD_CORPUS" \
  --reviewed-data-dir "$SIGID_REVIEWED_DATA_DIR" \
  --fldigi-data-dir "$SIGID_V4_FLIGHT_SOURCES" \
  --exclude-reviewed-sha-file "$SIGID_FROZEN_REAL_EXCLUSIONS" \
  --initial-model-path "$SIGID_PRIOR_CLASSIFIER_MANIFEST" \
  --epochs 500 \
  --feature-version passband-log-spectrum-envelope-roi-fixed-window-v4 \
  --classifier-window-seconds 2 \
  --output-manifest "$SIGID_OUTPUT_ROOT/candidate/classifier.json"
```

The classifier publishes *ranked mode suggestions* even when its
identity-acceptance calibration abstains. A reject-all calibration must **not**
be bypassed to label the candidate release-ready. Only source-independent
validation may establish family/mode acceptance and OOD safety.

## 3. Full-split spectrogram detector

On machines without a compatible GPU, `--training-imgsz` can reduce
training cost without modifying the input spectrogram or changing the
canonical inference image shape. It must be a multiple of 32 in
the range 320–2048. The default training resolution is unchanged.

```sh
python -m sigid.cv_training \
  --dataset "$SIGID_OUTPUT_ROOT/cv-dataset" \
  --base-model "$SIGID_PRIOR_CV_MODEL" \
  --output-model "$SIGID_OUTPUT_ROOT/candidate/cv-signal.pt" \
  --epochs 6 --batch 8 --workers 0 --device cpu \
  --training-imgsz 512 --train-fraction 1.0 \
  --seed 20261124
```

This trains using 100% of the assigned training split. The model trainer
selects a confidence threshold using **validation only**, and evaluates
the test split once with that fixed threshold. A shorter training resolution
is an explicit tradeoff, not evidence of comparable full-resolution quality.
Never present a short full-split run as a fully converged state-of-the-art
model without the required real tests.

### ROCm training on a Radeon RX 6700 XT (gfx1031)

The local AMD RX 6700 XT is usable with the existing ROCm 6.2
PyTorch 2.5.1 environment under `~/venv-rocm`. On this GPU, the exact
`HSA_OVERRIDE_GFX_VERSION=10.3.0` setting is needed: without it, simple
GPU matrix multiplication fails with `HIP error: invalid device function`.
With the override, forward/backward convolution, YOLO checkpoint loading
and GPU inference were verified. This is a hardware-specific compatibility
workaround and **must not be set indiscriminately on all AMD GPUs**.

The normal sig_id `.venv` uses a different Python minor release and a CUDA
build of torch, so training does **not** automatically pick up ROCm by
activating that venv. Create a dedicated training-only Python 3.11 venv,
keep the ROCm torch installed separately, and install matching ROCm
`torchvision` (0.20.1+rocm6.2), Ultralytics, SciPy, OpenCV, PIL, and
Matplotlib without replacing the ROCm torch wheels.

Run the same training command with the dedicated venv and `--device 0`:

```sh
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTHONPATH=.
# The project-specific ROCm training venv must contain compatible versions
# of torch 2.5.1+rocm6.2, torchvision 0.20.1+rocm6.2 and ultralytics.
/path/to/rocm-train-venv/bin/python -m sigid.cv_training \
  --dataset "$SIGID_OUTPUT_ROOT/cv-dataset" \
  --base-model "$SIGID_PRIOR_CV_MODEL" \
  --output-model "$SIGID_OUTPUT_ROOT/candidate-rocm/cv-signal.pt" \
  --epochs 12 --batch 8 --workers 0 --device 0 \
  --training-imgsz 640 --train-fraction 1.0 --seed 20261124
```

Do not terminate or promote a CPU run merely because GPU inference works.
Ultralytics training must complete **forward, backward, optimization,
validation and test** against the real split before the GPU model becomes a
candidate. Keep CPU and ROCm outputs in different directories.

## 4. Shadow real benchmark and public release gate

Evaluate each new detector and the prior baseline on **the same frozen**
original real test and newly frozen source-lineage shadow. Do not change
thresholds after seeing held-out results. Keep the original shadow source
hashes and pixels unchanged. Distinguish unmatched boxes in signal-positive
windows from *false positives per noise-only audio-hour*: you cannot infer
the latter from a positives-only benchmark.

An independent *real* CW/BPSK31 classifier comparison can be reproduced
using `tools/evaluate_classifier_real_holdout.py` with the frozen shadow
CV `windows.jsonl` and original recording CaptureStore. This evaluates
preselected accepted annotations only; it never fits models. Report
crop-weighted, region-weighted, and capture-weighted scores, and mention
that multiple annotations belong to just a few original receiver sessions.

Evaluate the classifier's held-out top-1/top-3 mode rankings separately
from **accepted** calibrated identities. The new core synthetic / Fldigi
results may move in opposite directions; neither metric alone proves useful
live receiver performance.

A production model bundle requires all of these:

- independent held-out real recall/precision/false-positive evidence;
- non-reject-all family/mode calibration;
- safe checkpoint metadata with no embedded local training paths;
- an explicit licensing/corresponding-source review for Ultralytics models;
- verified compatibility SHA-256 manifests and publication gates from
  `docs/MODEL_DISTRIBUTION.md`.

Create development bundles locally for smoke testing, but do **not** upload
unapproved training weights or raw captures to GitHub. An existing model
release pointer must not be updated as part of experiment training.

## 5. Separate GitHub code and model checkpoints

Push the composite dataset tool, explicit frozen real SHA exclusions,
CPU training-image-size option, regression tests, and this document through
normal PR/CI review. Model weights should be pushed only as model release
assets **after** the evaluation, metadata and licensing gates pass. The
GitHub repository itself must not contain raw WAVs, the source dataset, local
model paths or user session identifiers.

## November 24 frozen real classifier comparison

The **new 500-epoch hierarchical candidate was not a generalization winner**.
On the 26 frozen capture SHA inputs (19 contained common represented modes,
29 accepted CW/BPSK31 regions, 65 classifier crops, six original sessions),
the prior classifier achieved **55.4%** crop-weighted real top-1 and the new
candidate **38.5%**; top-3 was **93.9% for both**, and capture-averaged top-1
was **57.9% vs 49.1%**, respectively. Both accepted **0** calibrated mode
identities. The new candidate improved separate Fldigi top-1 from **9/17 to
14/17** but regressed core synthetic top-1 from **26/36 to 22/36**.

This is limited evidence: the real-mode sample is mostly CW and BPSK31, not
an independent survey of every supported modem. The counts include multiple
correlated crops and regions from only six sessions. They refute a *current*
claim of better generalization for this candidate, but do not prove that the
new dataset is worse overall. Keep the candidate in staging and retain the
existing default classifier until independent real-family calibration and
performance improve.


## Active review and new real-domain experiments

Use [ACTIVE_LEARNING_REVIEW.md](ACTIVE_LEARNING_REVIEW.md) to stage a
session-diverse, frozen-test-excluded review batch from existing real
receiver captures. The ranked proposal worklist is **not a training
corpus** until an operator accepts annotations and complete detection
coverage. For ablation-only training on existing verified real examples,
use `--real-train-repeat 4` for the detector composition and
`--reviewed-real-repeat 4` for the hierarchical classifier. Both
increase sampling weight, not independent examples. Compare against the
unaltered validation and test sets before promotion.

### November 24 real-weighted ablation — rejected, no model release

The **10-epoch ROCm CV detector trained to completion** with the original
1,207 training images plus 333 extra copies of previously human-reviewed real
training images (4x repeat, 1,540 training images total). There were only
111 unique real training sources. All original 159 validation and 161 frozen
test images were unchanged.

Using each model's own validation-selected threshold, the unchanged
held-out test reported:

| IoU 0.50 detector measure | Installed baseline | Real 4x candidate |
| --- | ---: | ---: |
| Overall precision | 85.7% (66 TP, 11 FP) | 74.0% (54 TP, 19 FP) |
| Overall recall | 60.0% | 49.1% |
| Live receiver recall | 25.0% (12/48) | 14.6% (7/48) |
| Multiple-signal recall | 41.2% (28/68) | 30.9% (21/68) |
| Test false detections per audio hour | 24.6 | 42.5 |

The candidate **failed five of seven conservative release checks**:
independent positive live test lineage count, live recall, overall recall,
false detection rate, and multiple-signal recall. This test false-detection
measure covers the reviewed test windows and should **not** be confused with
a separately measured pure noise-only false-positive rate. The comparison
still lacks five independent positive held-out live lineages.

The companion 4x real-weighted hierarchical classifier likewise did not
beat production on frozen real evidence: its crop-weighted top-1 was
**46.2% vs 55.4%**, top-3 **73.8% vs 93.8%**, and calibrated acceptance was
zero for both. Only six original sessions with CW/BPSK31 accepted regions
were represented. Both experimental checkpoints remain local and are not
promoted or published.

**Consequence:** repeating the same verified real examples is not a
substitute for new independent annotations or clean negatives. Retain the
installed checkpoints and frozen splits. The next model fitting cycle must
use newly confirmed examples, and must compare the same unchanged
held-out real/negative tests without tuning thresholds on that test.

The receiver review workbench was instead improved: an optional read-only
production-CV-versus-spectral-track screen considered 120 eligible clips,
with no failures and no accepted pseudo-labels. The 100 prioritized windows
now comprise 10 track-only proposals, 20 CV-only, 6 spatial disagreements,
8 agreements, and 56 possible (unverified) negative controls, across
12 source sessions. Detailed scores remain local. See
[ACTIVE_LEARNING_REVIEW.md](ACTIVE_LEARNING_REVIEW.md) for the procedure.
