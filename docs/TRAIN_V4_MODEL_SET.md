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
