# Training guide

This guide covers the local synthetic-to-CV training path. It keeps generated
Fldigi transmissions, ten-second CV scenes, exported spectrograms, and model
artifacts in separate, reproducible locations.

## 1. Generate native Fldigi samples

Run this from the repository root. It attempts every selectable Fldigi registry
mode, writes successful audio and manifests into the store, and reports progress
and per-mode failures while it runs:

```sh
uv run --extra api --extra cv sig-id-build-fldigi-corpus \
  --data-dir data/cv-seed/store \
  --output-dir data/cv-seed/fldigi \
  --mode all \
  --examples-per-mode 1
```

For a quick single-mode run, replace `all` with a registry ID such as `bpsk31`.
Fldigi must be configured in `~/.fldigi`; the host also needs `fldigi`,
`xvfb-run`, `pactl`, `parecord`, and the SciPy extra. Some receive-selectable
modes may fail to transmit. Review `data/cv-seed/fldigi/fldigi-corpus.json` for
the success and failure counts.

## 2. Build ten-second CV scenes

The Fldigi command captures native audio of each mode's natural duration. The
scene builder places accepted signal samples into ten-second windows, adds
recorded noise from two-second intervals with no active bounding-box overlap,
and records the expected signal boxes. Each signal is set to a randomly chosen
level from 40 dB below to 15 dB below the aligned noise power in its occupied
frequency band over its active interval.
Use the same dataset store as the Fldigi command and pass the recordings
database as a noise source:

```sh
uv run --extra api --extra cv sig-id-build-cv-scene-corpus \
  --data-dir data/cv-seed/store \
  --noise-data-dir ~/.local/share/sig-id \
  --output-dir data/cv-seed/scenes-live-v3 \
  --seed 4192026 \
  --overlap-rate 0.25 \
  --noise-only-examples 360
```

`--overlap-rate` controls how often a scene contains two signals; those signals
are placed in separate time intervals so their detection boxes cannot overlap.
The command reports its scan, source and noise counts, and progress for each
ten-second scene and each requested noise-only window. Progress goes to stderr;
the final JSON report goes to stdout and is saved as
`data/cv-seed/scenes-live-v3/cv-scenes.json`.

Scene output directories are recipe-specific. If a directory already contains
`cv-scenes.json` from a different seed, source pool, or noise pool, select a new
`--output-dir`; the builder preserves the existing output rather than replacing
it. If you change `--noise-data-dir`, pass that same dataset to CV export below
so source lineage is available to the exporter.

## 3. Export ten-second spectrograms

Export the synthetic scenes and the source recordings used as noise. The output
contains ten-second spectrogram images and detector bounding-box labels:

```sh
uv run --extra api --extra cv sig-id-export-cv-dataset \
  --data-dir data/cv-seed/store \
  --also-data-dir ~/.local/share/sig-id \
  --output data/cv-seed/export-live-v2 \
  --seed 4192026
```

The exporter prints review-scan progress by capture, then export progress by
capture and every 100 windows (or about every three seconds during long reads).
Progress is written to stderr; the final JSON report is printed to stdout and
saved as `export-report.json` in the export folder.

Training images stay unmarked; YOLO boxes are stored in matching text files
under `labels/`. To inspect ground-truth boxes, write separate overlays (these
are not used for training):

```sh
uv run --extra cv python tools/preview_cv_labels.py \
  --dataset data/cv-seed/export-live-v2 \
  --split train \
  --limit 24
```

The current CV spectrogram uses a live-waterfall-like landscape layout
(1024 × 640), time runs newest-to-oldest from top to bottom, and its dB range
matches the live display. This preprocessing change requires a fresh dataset
export and retrained CV model. The scene recipe draws a per-signal target from
−40 to −15 dB; use a new scene output directory because existing indexes are
recipe-specific.

The classifier uses fixed 2-second crops filtered to each signal's bounded
frequency range. Review classification samples 1–N evenly spaced crops from the
selected region and combines their predictions. The full-model training script
builds variable-SNR mixed scenes, assigns them by source lineage, and includes
their bounded crops in classifier fitting and validation.

Use a fresh `--output` directory if that export already exists. The exporter
requires sufficient independent reviewed groups with signal and completely
reviewed no-signal windows for train, validation, and test; it stops with a
reporting error when the source data does not meet those requirements.

## 4. Train the CV detector

```sh
uv run --extra api --extra cv sig-id-train-cv-detector \
  --dataset data/cv-seed/export-live-v2 \
  --output-model data/cv-seed/signal-live-v2.pt \
  --epochs 100
```

The detector is class-agnostic: its labels mark signal regions, while the
classifier handles signal identity. The held-out report is based on capture and
lineage groups; do not treat synthetic-only results as evidence of performance
on live radio recordings.

To train the already exported scene run at `data/cv-runs/20261025T184754Z`,
invoke Python from the ROCm environment directly. This skips scene creation and
dataset export, and avoids `uv` selecting or repairing a different environment:

```sh
cd /home/tyler/dev/sig_id
/home/tyler/dev/sig_id/.venv-rocm/bin/python -m sigid.cv_training \
  --dataset data/cv-runs/20261025T184754Z/dataset \
  --output-model data/cv-runs/20261025T184754Z/signal.pt \
  --base-model yolo26n.pt \
  --epochs 100 --batch 8 --device 0 --workers 4 \
  --seed 4192026
```

This command uses the ROCm environment's Python directly, so `uv` and any
stale `VIRTUAL_ENV` setting cannot select another environment.

## Train and activate the full model set

The project wrapper trains the hierarchical classifier and CV detector, then
writes the selected model paths to `data/models/active-models.env`:

```sh
SIGID_NOISE_DATA_DIR="$HOME/.local/share/sig-id" scripts/train_all_models.sh
scripts/run_sig_id.sh
```

By default, the wrapper includes Fldigi samples from `data/cv-seed/store` when
present, along with the configured reviewed datasets. Set
`SIGID_FLDIGI_DATA_DIR`, `SIGID_CV_DATA_DIR`, or
`SIGID_REVIEWED_CORPUS_DIR` to use other inputs. Training creates a timestamped
release under `data/models/releases`; set `SIGID_MODEL_RELEASE` or
`SIGID_MODEL_OUTPUT_DIR` if you need a particular new release path.

## In-app reviewed-data update

The annotation panel's **Train both models from new reviews** action updates the
active pair. It adds previously unused, fully reviewed labeled live captures to
classifier training, rebuilds the CV export from configured reviewed stores plus
the app capture store, and fine-tunes from the active detector checkpoint. Both
artifacts are staged in a timestamped release and activated only after both
models load successfully. The classifier continues training from its existing
family and mode weights using its original synthetic split plus reviewed crops;
new taxonomy classes start with fresh output weights. The CV detector continues
from its current checkpoint.

The app and `scripts/train_all_models.sh` share
`data/models/active-models.json` as the canonical model-path pointer.
`active-models.env` remains available to the shell launcher. Running the full
pipeline publishes a new synthetic/collected-data base pair and supersedes an
incremental pair.

## Live capture and review queue

The live capture selector has three modes:

- **Record + review windows** saves the complete live recording and queues
  successive ten-second clips, as before.
- **Scan only** runs the live waterfall, CV proposals, and classifier analysis
  from the rolling in-memory audio buffer. Stopping the scan discards the audio
  instead of adding a recording to the library.
- **Unattended** requires an active CV detector. It scans successive ten-second
  windows, saves windows with CV detections, and saves a no-detection window at
  the selected interval (one to fifteen minutes). Those clips appear in the
  Review queue with their trigger and detector/classifier summary. The full
  unattended recording is not saved. Sessions stop manually or after eight
  hours.

Unattended detector-negative clips are review candidates, not automatically
verified noise. Complete human review of the saved ten-second clip is required
before it can be used as a real noise-only training example. Classifier
proposals are informational metadata and do not replace signal annotation.
