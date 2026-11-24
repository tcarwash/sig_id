# Active-learning review loop and real-domain reweighting

**Purpose:** turn stored receiver recordings into independently reviewed training
examples without treating model suggestions as ground truth or contaminating
frozen evaluation sessions.

## What is implemented

The local workbench review queue now reads an optional
`active-learning-review-batch.json` from the persistent SigID data
directory. If present, the 100 selected pending captures appear first, in
rank order, with a visible reason and rank. Opening a capture automatically
runs the normal CV detector and classifier suggestions. Those remain
**provisional**: the operator must independently confirm signal geometry
and/or mode identity, and record a complete detection coverage review.
`Mark reviewed` only removes a task from the queue and does **not** create
a verified negative or any supervision.

The batch preparation tool excludes *all captures from each frozen test
session*, including files with different content SHA-256 within the same
session. It skips already-complete reviews, duplicate SHA-256 recordings,
synthetic loopback inputs, and incomplete/long windows. Selection
round-robins over independently recorded sessions and seeks device
diversity. Model detections, when present in existing metadata, influence
priority, but do not create accepted labels. Scores are merely review
suggestions, not inference confidence calibration.

```sh
python -m tools.prepare_active_learning_queue \
  --data-dir ~/.local/share/sig-id \
  --frozen-exclusions /path/to/frozen-noise-exclusions.json \
  --limit 100
```

The CLI writes the worklist atomically into the data directory and records
SHA-256 provenance for the frozen exclusion manifest. Re-run after
reviewing a batch to generate new priorities. You can operate offline.

### Optional second-pass model disagreement scoring

The metadata-only review queue is useful for session diversity but cannot
distinguish strong CV misses from ordinary unannotated recordings. For a
stronger **read-only** review ranking, screen the eligible (non-frozen) pool
against the installed baseline CV detector and independent spectral-track
heuristic. Score CV-only, track-only, spatially conflicting, agreeing, and
neither-detected proposals. The last category is a **possible negative
control**, never a verified noise-only label. Counts are hints; do not use
them as supervision.


```sh
python -m tools.score_active_learning_queue \
  --data-dir ~/.local/share/sig-id \
  --frozen-exclusions /path/to/frozen-noise-exclusions.json \
  --model /path/to/verified-cv-signal.pt \
  --scan-limit 200 --limit 100 \
  --evidence-output /path/to/local/active-review-proposal-evidence.json

```

The scanner must receive the frozen exclusion manifest and excludes
entire held-out source sessions **before** reading audio or running either
detector. It keeps the live DB, accepted regions, complete coverage, and
training datasets unchanged. Proposal evidence is recorded by source-waveform
SHA, and the worklist remains `training_eligible: false`.
Both detectors can fail on weak RF or noisy inputs; even agreement is not
a verified signal. Unscored clips remain eligible for ordinary review.

### Adding another recording source

Use `tools/import_review_sources.py` to bring receiver audio from a
separate CaptureStore into the workbench queue with exact-waveform SHA checks
and an SQLite backup first:

```sh
python -m tools.import_review_sources \
  --source-data-dir /path/to/preview/data \
  --destination-data-dir ~/.local/share/sig-id \
  --device-match 'USB Audio CODEC' \
  --backup /path/to/staging/main-before-import.sqlite3
```

The importer is idempotent by waveform SHA and preserves the original
session identifier. The `sigid_signal_lab.monitor` loopback source is
excluded. Imported clips have **zero** accepted annotations or coverage and
are not automatically eligible for training.

## Training: emphasizing verified real recordings

`tools.compose_cv_training_dataset.py` now accepts
`--real-train-repeat 1..8`. It repeats only original *human-reviewed*
10-second real training windows, and makes the extra image copies explicit
in the manifest as `base-real-resample`. It never duplicates validation or
test images or claims resampled images are new independent sources.

```sh
python -m tools.compose_cv_training_dataset \
  --base /path/to/original-cv-dataset \
  --supplement /path/to/v4/train-supplement \
  --output /path/to/experiment/cv-real4 \
  --real-train-repeat 4
```

The November 24 experiment yielded 1,540 training images (1,207 originals
plus 333 resampling copies), with the **original 159 validation and 161 test
images left unchanged**, and only 111 unique real training sources. All
source checksums and lineage exclusions still apply.

`sigid.hierarchical_training` now accepts
`--reviewed-real-repeat 1..8`, which deterministically upweights the
human-reviewed real training rows without changing validation/OOD
calibration. This is equivalent to a simple sample weighting, not creation
of independent recordings, and it must be evaluated against a
same-input, same-split baseline.

**Important:** retraining before new human confirmations can only
reweight previously reviewed observations. It cannot learn valid signal
geometry, negative ground truth, or modulation labels from unresolved
captured audio. Add newly reviewed examples to training only after the
source-independent frozen set is excluded and the provenance is checked.

## Review priority and caveats

The ideal next user review is a 100-window batch spanning independent
receiver sessions. Session diversity matters much more than the nominal
number of 10-second excerpts. Prioritize missing detections and
model/classifier disagreement when those signals are available, while
retaining independent noise-only examples. Current ranking relies on
available capture metadata and session/device diversity, not an
exhaustive second-pass RF classifier sweep. Neither an empty model
prediction nor apparent silence establishes a negative label.

All frozen evaluation sessions remain intact. Raw capture WAVs, staged
training datasets, and private SQLite backups stay **outside GitHub**.
Production checkpoint pointers do not change during a training experiment.

Model promotion still needs the independent real recall, precision,
false-detections-per-hour, confidence calibration, checkpoint-privacy,
and licensing gates described in
[MODEL_DISTRIBUTION.md](MODEL_DISTRIBUTION.md).
