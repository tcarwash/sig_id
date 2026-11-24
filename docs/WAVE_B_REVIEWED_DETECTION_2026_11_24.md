# Wave B / Signal ID: real-recording benchmark and detector pilot

**Date:** 2026-11-24
**Repository:** `/storage5/dev/sig_id-cw-keying`
**Branch:** `feature/continuous-cw-keying`
**Implementation:** `tools/bench_reviewed_chain.py`, `tools/freeze_reviewed_shadow_holdout.py`, training and comparison safeguards.
**State:** benchmark completed; a **non-release** CPU pilot model trained and compared; no model/service was promoted.

## Why the original metric needed scrutiny

The existing frozen CV dataset contains 161 held-out windows at 10 s each:
13 reviewed real-receiver windows from only **3 independent session/source lineages**, plus 148 synthetic windows from 80 lineages. Real ground truth contains 48 boxes, of which 35 are from one 8-GFSK lineage. Thus box-weighted recall on 48 real labels is not the same as broad on-air generalization.

The original export's training, validation, and test source lineages (and capture SHA-256 values) have no overlaps. New code independently rechecks this split, verifies source SQLite provenance, checks every original WAV's SHA-256, verifies exported image/label checksums, and rerenders every held-out v2 spectrogram pixel-identically. Ground-truth labels never flow into model inference or decoder selection.

## Original frozen test: 161 windows / 110 signal boxes

| Measure | Original detector | 20%-data, 2-epoch pilot |
|---|---:|---:|
| IoU 0.50 overall TP/GT | **66/110 (60.0%)** | **69/110 (62.7%)** |
| IoU 0.50 live TP/GT | **12/48 (25.0%)** | **16/48 (33.3%)** |
| IoU 0.50 synthetic TP/GT | **54/62 (87.1%)** | **53/62 (85.5%)** |
| Overall unmatched detections (FP) | **11** | **21** |
| Live unmatched detections (FP) | **5** | **12** |

Baseline original detector tested at the metadata's `confidence_threshold=0.5`. Pilot selected threshold **0.5 on the validation split**; it did not tune the threshold on the test set. The pilot's improved live recall came with more false detections and does not constitute a production improvement. Pilot model metadata explicitly says `train_fraction=0.2` and `pilot_partial_training=true`.

The baseline classifier generated rankings for 77 detections but **accepted zero confirmed protocol identities** under strict calibrated thresholds. This is correctly reported as zero accepted, not zero available rankings.

### Native decoder routing baseline (original frozen test only)

First-party native decoder attempted 12 real, IoU-matched detector proposals, using two bounded candidate modes and no source truth:
- 1 produced `coherent-unverified` text
- 11 produced no coherent output
- 0 generated a CRC/protocol-verified payload
- **There is no independent message-transcript truth for these recordings**, so exact payload correctness remains `null`, not zero or one successful decodes

Use `--payload-references` with explicitly independent, human-transcribed SHA256+sample-window truth records to obtain exact-text correctness. The schema has `schema_version:1` and `references` entries including `capture_sha256`, `sample_start`, `sample_count`, `expected_text`, `independently_reviewed:true`; scoring occurs strictly *after* decoder inference, and labels are never used as acquisition hints.

## Second, independent SHADOW holdout

Without changing or relabeling the source library, `tools.freeze_reviewed_shadow_holdout` snapshotted the source SQLite database, selected entirely **unused source sessions** from the live review queue, and required full non-stale detector annotation coverage (0–3 kHz, all 10 s), accepted non-noise signals, no unresolved active regions, and no ignore/uncertain areas.

- **26 reviewed live windows; 60 accepted signal boxes; 6 new independent source/session lineages**
- Source snapshot database SHA-256: `9f7f71e44819faa0c537d3d090d6ba4df7856c43a4013007ee725cd8d9d392e8`
- Frozen shadow test window manifest SHA-256: `d31ed5c44ce9b2e6322647ec612e590cc8c3cd913ac247e89f5c826ca3330817`
- No train/validation images in the shadow export. The shadow **must never** be used to tune training or confidence threshold.
- Both original and shadow snapshots use `sig-id-live-waterfall-v2` pixels with independent image/label hashes and complete verified PCM/SHA provenance.
- The new shadow set contains **only signal-positive windows**. Unmatched detections in these windows measure localization/box mismatches, **not a noise-only false-alarm rate per audio-hour**. Dedicated independent reviewed real noise negatives are still required.

| IoU 0.50 metric, shadow only | Original detector | Pilot fine-tune |
|---|---:|---:|
| Correctly matched boxes / 60 | **20/60 (33.3%)** | **20/60 (33.3%)** |
| Precision | **60.6%** | **43.5%** |
| Unmatched predicted boxes | **13** | **26** |
| Positive lineage macro recall | **43.9%** | **43.7%** |
| CW matched / 29 boxes | **12** | **14** |
| MFSK matched / 26 boxes | **5** | **3** |
| BPSK matched / 4 boxes | **2** | **2** |

The pilot added CW detections while losing MFSK detections; overall recall stayed flat and false detections doubled on unseen real recordings. **Reject this pilot for deployment.** The baseline active model pointer was checked and remains unchanged.

### Combining evidence carefully

Original held-out live: 3 lineages, 48 boxes. New nonoverlapping shadow: 6 lineages, 60 boxes. **Nine lineages** are now available as distinct real evaluation sources. This does not provide robust radio/mode diversity: the newer shadow is primarily CW/MFSK and the earlier test emphasizes 8-GFSK. Original training may also contain synthetic scenes derived from recorded noise; the shadow freezer excludes known capture/source-session/lineage duplicates, but this does not prove no untracked noise-provenance dependency. Report that uncertainty.

Because the shadow test is newly discovered and separately frozen, avoid repeated tuning to its results. Benchmark candidates only at independently validation-selected thresholds, with no shadow-driven hyperparameter selection.

## Reproducible commands

Paths for this machine (the generated dataset and reports are **local, not tracked in Git**):

```sh
cd /storage5/dev/sig_id-cw-keying
export PYTHONPATH=.
BASE=/home/tyler/.local/share/sig-id/models/releases/incremental-20261026T041149Z
R=/storage5/dev/sig_id-cw-preview-data/impact-baseline-20261124

# Original frozen CV test, including optional bounded native decoder routing:
python -m tools.bench_reviewed_chain \
  --dataset "$BASE/cv-dataset" --detector "$BASE/cv-signal.pt" \
  --classifier "$BASE/classifier.json" --split test \
  --native-decode real --max-native-attempts 16 \
  --output "$R/reproduced-old-test.json"

# To regenerate from the source library (choose a NEW output name each time):
python -m tools.freeze_reviewed_shadow_holdout \
  --source-data-dir /home/tyler/.local/share/sig-id \
  --prior-dataset "$BASE/cv-dataset" \
  --output "$R/shadow-real-holdout-new"

# Original model vs pilot, both with full source/pixel verification:
python -m tools.bench_reviewed_chain \
  --dataset "$R/shadow-real-holdout-v3" --detector "$BASE/cv-signal.pt" \
  --output "$R/shadow-real-baseline-verified.json"

python -m tools.bench_reviewed_chain \
  --dataset "$R/shadow-real-holdout-v3" \
  --detector "$R/pilot/cv-signal-pilot.pt" \
  --output "$R/shadow-real-pilot-verified.json"

# CPU-only PILOT recipe, on separately staged training-only images.
# Never treat --train-fraction below 1.0 as deployable.
python -m sigid.cv_training \
  --dataset /storage5/dev/sig_id-cw-preview-data/training-20261122-v2/cv-fast-dataset \
  --base-model "$BASE/cv-signal.pt" \
  --output-model "$R/pilot/cv-signal-pilot.pt" \
  --epochs 2 --batch 4 --workers 0 --device cpu \
  --train-fraction 0.2 --seed 11242026
```

Model trainer now disables Ultralytics HSV jitter, random erasing, spatial translation, shear, perspective, copy/paste, mosaic, mixup, flips, rotation and scale, because those image transforms do not correspond to sound waveform propagation/receiver impairments. Instead, model training data should contain physically plausible, labeled gain/passband tilt, fading, QRN/QRM, drift and simultaneous transmissions generated in the audio domain.

The comparison tool has new fail-closed release checks:
1. at least five independent positive held-out **real** test lineages (the original test alone has only three),
2. a candidate trained on **100%** of its declared train split (pilot fraction .2 fails),
3. preserved recall, bounded false detections and multi-signal performance against the validation-selected threshold.

A pilot can be compared and retained for research, never automatically promoted. The new shadow also needs independent negative controls before any production claim.

## What to do next

1. **Review additional distinct real sources**, preferably other receivers, bands, noise-only segments, wideband modes, crowded transmissions and multiple time-separated operating sessions. Add fully covered and explicit no-signal records to a separate negative shadow holdout.
2. **Improve real domain transfer**, first with versioned, per-frequency noise-floor normalization and physically realistic impairment augmentation. Regenerate all train/validation images and train a **full** dataset candidate, keeping original validation and test held-out.
3. Address training runtime: the current host CPU successfully fine-tunes, while prior ROCm PyTorch/Ultralytics init crashed. Full training should run on a stable accelerator/idle compute to avoid jeopardizing live capture.
4. **Improve native acquisition**. The low native text yield on matched real signals requires frequency/clock acquisition, framed evidence and eventually streaming demodulators. Evaluate with independently transcribed payload truths.
5. Only promote after success on **both** original frozen holdout and shadow, no threshold tuning on either, adequate false-positive controls, and robust independent real lineage diversity.

## Safety / data integrity

- Source DB and original WAVs were never edited; freezer uses read-only SQLite backup and modifies only its independent local snapshot.
- All live radio/capture servers and model pointers were left unchanged.
- Experimental pilot and shadow files stay in `/storage5/dev/sig_id-cw-preview-data/`, not committed into the source tree.
- No live RF hardware or transmissions were used.
