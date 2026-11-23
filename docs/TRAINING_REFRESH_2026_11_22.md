# Signal ID training refresh — November 22, 2026

## Scope

Refresh the class-agnostic signal detector and family/mode classifier without
changing production models during training. The reference baseline is:

- Classifier: `incremental-20261026T041149Z/classifier.json`
- Detector: `incremental-20261026T041149Z/cv-signal.pt`
- Preview: `127.0.0.1:8014` (isolated from port 8013)

This worktree is `/storage5/dev/sig_id-cw-keying`. Staging artifacts and logs:
`/storage5/dev/sig_id-cw-preview-data/training-20261122-v2/`.

## Baseline diagnosis (from model's stored held-out report)

| Condition | IoU 0.5 detector recall |
| --- | ---: |
| Live | 25.0% (12/48 boxes) |
| Synthetic | 87.1% (54/62 boxes) |
| Multiple signal | 41.2% (28/68 boxes) |
| Single signal | 90.5% (38/42 boxes) |

Center-frequency localization MAE of matched boxes: 10.9 Hz. The classifier
strict-calibration status is `reject_all_no_threshold_meets_validation_constraints`
for its active family/mode acceptance. **Do not reinterpret uncalibrated
rankings as confirmed protocol identity.** Native decoders may use the rankings
to propose *bounded trials*, independently checked by demodulation/CRC.

## Training method changes

1. **Actually simultaneous waveforms.** `sigid.cv_scene_corpus` now accepts
   `--simultaneous-rate` (fraction *of paired scenes*) alongside the existing
   `--overlap-rate` (fraction of scenes containing a pair). The new placement
   guarantees at least 50% temporal overlap of the shorter burst, retains two
   separately labeled boxes, uses the original physically grounded noise/SNR
   machinery, and increments the scene recipe version.

2. **Source and negative provenance.** The 154 native Fldigi captures were
   snapshotted into a distinct SQLite store; user recordings and source stores
   were not modified. The new corpus has 92 generated signal scenes, 62 paired,
   **42 genuinely simultaneous** and 100 extra noise-only windows, with
   recorded signal-free noise used for the 92 signal scenes.

3. **Prevent one-shot mode-class explosion.** The Fldigi corpus spans 150 modes,
   most with one example. The old trainer inadvertently created a new expert
   class for each. The revised hierarchical trainer reserves independent
   mode-supervised classes for established modes or new variants with at least
   three independent exemplars. Rare Fldigi examples are used only as
   *family-level* evidence when a supported family exists; their per-family
   contribution is capped so they cannot overwhelm reviewed core samples.
   The strict calibrated-identity gates remain fail-closed.

4. **True holdout-safe fast extension.** The model-training export combines the
   immutable baseline spectrograms with new *training-only* scenes. Shared
   review-derived noise from validation/test capture SHA-256s excludes a scene.
   The tool rejected **40** such scenes, admitted **52** positive scenes
   (including **23 simultaneous** and 33 paired), and added 30 independently
   generated negatives. The original **320 validation/test image rows and
   their image/label bytes are unchanged**. The baseline metadata checksums
   passed after staging.

5. **Train-only live balancing.** Add two extra training presentations per
   93 reviewed live positive windows (186 repeated examples), from source
   hard-links within the staged export. These are **NOT independent captures**,
   and they never enter validation/test. Evaluation must remain on independent
   held-out source lineages.

## Reproducible recipe

Paths here are specific to the isolated host; use a new output directory for
a new run.

```sh
export PYTHONPATH=/storage5/dev/sig_id-cw-keying
R=/storage5/dev/sig_id-cw-preview-data/training-20261122-v2
BASE=/home/tyler/.local/share/sig-id/models/releases/incremental-20261026T041149Z

# The store snapshot is already captured in $R/store/dataset.sqlite3.
python -m sigid.cv_scene_corpus \
  --data-dir "$R/store" --output-dir "$R/scenes" \
  --seed 11222026 --overlap-rate 0.70 --simultaneous-rate 0.75 \
  --noise-only-examples 100 \
  --noise-data-dir /home/tyler/.local/share/sig-id

python -m tools.append_cv_scenes_to_baseline \
  --baseline "$BASE/cv-dataset" --store "$R/store" \
  --scenes "$R/scenes/cv-scenes.json" \
  --output "$R/cv-fast-dataset" --max-noise-train 30
python -m tools.prepare_cv_training_balance \
  --dataset "$R/cv-fast-dataset" --live-repetitions 3

# Run the detector fit from a ROCm-capable Python environment.
python -m sigid.cv_training \
  --dataset "$R/cv-fast-dataset" \
  --output-model "$R/artifacts/cv-signal.pt" \
  --base-model "$BASE/cv-signal.pt" \
  --epochs 18 --batch 8 --device 0 --workers 4 --seed 11222026
```

The classifier candidates run from the existing curated split and reviewed
data, optionally native Fldigi store2, 2-second ROI feature version v4, the
original classifier's compatible warm-start weights, and 1,200 epochs.
The selected preview candidate is `$R/artifacts/classifier-v4-reviewed.json`,
trained with the expanded reviewed-real corpus but **without** the one-shot
Fldigi mode classes. Candidate v3 exercised Fldigi family-only training, but
regressed original-corpus top-three recall; it remains staged for comparison.
The comparison files are `$R/classifier-v4-comparison.json` and
`$R/classifier-v3-comparison.json`. The detector comparison is run by
`tools.compare_cv_refreshed` on the same frozen test images.

## Deployed classifier preview (not production)

A separate local service at `http://localhost:8015` loads
`$R/preview-active-models.json`:

- Classifier: `$R/artifacts/classifier-v4-reviewed.json`, SHA-256
  `23f1f021d09027cd5eacc8de467f2d289bfdf19eabde1e89edf5226dcde07423`.
- Detector: unchanged original release
  `incremental-20261026T041149Z/cv-signal.pt`.
- Preview-specific captures: `$R/preview-data`.

Verified `/api/model/status` and `/api/cv/status` return those respective
artifacts; `/api/health` returns OK on 8013, 8014, and 8015. Existing
`8014` recording was active during deployment and was **not restarted**.

| Classifier | Original corpus top-1 | Original corpus top-3 | Fldigi holdout top-1 | Calibrated identities |
| --- | ---: | ---: | ---: | ---: |
| Original | 26/36 | 36/36 | 0/4 | 0 |
| v3 Fldigi family-only | 27/36 | 33/36 | 4/4 | 0 |
| v4 reviewed-expanded (8015) | 28/36 | 36/36 | 0/4 | 0 |

The comparison is small; top-three routing rank is *not* confirmation of
protocol identity. Strict calibration remains in fail-closed/reject-all mode.

## Detector training runtime status

ROCm 7.1 / PyTorch 2.13 on the host's RX 6700 crashed twice on YOLO
initialization (including with AMP disabled and zero loader workers). Neither
run created a checkpoint. The alternate CPU training attempt uses six epochs
of warm-start fine-tuning on the frozen, safely extended CV dataset; CPU work
was reduced to nice 15 and affinity CPUs 6–7 to limit interference with an
active recording. This CPU run did not reach its first checkpoint and was
terminated to protect the user's active live recording from further contention.
**No new detector has been deployed or claimed validated.**
An output checkpoint is eligible only after held-out baseline comparison.
A repeat run should use a supported stable GPU runtime or be scheduled when
live audio capture is idle. Logs: `$R/cv-training.log`,
`$R/cv-training-safe.log`, `$R/cv-training-cpu.log`.


## Evaluation / deployment rules

- **Do not use the held-out test split to choose model thresholds.** The new
  detector chooses its confidence threshold from the *validation* subset.
- Compare old/new detector on exactly the same frozen test with their respective
  validation-selected thresholds: overall, live, simultaneous/multi-signal
  recall, false detections per audio-hour, and carrier-center MAE.
- The detector comparison script reports conservative gates: live test has at
  least ten real signal boxes; live recall not worse; overall recall decline
  under three points; false alarms not more than 1.35x baseline; and no
  substantial multi-signal regression.
- Compare classifier top-one and top-three **routing hypotheses** separately
  from **accepted, calibrated identities**. Some rare-mode test slices are
  tiny; they are not evidence of generalization without further review.
- Only deploy a passing candidate into the **8014 isolated preview**. Keep
  port 8013 and the shared `data/models/active-models.json` unchanged. Use a
  distinct preview JSON model pointer and preserve the previous pointer for
  rollback. Deploying on 8014 is a field trial, not production promotion.
- Native protocol decoders remain independent verification. The active
  AX.25 decoder's CRC and PSK31 coherence rules are unchanged by retraining.

## Known gaps

- RTTY reviewed data still contains very few independent captures; more live
  RTTY/AX.25 examples are needed.
- The official class-agnostic detector does not yet identify AX.25 as a
  classifier mode. Bell-202 AFSK is modulation evidence, AX.25 a protocol
  verified by CRC, and the two must remain distinct.
- A fully regenerated CV export using three source stores is an optional
  additional experiment, but it must **not** replace the fixed held-out test
  during this model comparison.
