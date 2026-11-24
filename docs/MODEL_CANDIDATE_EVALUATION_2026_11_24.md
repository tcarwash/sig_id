# Signal ID model-candidate evaluation — November 24, 2026

**Status:** Experimental candidates only. **Do not promote, distribute as default, or publish trained model weights.** This report intentionally contains no private audio, capture UUIDs, local usernames, training file paths, or checkpoint contents.

## Source code and reproducibility

The model distribution and synthetic corpus source code are merged into `main`:

- [PR #20: model distribution](https://github.com/tcarwash/sig_id/pull/20)
- [PR #21: synthetic corpus v4](https://github.com/tcarwash/sig_id/pull/21)
- [PR #22: frozen-holdout-safe training and ROCm](https://github.com/tcarwash/sig_id/pull/22)

Training recipe: [TRAIN_V4_MODEL_SET.md](TRAIN_V4_MODEL_SET.md). The v4 supplement adds 388 train-only CV images to the original 819 training images, while keeping 159 validation and 161 test images byte-for-byte and in the same lineage groups. The 176 distinct Fldigi signals include 22 new independently generated CW, RTTY and BPSK31 examples. These generated modes and messages are **not** assumed to be independently decoded payload truth.

All model training and local artifacts are staged outside the Git repository. The active production pair remains unchanged.

## Hierarchical classifier (500 epochs, warm-started)

| Evaluation subset | Previous | v4 candidate |
| --- | ---: | ---: |
| Core synthetic, heldout top-1 | 26/36 (72.2%) | 22/36 (61.1%) |
| Fldigi, heldout top-1 | 9/17 (52.9%) | 14/17 (82.4%) |
| Fldigi, heldout top-3 | 14/17 (82.4%) | 17/17 (100%) |
| Frozen recorded CW/BPSK31, top-1 across 65 crops | 55.4% | 38.5% |
| Frozen recorded CW/BPSK31, top-3 | 93.9% | 93.9% |
| Frozen recorded CW/BPSK31, capture-mean top-1 | 57.9% | 49.1% |
| Validated accepted mode identities | 0 | 0 |

The real test comprises 26 frozen capture SHA inputs, 19 captures with a common represented mode, 29 annotated regions, 65 correlated feature crops, and six recording sessions. It is **not** a comprehensive survey of all 12 modeled modes. Same-session audio is excluded from the candidate classifier's extra reviewed training input. The candidate still has `reject_all_no_threshold_meets_validation_constraints` family and expert calibration.

**Interpretation:** Better Fldigi ranking does not demonstrate better real-world generalization. In this frozen sample, the old classifier remains better for top-1.

## Detector comparison — original heldout test, IoU 0.50

Both detector models used the same pixel-hash-verified heldout images and YOLO matching logic. The new CPU candidate trained on **100%** of the assigned 1,207 training images for six epochs at 512-pixel image size, warm-started from the previous checkpoint. The operating confidence threshold was selected on validation rather than adjusted against test.

| Measure | Previous detector | CPU v4 | ROCm v4 |
| --- | ---: | ---: | ---: |
| Overall true positives | 66 | 56 | 61 |
| Overall false positives | 11 | 37 | 50 |
| Overall false negatives | 44 | 54 | 49 |
| Overall recall at IoU 0.50 | **60.0%** | 50.9% | 55.5% |
| Overall precision at IoU 0.50 | **85.7%** | 60.2% | 55.0% |
| False detections per audio-hour | **24.6** | 82.7 | 111.8 |
| Original real-source recall (48 boxes) | **12/48 (25.0%)** | 5/48 (10.4%) | 9/48 (18.8%) |
| Original synthetic-source recall (62 boxes) | **54/62 (87.1%)** | 51/62 (82.3%) | 52/62 (83.9%) |

The real test covers only three independent original live lineages; box-weighted scores are affected by one dominant recording. It is important to improve independent source diversity before generalizing these rates to all RF receiving conditions.

An independent run of `tools.bench_reviewed_chain`, with the same image/label hashes, reproduced the ROCm model's 61 TP, 50 FP, 49 FN and 9/48 real-source detections. This is a consistency check, not a second independent test dataset.

**Interpretation:** Both new detectors underperform the prior checkpoint on live recall and produce substantially more false detections. The ROCm candidate partially recovers CPU model recall, but it is still not competitive with the prior detector. **Neither v4 model should replace the existing production checkpoint.** The previous detector's model operating point and the new detectors' validation-selected confidence were used unchanged for these test comparisons.

## ROCm experiment

A Radeon RX 6700 XT (gfx1031) was successfully exercised through PyTorch 2.5.1+rocm6.2 with `HSA_OVERRIDE_GFX_VERSION=10.3.0`, including convolution backward, YOLO inference, and a full pilot training epoch. The independent full-split GPU run completed **10 epochs at 640-pixel training resolution**, using the original 1,207/159/161 train/validation/test assignments without changing any held-out images or setting thresholds from test. The new model artifact SHA-256 is `fe33e930cc2ecb2717bca64ddc327f2f0b9da8ba6dcdb8f89a7134703e7eb796`. Its validation mAP@0.50 improved modestly to 0.704 (CPU candidate 0.683); however, its fixed test IoU0.50 precision was only 55.0%, and real-source recall 18.8%, both worse than the deployed candidate.

The compatibility workaround is specific to the tested GPU/ROCm build and should never be assumed safe for all Radeon devices. The run was completed without changing the active model pointer.

## Additional benchmark integrity

The legacy *shadow real source snapshot* benchmark rejected its dataset because the source SQLite snapshot SHA-256 no longer matched the original freeze manifest. The snapshot/manifest were **not rewritten**, and the source integrity gate was not bypassed. The model comparison above uses the existing pixel-hash-verified original CV validation/test split, not the invalid shadow snapshot. A new immutable independent receiver holdout is needed for future release-quality testing.

## Publication gate

**Not approved for GitHub model-weight release.** Outstanding conditions are independent real-receiver recall/precision, false alarms per real noise-hour, non-reject-all classifier identity calibration, checkpoint privacy sanitation (PyTorch files can embed local training paths), and Ultralytics framework/checkpoint redistribution plus corresponding-source licensing review. The project also requires an explicit source license decision. None of these gates should be sidestepped simply to distribute an experimental checkpoint.

The installer and distribution code are available in `main`; pretrained weights will be separately published as verified GitHub Release assets **only when genuinely approved**. Both CPU and ROCm model pairs were built as separate approximately 4.7 MB **development-only** ZIPs and installed into isolated testing directories. The ROCm development bundle's SHA-256 is `391e901a5b9b7ce29b941d7daef36aec0ab7f97aa5901b3eb33ce7fc98671025`; the installer labeled it `development`. Its detector checkpoint still contains recognizable local-path markers in the embedded PyTorch metadata, so the public packaging gate appropriately rejects it. Neither bundle was uploaded to GitHub Releases.
