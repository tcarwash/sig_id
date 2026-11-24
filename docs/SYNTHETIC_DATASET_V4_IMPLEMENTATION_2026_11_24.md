# Signal ID — synthetic dataset recipe v4: implementation and handoff

**Date:** 2026-11-24
**Branch:** `feature/synthetic-corpus-v4` (isolated worktree `/storage5/dev/sig_id-synth-v4`)
**Base:** `eb35fec`
**Scope:** Synthetic source metadata, generator, scene and CV export quality only. **No Wave B changes, active model updates, training jobs, radio interaction, or preview restarts.**

## Why this work was required

The November 22 v3 synthetic scene corpus has 92 scenes and 154 annotated signal components, using 27 distinct original reviewed-noise recordings. Its requested in-band SNR range is **−40 to −15 dB** (123 components below −20 dB; 31 from −20 to −15 dB). Of the 154 signal components, **150 have 1450–1550 Hz frequency labels**, including narrow CW, RTTY and BPSK modes. A read-only, 90%-power Welch measurement flagged **65/154 boxes wider than 4× measured source occupancy**. All 154 have zero sample-clock error. The base source distribution is strongly skewed (MFSK 75, PSK 30, CW 1, FSK/2-FSK 7).

The active detector was previously measured at 25% held-out **live** recall versus 87% **synthetic** recall, but this does **not** prove these exact label defects caused the gap. The aim is to remove identifiable biases and prove improvement against an independent frozen real-audio benchmark.

## Implemented changes

1. **Verified Fldigi source spectral labels** (`sigid/synthetic_quality.py`, `sigid/fldigi_synthetic.py`, `sigid/synthetic_dataset.py`):
   - Use the actually generated mono PCM WAV, 4096-sample Hann Welch spectra, 5th-to-95th percentiles of in-band energy and a conservative guard (at least 12 Hz or 3% of measured band).
   - Keep the core estimate and waveform activity statistics in manifest `parameters.band_measurement`.
   - New Fldigi sidecars are generator version `0.2.0` and are revalidated by measuring the WAV again at import. Historical `0.1.0` manifests remain valid and unchanged.
   - Spectral occupancy estimates are *not* authoritative modem emission bandwidths and do not independently verify that intended bits were transmitted. Source temporal annotations still span the full waveform; transmitter pre/post-roll trim needs a separately validated migration.
2. **Create-only migration of existing source captures** (`tools/relabel_fldigi_sources.py`):
   - Reads only accepted, SHA-256-verified original Fldigi source captures and their original manifests.
   - Reimports the byte-for-byte identical WAVs into a **new** CaptureStore with v0.2.0 spectral labeling, immutable original-manifest hash and recorded source digest.
   - Idempotent: runs can be resumed without duplicating captured audio.
   - Full staging completed: **154 of 154 Fldigi sources, 0 errors** (6 from initial smoke test + 148 later). Mean original label width **1470.05 Hz**, mean newly estimated/padded width **801.24 Hz** across the source set. Wide modes legitimately remain wide.
3. **New v4 scene recipe** (`sigid/cv_scene_corpus.py`):
   - Versioned separately: schema 5, scene recipe 4, new output dir required; no in-place v3 corpus mutations.
   - Default `--snr-profile balanced` chooses reproducible per-source in-band SNR targets from: clear +10..+25 dB (15%), moderate 0..+10 (20%), borderline −10..0 (25%), weak −20..−10 (25%), extreme −40..−20 (15%). `--snr-profile legacy_weak` preserves −40..−15 as a comparable control.
   - Continuous randomized carrier placement across the feasible visible passband. `--frequency-layout mixed|separated|adjacent|cochannel` allows deliberate pair geometry, with infeasible requested layouts explicitly marked instead of silently claimed.
   - Existing mixer supports deterministic clock errors in {0, ±100, ±250} ppm (with seeded probability) and selective non-destructive low-pass source filtering. A negative clock error cannot shift the transmission beyond the 10-second scene.
   - The v4 scene index logs each component's SNR bin/target, selected carrier frequency offset, receiver clock error, low-pass cutoff and frequency-placement request.
4. **More accurate CV sample metadata** (`sigid/cv_dataset.py`):
   - Legacy `signal_overlap` still reports box count for compatibility; additional `signal_count`, `temporal_overlap`, `frequency_overlap` and `time_frequency_collision` prevent sequential transmissions from being conflated with cochannel collisions.
   - Numerical per-signal SNR lives in the v4 **scene index**. It is **not yet propagated** to `windows.jsonl` when those scenes are exported; this is an explicit next task.
5. **Read-only quality audit** (`python -m sigid.synthetic_quality`): measures source spectral extents and checks oversized labels, SNR mix, pair geometry, noise-source reuse, carrier offsets and clock impairment distribution.

## Staged data, without activating it

Baseline v3 read-only report:
`/storage5/dev/sig_id-cw-preview-data/synthetic-v4-audit/legacy-quality-report.json`

New v4 source staging (separate from Wave B):
`/storage5/dev/sig_id-cw-preview-data/synthetic-v4-staging/sources/`

Complete source migration report:
`/storage5/dev/sig_id-cw-preview-data/synthetic-v4-staging/full-source-migration.json`

Initial 6-source v4 scenes:
`/storage5/dev/sig_id-cw-preview-data/synthetic-v4-staging/scenes-smoke/`

Initial v4 smoke results: 4 scenes from six source waveforms; five SNR bins represented (1 clear, 1 moderate, 1 borderline, 2 weak, 1 extreme); two pairs; at least one temporal overlap; no frequency-overlap pairs in that small seed; zero oversized-label warnings; clock errors include 0, −100 and −250 ppm. **This tiny smoke is not evidence of trained model improvement.** The initial small scene run used uniform-noise fallback because the supplied noise dataset directory contained only synthetic captures.

The **recorded-receiver-noise smoke test completed**: all 3 scenes used human-reviewed real receiver noise sampled from 135 eligible two-second intervals (read-only source `/home/tyler/.local/share/sig-id`), all five requested SNR bands appeared in its 6 components, all 3 scenes passed manifest revalidation, **zero oversized-label warnings**, and source clock offsets included −250, 0, +100 and +250 ppm. The test outputs live under `synthetic-v4-staging/real-noise-smoke-*`. These are small generator conformance checks, *not* held-out model accuracy results.

## Repro commands

```sh
cd /storage5/dev/sig_id-synth-v4
export PYTHONPATH=.

# Existing v3 corpus audit (read only):
python -m sigid.synthetic_quality \
  --data-dir /storage5/dev/sig_id-cw-preview-data/training-20261122-v2/store \
  --scene-index /storage5/dev/sig_id-cw-preview-data/training-20261122-v2/scenes/cv-scenes.json \
  --output /storage5/dev/sig_id-cw-preview-data/synthetic-v4-audit/legacy-quality-report.json

# Create-only migration, idempotent, NO change to original source library:
python -m tools.relabel_fldigi_sources \
  --source-data-dir /storage5/dev/sig_id-cw-preview-data/training-20261122-v2/store \
  --target-data-dir /storage5/dev/sig_id-cw-preview-data/synthetic-v4-staging/sources \
  --report /storage5/dev/sig_id-cw-preview-data/synthetic-v4-staging/full-source-migration.json

# Build a NEW v4 scene corpus only when source/heldout noise lineage
# assignments have been checked. Do NOT reuse the v3 output directory.
python -m sigid.cv_scene_corpus \
  --data-dir /storage5/dev/sig_id-cw-preview-data/synthetic-v4-staging/sources \
  --output-dir /storage5/dev/sig_id-cw-preview-data/synthetic-v4-staging/scenes-v4-new \
  --seed 20261124 --overlap-rate 0.7 --simultaneous-rate 0.75 \
  --snr-profile balanced --frequency-layout mixed --noise-only-examples 100 \
  --noise-data-dir /home/tyler/.local/share/sig-id
```

For a credible ablation: compare **v3 vs v4 synthetic train supplements only**, on exactly the same frozen real held-out sessions and detector preprocessing, with matched total training images, class-balanced input sources, explicit false positives/hour, IoU0.5 recall, frequency-center MAE, SNR-stratified recall and overlap recall. Prevent *raw receiver-noise recording SHA* used in training from crossing into validation/test. Keep active model pointer unchanged until metrics pass.

## Remaining open improvements

- **Mode balance:** Regenerate multiple independent messages for underrepresented CW, RTTY and BPSK31. V4 relabeling alone cannot create new independent signal content; simply repeating one CW source across splits would constitute leakage. Validate the requested payload against a second encoder/decoder wherever feasible.
- **Temporal label verification:** Trim/split Fldigi TX audio pre/post-roll and actual key-up gaps without truncating preambles or CW word pauses; retain full waveform source lineage and exact subinterval labels.
- **Receiver diversity:** More real noise receivers/days/bands, AGC/pump/compression, impulse and periodic QRM, sample-rate distortion, HF multipath/fading, broad/nonstationary interference and near-miss negatives. Each impairment needs a recorded parameter and test.
- **SNR QA:** The current `measured_snr_db` is an **analytic pre-quantization band-power ratio** computed from aligned source and noise WAVs, not an independently measured decoded SNR. Add final-render checks and reject clipped/inaudible or mislabeled examples.
- **Export propagation:** Write per-component SNR/noise and frequency-layout provenance into new `windows.jsonl` for automatic coverage stratification.
- **Lineage/holdout safety:** The previous frozen held-out real group must not be changed, and no newly borrowed real-noise source SHA may appear across independent train and test populations. Treat this new v4 corpus as **train-only staging** until the data lineage audit and Wave B release gates approve it.
- **Full-generation scheduling:** The entire v4 scene corpus and retraining were deliberately not triggered during Wave B's work; the six-source scene smoke was generated to verify the generator safely.

## Tests

`tests/test_synthetic_v4.py` covers deterministic five-bin SNR selection, feasible pair layouts, clock overflow protection, narrow waveform measurement, reject silent waves, v0.2 manifest validation against modified frequency metadata, exact generated scene provenance, idempotent scene generation, and old-source create-only migration/rollback/idempotence. The existing synthetic, scene, dataset-splitting, augmentation, CV overlap, and Signal Lab suites are also included in regression checks.

**Verified regression run:** 120 Python tests, 0 failures, 0 errors, 0 skips (synthetic generators, manifests, mixed scenes, CV splits, augmentation, Signal Lab, Fldigi capability mapping and the new v4 tests). The 6-source/3-scene real-noise smoke used 135 eligible reviewed-noise intervals and passed the same scene quality audit.

**Integration caveat:** This branch is based on `eb35fec` and must be merged/cherry-picked with Wave B's in-progress changes, not used to overwrite its working tree.

**Important staging note:** the initial 6-source smoke imported its test scene/noise captures into the same `sources/` CaptureStore used by the subsequent 154-source migration. Treat it as a **prototype staging dataset**, not a clean final training export. The create-only migration tool can reconstruct a clean full source CaptureStore at a fresh `target-data-dir` once Wave B schedules the next training export. The original 154 audio hashes and measured v0.2.0 source labels remain independently reproducible.
