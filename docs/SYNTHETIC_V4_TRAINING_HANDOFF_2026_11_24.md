# Synthetic dataset v4 — train-only corpus handoff

**Date:** November 24, 2026
**Branch:** `feature/synthetic-corpus-v4`
**Worktree:** `/storage5/dev/sig_id-synth-v4`
**Artifact root:** `/storage5/dev/sig_id-cw-preview-data/synthetic-v4-trainonly-20261124`
**Deployment policy:** **STAGING / TRAIN ONLY.** Do not change the active detector/classifier, Wave B's worktree, or any held-out test/validation files.

## What changed

This builds on initial v4 recipe commit `865abd2`.

- **Original corrected Fldigi sources:** 154 source WAVs were migrated into a clean, new staging database, preserving their original SHA-256 bytes; each source receives measured spectral bandwidth annotation and a v0.2.0 manifest, with original source-manifest provenance.
- **New independent message sources:** 7 deterministic, different received-mode demonstration messages for RTTY, 7 for BPSK31 and 7 for Morse/CW, generated through Fldigi's isolated virtual audio transmitter; plus one earlier CW v4 smoke example. Total **22 newly generated**, yielding **176 unique Fldigi source recordings** before scene mixing. Each message is deterministic for its seed, mode, and index, and differs in content; changing noise or gain on a single recording does **not** count as an independent source. The generator's requested payload is not by itself proven decoded text.
- **Spectral label and temporal-source QA:** v0.2.0 Fldigi sources retain measured active-start/end hints; short mixed-scene source excerpts prefer the active interval, avoiding synthetic labeling of the virtual transmitter's initial silence. Retain the original WAV and complete source SHA.
- **Frozen real holdout isolation:** Wave B's `shadow-real-holdout-v3/windows.jsonl` identifies 26 frozen evaluation recordings across six real capture sessions. The exclusion manifest blocks **33 SHA-identified original recordings** from these sessions from becoming synthetic-noise sources, including other WAVs with *different hashes* from the same session. This removes 25 of 135 otherwise eligible two-second reviewed noise candidates; 110 candidates remain from 22 independent original captures.
- **Balanced v4 recipe:** a deterministic mix of +10..+25, 0..+10, −10..0, −20..−10, and −40..−20 dB **requested per-source in-band** SNR; source-specific continuous audio carrier shifts, realistic separated/adjacent/cochannel pairs, clock variation and non-destructive receiver lowpass where applicable.
- **Train-only exporter:** `--train-only` forces **every** source lineage to `train`; it does **not** generate alternative synthetic validation/test subsets. `--scene-index` attaches component-wise SNR bin, requested/measured SNR, and frequency-placement provenance to `windows.jsonl`. Legacy export behavior remains unchanged when these options are absent.
- **Final release verifier:** `tools/verify_synthetic_train_release.py` checks the frozen evaluation exclusion SHA against every borrowed noise source, tracks unique source hashes across all scenes, validates per-scene SHA and per-component references, verifies all exported image and label checksums, and rejects nontrain images.
- **Independent decoder audit:** `tools/audit_new_fldigi_payloads.py` checks the newly generated source recordings against first-party receive decoders and reports exact/contained payload recovery separately from source mode provenance. CW/RTTY/BPSK31 text has no CRC; printable output must never be equated with verified integrity.

## Artifacts

| Artifact | Path relative to artifact root |
|---|---|
| Source migration log/report | `source-migration.json` |
| Frozen real holdout exclusion manifest | `frozen-noise-exclusions.json` |
| Original 154 + newly transmitted Fldigi source database | `sources/` |
| Three isolated generated modem batches | `new-audio/{rtty,bpsk31,fldigi-cw}-v4/` |
| Balanced multi-signal/recorded-noise scene index | `scenes-balanced/cv-scenes.json` |
| **TRAIN ONLY** spectrogram/image/label supplement | `train-supplement/` |
| Heldout + file checksum validation | `train-release-verification.json` |
| Supplemental native RX quality results | `new-message-decode-audit.json` |

All files live in this model's *remote development host's staging directory*, not in the main repository or the active model deployment.

## Verified full-size staging results (November 24)

The release verifier completed successfully and wrote `train-release-verification.json` with `valid_train_only_release=true`:

| Measurement | Verified result |
| --- | ---: |
| Fldigi source recordings (unchanged originals + new) | 176 (154 + 22) |
| New CW/RTTY/BPSK31 requests | 8 / 7 / 7 |
| Balanced mixed-signal scenes | 108 |
| Signal components represented in scenes | 176 (one source SHA per scene) |
| Recorded-noise-backed scenes | 108 / 108 |
| Unique independent reviewed-noise recordings | 22 |
| CV training images | **388** |
| Positive / negative CV images | **180 / 208** |
| CV validation / test images | **0 / 0** |
| Oversized source-label warnings (>4x spectral estimate) | 0 |
| Source measurement errors | 0 |
| Frozen evaluation exclusion | 33 original WAV SHA from 6 source sessions, confirmed |

Requested in-band SNR components: clear 23, moderate 34, borderline 39, weak 50, extreme 30. Among 108 mixed scenes, 46 have temporal pair overlap and 32 have overlap in annotated frequency; these are independent properties, not a proven count of time-frequency collisions. Clock-error components cover −250, −100, 0, +100, +250 ppm. Reusing one reviewed-noise recording up to 33 times **inside training** remains an important diversity limitation.

Native receiver check (not a source-identity or checksum proof): 8/8 CW payloads recovered exactly, BPSK31 0/7 exact/contained but 7 partial results, RTTY 0/7 exact/contained and 7 abstentions. The generated WAVs remain valid known-mode sources with requested payloads, **not independently proven successful text transmissions**. Do not describe these data as a verified payload-decode corpus.

Full quality report: `scene-quality-report.json` in the same artifact root. No training or model activation was triggered.

## How Wave B should consume it

1. **Do not directly train against this supplement's `data.yaml`** as a complete standalone experiment. Its validation and test directories are empty intentionally.
2. Fold the new training image/label pairs into the *training split only* of the existing versioned detector training experiment. Keep exactly the same real held-out validation and test groups, threshold selection policy, model architecture, and preprocessing version for an A/B ablation.
3. Verify `train-release-verification.json` reports `valid_train_only_release=true`, zero heldout images, no reused source SHA across scenes, and no borrowed frozen-session noise source.
4. Evaluate detection recall, false alarms/hour, localization MAE and crowded-scene recall on **the same frozen original real test sessions** against a source-count-/class-balanced v3 training control. A greater synthetic image count alone does not establish better detection.
5. Deploy only if Wave B release gates accept the model; do not alter 8013/8014/8015, active model pointers or existing snapshots from this branch.

The mode imbalance is improved (especially Morse/CW, RTTY and BPSK31) but the 154 legacy sources still include many different MFSK/PSK variants. **Do not describe the overall mode classes as statistically balanced.** Recorded noise is reused from only 22 original sources after frozen-session filtering, so additional independent real receiver recordings remain valuable.

## Reproduction

```bash
cd /storage5/dev/sig_id-synth-v4
export PYTHONPATH=.
ROOT=/storage5/dev/sig_id-cw-preview-data/synthetic-v4-trainonly-20261124

# 1. Freeze the source SHA exclusions, including related test sessions:
python -m tools.prepare_synthetic_holdout_exclusions \
  --noise-data-dir /home/tyler/.local/share/sig-id \
  --holdout-windows /storage5/dev/sig_id-cw-preview-data/impact-baseline-20261124/shadow-real-holdout-v3/windows.jsonl \
  --output "$ROOT/frozen-noise-exclusions.json"

# 2. Create-only v0.2.0 manifest migration from original v3 source library:
python -m tools.relabel_fldigi_sources \
  --source-data-dir /storage5/dev/sig_id-cw-preview-data/training-20261122-v2/store \
  --target-data-dir "$ROOT/sources" --report "$ROOT/source-migration.json"

# 3. Deterministic new virtual-audio transmissions (no radio transmission).
#   Reproduction should use fresh output directories. Invocations with the
#   same output folder are NOT duplicate-safe for generated live waveforms:
for mode in rtty bpsk31 fldigi-cw; do
  python -m sigid.fldigi_synthetic --data-dir "$ROOT/sources" \
    --output-dir "$ROOT/new-audio/$mode-v4" \
    --mode "$mode" --examples-per-mode 7 --seed 20261125 --timeout-seconds 150
done

# 4. Balanced v4 scenes, with real frozen-session noise excluded:
python -m sigid.cv_scene_corpus --data-dir "$ROOT/sources" \
  --output-dir "$ROOT/scenes-balanced" --seed 20261124 \
  --overlap-rate 0.65 --simultaneous-rate 0.65 \
  --snr-profile balanced --frequency-layout mixed --noise-only-examples 100 \
  --noise-data-dir /home/tyler/.local/share/sig-id \
  --exclude-noise-sha-file "$ROOT/frozen-noise-exclusions.json"

# 5. Training-only export and immutable release verification:
python -m sigid.cv_dataset --data-dir "$ROOT/sources" \
  --output "$ROOT/train-supplement" --train-only \
  --scene-index "$ROOT/scenes-balanced/cv-scenes.json" --seed 20261124

python -m tools.verify_synthetic_train_release \
  --source-data-dir "$ROOT/sources" \
  --scene-index "$ROOT/scenes-balanced/cv-scenes.json" \
  --train-export "$ROOT/train-supplement" \
  --frozen-exclusions "$ROOT/frozen-noise-exclusions.json" \
  --output "$ROOT/train-release-verification.json"
```

## Known remaining limitations

- Fldigi's requested TX payload is not independently verified simply because isolated virtual-audio capture completed. The separate native decoder check records exact/partial/abstained cases, without modifying mode labels.
- Fldigi Morse has intentional key-up gaps; active-window tracking cannot simply trim every silent interval without destroying the CW message.
- Welch 5th–95th-percentile spectral occupancy is a conservative machine-readable frequency box, not a standards-defined occupied-emissions proof.
- V4 in-band SNR measurements derive from pre-mix component energies; they do **not** estimate the true post-normalization decoder input SNR with a calibrated receiver.
- Real recorded noise includes repeat uses of 22 underlying captures; reuse is permitted **within training only**, not across frozen sessions or heldout sets.
- Further gains need real receiver diversity, RF front-end filtering/AGC/channel-model realism, independently checked payload contents, and model training A/B tests rather than simply increasing synthetic volume.

## Test status

The extended generator/export/lineage suite passed **123 tests**, 0 failures, before release export; these cover train-only manifests, short source activity, independent payload generation, heldout-session exclusion, message/scene derivation, and original generator behavior. Run the release verifier before merging to Wave B.
