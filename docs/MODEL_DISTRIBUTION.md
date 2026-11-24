# Pretrained model releases and installation

Signal ID separates source code, training data, local checkpoints, and **approved public model releases**. End users should not have to train their own detector or classifier, once a validated release exists.

## Current status

**November 24, 2026: distribution implementation exists, but there is no approved public model release.** The active October 26 classifier has an all-abstain/reject-all validation status and its detector only recovered 25% of the held-out live boxes in the last evaluated set. **Do not publish those weights as the default pretrained experience.**

The source repository also needs an explicit project license and a review of the Ultralytics YOLO/AGPL-3.0 requirements (or proof of an appropriate separate license). The publishing script refuses to upload until the source LICENSE file and reviewed Git tag exist.

## User installation (after an approved GitHub model release exists)

Normal install with built-in CLI:

```sh
sig-id-models --install-latest
sig-id-serve
```

Equivalent module entry point for development:

```sh
python -m sigid.model_distribution --install-latest --data-dir ~/.local/share/sig-id
```

For Docker, opt into a once-on-first-run download to the persistent volume, and install CV dependencies:

```sh
SIG_ID_AUTO_INSTALL_MODELS=1 SIG_ID_INSTALL_CV=1 docker compose up --build -d
```

The app does **not** fetch models without explicit opt-in. On offline/failure cases it continues in manual-analysis mode; it never silently accepts an unvalidated development model. Models are installed in `$SIG_ID_DATA_DIR/models/releases/models-vYYYY.MM.DD/`; the active pair is selected by an atomic `models/active-models.json` pointer. User-provided `SIGID_ACTIVE_MODELS_JSON` or explicit local model paths still take priority. The installer preserves existing pointer content if a downloaded asset fails validation.

GitHub releases are queried from the official `tcarwash/sig_id` repository. Downloads require:

- a non-draft, non-prerelease `models-vYYYY.MM.DD[.N]` release;
- `sig-id-models.zip` and its `sig-id-models.zip.sha256` asset;
- matching GitHub tag, archive manifest, and archive SHA-256;
- individually hash-verified model components, compatible feature and CV preprocessing versions;
- an `approved` evaluation verdict. Developer-only bundles cannot be installed automatically.

This is SHA-256 **integrity verification**, not a cryptographic signature or a substitute for trusting the repository/release account. To minimize risk, the downloader only accepts expected GitHub asset URLs, limits download size, and rejects unsafe archive paths and mismatched metadata.

## Model creator: stage a local release

Build from **an evaluation-passing pair**; the source files must include `classifier.json`, all component NPZ weights it references, `cv-signal.pt`, and `cv-signal.pt.json`.

```sh
python -m sigid.model_distribution --build \
  --tag models-v2026.11.24 \
  --classifier /path/to/approved/classifier.json \
  --detector /path/to/approved/cv-signal.pt \
  --output /tmp/sigid-release/sig-id-models.zip
```

Without `--approval-report`, the bundle is explicitly `development`. Local integration testing is possible:

```sh
python -m sigid.model_distribution --install /tmp/sigid-release/sig-id-models.zip \
  --data-dir /tmp/sig-id-test-install --allow-development
```

**Never distribute development bundles as default models.** The bundle contains only required model files and sanitized JSON metadata; private recording paths, training capture UUIDs and dataset split manifests are intentionally excluded from JSON. **PyTorch checkpoints may still embed absolute training paths internally.** An approved public build is rejected if common private-path markers are detected in checkpoint metadata; the actual trained checkpoint must be separately sanitized and its hash updated before release. This scan is conservative, not proof that all private data has been removed.

### Public approval gate

Pass a separate evaluation report with required fields (example values shown **for schema illustration only**, not actual metrics):

```json
{
  "classifier_model_sha256": "64 lowercase SHA-256 hex characters from classifier metadata",
  "detector_model_sha256": "64 lowercase SHA-256 hex characters from CV detector metadata",
  "independent_real_holdout": "frozen real source-lineage/benchmark identifier",
  "evaluator": "independent evaluator name/revision",
  "live_cv_recall_iou50": 0.72,
  "live_cv_precision_iou50": 0.83,
  "false_positives_per_hour": 2.0,
  "licensing_reviewed": true,
  "corresponding_source_available": true
}
```

The builder checks **real IoU-0.50 recall ≥0.50**, **precision ≥0.70**, **false positives/hour ≤10**, matching checkpoint hashes, explicit real holdout provenance, and non-reject-all classifier calibration. These are conservative initial publication gates and should be updated from genuine user needs/frozen benchmark results before release. Claims must be based on measured evidence; setting the fields without performing the evaluation is not an acceptable release process.

After independent validation and confirmed corresponding-source/redistribution compliance:

```sh
python -m sigid.model_distribution --build \
  --tag models-v2026.11.24 \
  --classifier /path/to/approved/classifier.json \
  --detector /path/to/approved/cv-signal.pt \
  --approval-report /path/to/reviewed-evaluation.json \
  --output /tmp/sigid-release/sig-id-models.zip

# Only after merging corresponding source, adding the project license,
# and creating a reviewed Git tag:
scripts/publish_model_release.sh models-v2026.11.24 /tmp/sigid-release/sig-id-models.zip
```

The currently installed October 26 detector checkpoint contains absolute training paths embedded in its PyTorch metadata; it currently **fails** the privacy gate as well as the accuracy/calibration gate. It is unsuitable for publication as-is. The publishing command refuses unapproved bundle contents, missing LICENSE, non-clean source checkouts, missing reviewed tags or an existing release name. It uploads only the validated ZIP and its checksum to GitHub Releases—not training recordings, validation clips, raw model release directories or Git history. **No GH publishing command is executed as part of the distribution implementation.**

## Compatibility and rollback

Only the hierarchical classifier metadata schema 1 and compatible `sig-id-live-waterfall-v2` detector are supported currently. New feature schemas and CV preprocessing versions require a new release and a matching client. Installing a new model switches an atomic pointer; previous release directories are retained, and the pointer can be rolled back to earlier verified models. The installer does not retrain models, upload user recordings, or edit the source repository.

Ultralytics' public licensing documentation describes YOLO framework/model AGPL-3.0 obligations; code and model distribution should be reviewed by the project owner before pushing any trained detector weights publicly. The repository currently lacks a LICENSE file, which the publishing script deliberately treats as a hard blocker.
