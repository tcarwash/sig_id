# Signal ID — experimental model prerelease (November 24, 2026)

**EXPERIMENTAL TESTER BUILD — NOT AN APPROVED MODEL RELEASE**

Tag: models-v2026.11.24.1 • Ultralytics 8.4.184 • source under AGPL-3.0.

This prerelease packages the previously installed October 26 Signal ID
spectrogram detector and hierarchical classifier. The detector's PyTorch
checkpoint was rebuilt from identical model tensors with private training
paths and irrelevant training-state metadata removed. The classifier
weights are the original baseline; its public sidecar metadata is filtered
to omit private source paths and reviewer/session identifiers.

This pair is offered so testers can run, report results, and contribute
verified annotations without training models locally. It is **not**
suitable as a default model set or as verified modulation identification.

## Known limitations

- On frozen evaluated real-receiver windows, detector recall at IoU 0.50
  was **25.0% (12/48 positive boxes)**; precision **70.6%** (12 TP, 5 FP).
  Overall recall was 60% (66/110). Test-window false detections were
  **24.6 per audio hour**; not an independently measured noise-only rate.
- Hierarchical classifier calibration rejects all identities rather than
  asserting unsupported modulation certainty. Uncalibrated suggestions
  and detector geometry remain provisional.
- Evaluation lacks independent receiver diversity. Newer November
  real-weighted models performed worse and are not included.
- For exploratory work only; not for safety-critical, autonomous, or
  other correctness-sensitive decisions.

## Install: deliberate opt-in only

Requires Signal ID source and optional API/CV dependencies. From this
source tag or a compatible newer Signal ID checkout:

~~~sh
python3 -m pip install -e '.[api,cv,audio]'
gh release download models-v2026.11.24.1 \
  --repo tcarwash/sig_id \
  --pattern 'sig-id-models.zip*'
sha256sum -c sig-id-models.zip.sha256
python3 -m sigid.model_distribution --install ./sig-id-models.zip \
  --allow-development --data-dir ~/.local/share/sig-id
sig-id-serve
~~~

**This release is deliberately NOT available through --install-latest.**
That command accepts only approved, non-prerelease models. The
--allow-development flag is intentionally required for this build.

Use a separate --data-dir if preserving your current models. Installation
updates the active-models.json pointer but retains prior release folders.
To roll back, restore your prior active-models.json.

## Contents, verification and licensing

The ZIP includes classifier.json, six .npz classifier components,
cv-signal.pt, its metadata, and manifest.json. SHA-256 hashes cover the
archive and every component. No raw audio, dataset files or reviewer
manifests are included.

The detector derives from Ultralytics YOLO and is distributed under
GNU Affero General Public License version 3. The Signal ID application
and corresponding source/training scripts are in this public source tag
with the LICENSE file; the Ultralytics dependency version is pinned.
Comply with AGPL-3.0 source-disclosure requirements or obtain a separately
applicable Ultralytics license when reusing these components.

This experiment does not change approved-release evaluation gates or
enable unattended automatic model installation. Reports of detection
misses, false positives and review UX are welcome; please avoid placing
private recordings or identifying data in public issues.
