#!/usr/bin/env bash
# Publish a previously approved Signal ID model archive to GitHub Releases.
# Never publishes local training captures or any model without its release gate.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: scripts/publish_model_release.sh models-vYYYY.MM.DD /path/to/sig-id-models.zip" >&2
  exit 2
fi
tag="$1"
archive="$(realpath "$2")"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checksum="${archive}.sha256"
[[ -f "$archive" && -f "$checksum" ]] || { echo "Bundle and .sha256 asset are required" >&2; exit 2; }
[[ "$(basename "$archive")" == "sig-id-models.zip" ]] || { echo "Bundle must be named sig-id-models.zip" >&2; exit 2; }
[[ -z "$(git -C "$root" status --porcelain)" ]] || { echo "Commit reviewed release code before publishing" >&2; exit 2; }
[[ -f "$root/LICENSE" ]] || {
  echo "BLOCKED: repository has no LICENSE. Resolve AGPL-3.0 corresponding source/redistribution obligations before public publication." >&2
  exit 2
}
git -C "$root" rev-parse --verify "refs/tags/$tag" >/dev/null 2>&1 || {
  echo "BLOCKED: create the reviewed model release Git tag after source promotion" >&2
  exit 2
}
(
  cd "$(dirname "$archive")"
  sha256sum -c "$(basename "$checksum")"
)
PYTHONPATH="$root" python3 - "$archive" "$tag" <<'PY'
import json
import sys
from pathlib import Path
from sigid.model_distribution import _verify_archive
manifest, _files = _verify_archive(Path(sys.argv[1]).read_bytes(), allow_development=False)
if manifest["release_tag"] != sys.argv[2]:
    raise SystemExit("Release tag does not match model manifest")
if manifest["evaluation"]["status"] != "approved":
    raise SystemExit("Model has not passed the public release evaluation gate")
print("Public model release gate passed:", manifest["release_tag"])
PY
if gh release view "$tag" --repo tcarwash/sig_id >/dev/null 2>&1; then
  echo "BLOCKED: release tag already exists; don't silently overwrite published model weights" >&2
  exit 2
fi
gh release create "$tag" "$archive" "$checksum" \
  --repo tcarwash/sig_id --verify-tag \
  --title "Signal ID pretrained models $tag" \
  --notes "Versioned, evaluation-approved Signal ID pretrained model pair. Use sig-id-models --install-latest. The bundle includes SHA-256 manifests and compatibility metadata. See the repository for source and licensing."
echo "Model release published. Validate the public download in a clean model data directory."
