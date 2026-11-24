#!/usr/bin/env bash
# Explicit, auditable development-model *prerelease* workflow.
# Never use this for approved/default model releases.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 models-vYYYY.MM.DD[.N] /path/to/sig-id-models.zip docs/EXPERIMENTAL_MODEL_RELEASE.md" >&2
  exit 2
fi
tag="$1"
archive="$(realpath "$2")"
notes="$(realpath "$3")"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checksum="${archive}.sha256"

[[ -f "$archive" && -f "$checksum" && -f "$notes" ]] || { echo "Missing bundle, SHA256 file, or notes" >&2; exit 2; }
[[ "$(basename "$archive")" == "sig-id-models.zip" ]] || { echo "Expected sig-id-models.zip" >&2; exit 2; }
[[ -z "$(git -C "$root" status --porcelain)" ]] || { echo "Commit reviewed release source first" >&2; exit 2; }
[[ -f "$root/LICENSE" ]] && grep -q 'GNU AFFERO GENERAL PUBLIC LICENSE' "$root/LICENSE" || {
  echo "AGPL-3.0 LICENSE and corresponding source are required for this Ultralytics release" >&2
  exit 2
}
[[ "$(git -C "$root" rev-parse --verify "refs/tags/$tag")" == "$(git -C "$root" rev-parse HEAD)" ]] || {
  echo "Release tag must exist and refer to reviewed source HEAD" >&2; exit 2;
}
git -C "$root" ls-remote --exit-code --tags origin "refs/tags/$tag" >/dev/null || {
  echo "Reviewed tag must be pushed to origin" >&2; exit 2;
}
(cd "$(dirname "$archive")"; sha256sum -c "$(basename "$checksum")")
PYTHONPATH="$root" python3 - "$archive" "$tag" <<'PY'
import hashlib, sys
from pathlib import Path
from sigid.model_distribution import _verify_archive, _checkpoint_private_path_markers
bundle = Path(sys.argv[1]).read_bytes()
manifest, payloads = _verify_archive(bundle, allow_development=True)
if manifest["release_tag"] != sys.argv[2] or manifest["evaluation"]["status"] != "development":
    raise SystemExit("Experimental tag/manifest/status mismatch")
if manifest["checkpoint_privacy"] != "no-common-local-path-markers":
    raise SystemExit("Checkpoint still carries private training paths")
private_patterns = (b"/home/", b"/storage5/", b"/Users/", b"\\Users\\", b"\\Documents\\")
for name, data in payloads.items():
    if any(marker in data for marker in private_patterns):
        raise SystemExit(f"Private path marker found inside {name}")
if _checkpoint_private_path_markers(payloads["cv-signal.pt"]):
    raise SystemExit("Unsanitized checkpoint cannot be released")
print("Verified experimental-only model bundle, SHA256:", hashlib.sha256(bundle).hexdigest())
PY
if gh release view "$tag" --repo tcarwash/sig_id >/dev/null 2>&1; then
  echo "Release already exists; refusing to replace public model weights" >&2
  exit 2
fi
gh release create "$tag" "$archive" "$checksum" "$root/LICENSE" \
    --repo tcarwash/sig_id --verify-tag --prerelease --latest=false \
    --title "Signal ID experimental model pair — $tag" \
    --notes-file "$notes"
gh release view "$tag" --repo tcarwash/sig_id --json url,isPrerelease,isDraft,assets
