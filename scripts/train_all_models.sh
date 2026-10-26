#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="${SIGID_MODEL_RELEASE:-$(date -u +%Y%m%dT%H%M%SZ)}"
RELEASE_DIR="${SIGID_MODEL_OUTPUT_DIR:-$ROOT/data/models/releases/$STAMP}"
SPLIT="${SIGID_CLASSIFIER_SPLIT:-$ROOT/data/training/ask-ppm-2026-10-24/known/splits.json}"
CORPUS="${SIGID_CLASSIFIER_CORPUS:-$(dirname "$SPLIT")/corpus.json}"
CLASSIFIER_DATA="${SIGID_CLASSIFIER_DATA_DIR:-$ROOT/data/training/ask-ppm-2026-10-24/dataset}"
OOD="${SIGID_OOD_CORPUS:-$ROOT/data/training/ask-ppm-2026-10-24/diverse/ood-corpus.json}"
CV_DATA="${SIGID_CV_DATA_DIR:-${SIG_ID_DATA_DIR:-$HOME/.local/share/sig-id}}"
REVIEWED_CORPUS="${SIGID_REVIEWED_CORPUS_DIR:-$ROOT/data/training/ask-ppm-2026-10-24/dataset-reviewed}"
FLDIGI_DATA="${SIGID_FLDIGI_DATA_DIR:-$ROOT/data/cv-seed/store}"
SYNTHETIC_CV_DATA="${SIGID_SYNTHETIC_CV_DATA_DIR:-$ROOT/data/cv-seed/store}"
NOISE_DATA="${SIGID_NOISE_DATA_DIR:-}"
CV_BASE="${SIGID_CV_BASE_MODEL:-$ROOT/yolo26n.pt}"
CV_EPOCHS="${SIGID_CV_EPOCHS:-100}"
CV_BATCH="${SIGID_CV_BATCH:-8}"
ACTIVE_ENV="${SIGID_ACTIVE_MODELS_ENV:-$ROOT/data/models/active-models.env}"
ACTIVE_JSON="${SIGID_ACTIVE_MODELS_JSON:-$ROOT/data/models/active-models.json}"

if [[ -e "$RELEASE_DIR" ]]; then
  echo "Model release path already exists; choose a new SIGID_MODEL_RELEASE or SIGID_MODEL_OUTPUT_DIR: $RELEASE_DIR" >&2
  exit 2
fi
mkdir -p "$RELEASE_DIR" "$(dirname "$ACTIVE_ENV")"
if [[ ! -f "$SPLIT" || ! -d "$CLASSIFIER_DATA" ]]; then
  echo "Classifier split or dataset is missing. Set SIGID_CLASSIFIER_SPLIT and SIGID_CLASSIFIER_DATA_DIR." >&2
  exit 2
fi
if [[ ! -f "$CORPUS" ]]; then
  echo "Classifier source corpus not found at $CORPUS. Set SIGID_CLASSIFIER_CORPUS." >&2
  exit 2
fi
if [[ ! -d "$CV_DATA" ]]; then
  echo "CV source dataset not found at $CV_DATA. Set SIGID_CV_DATA_DIR to the reviewed capture dataset." >&2
  exit 2
fi
if [[ ! -d "$REVIEWED_CORPUS" ]]; then
  echo "Compiled reviewed corpus not found at $REVIEWED_CORPUS. Set SIGID_REVIEWED_CORPUS_DIR to its dataset root." >&2
  exit 2
fi
if [[ ! -f "$CV_BASE" ]]; then
  echo "CV base model not found at $CV_BASE. Set SIGID_CV_BASE_MODEL to a supported YOLO checkpoint." >&2
  exit 2
fi

CLASSIFIER_MANIFEST="$RELEASE_DIR/classifier.json"
CLASSIFIER_SCENES="$RELEASE_DIR/classifier-scenes"
CLASSIFIER_SPLIT="$RELEASE_DIR/classifier-splits.json"
CV_EXPORT="$RELEASE_DIR/cv-dataset"
CV_MODEL="$RELEASE_DIR/cv-signal.pt"

SCENE_ARGS=(--data-dir "$CLASSIFIER_DATA" --output-dir "$CLASSIFIER_SCENES"
  --seed "${SIGID_CLASSIFIER_SCENE_SEED:-4192026}" --overlap-rate 0.25)
SCENE_ARGS+=(--noise-data-dir "$CV_DATA" --noise-data-dir "$REVIEWED_CORPUS")
if [[ -n "$NOISE_DATA" && -d "$NOISE_DATA" ]]; then
  SCENE_ARGS+=(--noise-data-dir "$NOISE_DATA")
fi
echo "Building variable-SNR classifier scenes from -30 to -5 dB..."
uv run --extra api --extra cv sig-id-build-cv-scene-corpus "${SCENE_ARGS[@]}"
uv run --extra api --extra cv sig-id-split-corpus \
  --corpus "$CORPUS" --data-dir "$CLASSIFIER_DATA" \
  --output "$CLASSIFIER_SPLIT" --seed "${SIGID_CLASSIFIER_SPLIT_SEED:-20261024}" \
  --assignment-version 2

CLASSIFIER_ARGS=(
  --split "$CLASSIFIER_SPLIT" --data-dir "$CLASSIFIER_DATA" --ood-corpus "$OOD"
  --reviewed-data-dir "$CV_DATA" --reviewed-data-dir "$REVIEWED_CORPUS"
  --feature-version passband-log-spectrum-envelope-roi-fixed-window-v4
  --classifier-window-seconds 2
)
if [[ -f "$FLDIGI_DATA/dataset.sqlite3" ]]; then
  CLASSIFIER_ARGS+=(--fldigi-data-dir "$FLDIGI_DATA")
fi

uv run --extra api --extra cv python -m sigid.hierarchical_training \
  "${CLASSIFIER_ARGS[@]}" \
  --output-manifest "$CLASSIFIER_MANIFEST"

CV_EXPORT_ARGS=(--data-dir "$REVIEWED_CORPUS" --also-data-dir "$CV_DATA" --output "$CV_EXPORT")
if [[ -f "$SYNTHETIC_CV_DATA/dataset.sqlite3" ]]; then
  CV_EXPORT_ARGS+=(--also-data-dir "$SYNTHETIC_CV_DATA")
fi
if [[ -n "$NOISE_DATA" && -f "$NOISE_DATA/dataset.sqlite3" ]]; then
  CV_EXPORT_ARGS+=(--also-data-dir "$NOISE_DATA")
fi
uv run --extra api --extra cv sig-id-export-cv-dataset "${CV_EXPORT_ARGS[@]}"
uv run --extra api --extra cv sig-id-train-cv-detector \
  --dataset "$CV_EXPORT" --output-model "$CV_MODEL" \
  --base-model "$CV_BASE" --epochs "$CV_EPOCHS" --batch "$CV_BATCH"

TEMP_ENV="$(mktemp "${ACTIVE_ENV}.XXXXXX")"
printf 'export SIG_ID_MODEL_PATH=%q\n' "$CLASSIFIER_MANIFEST" > "$TEMP_ENV"
printf 'export SIG_ID_CV_MODEL_PATH=%q\n' "$CV_MODEL" >> "$TEMP_ENV"
mv -f "$TEMP_ENV" "$ACTIVE_ENV"
mkdir -p "$(dirname "$ACTIVE_JSON")"
python - "$ACTIVE_JSON" "$CLASSIFIER_MANIFEST" "$CV_MODEL" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

target = pathlib.Path(sys.argv[1]).expanduser()
target.parent.mkdir(parents=True, exist_ok=True)
payload = {"classifier": str(pathlib.Path(sys.argv[2]).resolve()),
           "cv_detector": str(pathlib.Path(sys.argv[3]).resolve())}
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
    temporary = pathlib.Path(stream.name)
os.replace(temporary, target)
PY
echo "Trained classifier and CV detector. Active model paths saved to: $ACTIVE_ENV"
echo "Run the workbench with: $ROOT/scripts/run_sig_id.sh"
