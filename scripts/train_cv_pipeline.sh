#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd -- "${1:-$DEFAULT_PROJECT_DIR}" && pwd)"

STORE_DIR="${SIGID_CV_STORE_DIR:-$PROJECT_DIR/data/cv-seed/store2}"
REAL_NOISE_DIR="${SIGID_REAL_NOISE_STORE_DIR:-$HOME/.local/share/sig-id}"
BASE_MODEL="${SIGID_CV_BASE_MODEL:-$PROJECT_DIR/yolo26n.pt}"
RUN_ID="${SIGID_CV_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${SIGID_CV_OUTPUT_DIR:-$PROJECT_DIR/data/cv-runs/$RUN_ID}"
SEED="${SIGID_CV_SEED:-4192026}"
EPOCHS="${SIGID_CV_EPOCHS:-100}"
BATCH="${SIGID_CV_BATCH:-8}"
DEVICE="${SIGID_CV_DEVICE:-0}"
WORKERS="${SIGID_CV_WORKERS:-4}"
NOISE_ONLY_EXAMPLES="${SIGID_CV_NOISE_ONLY_EXAMPLES:-24}"

SCENES_DIR="$OUTPUT_ROOT/scenes"
DATASET_DIR="$OUTPUT_ROOT/dataset"
MODEL_PATH="$OUTPUT_ROOT/signal.pt"

if [[ ! -f "$STORE_DIR/dataset.sqlite3" ]]; then
  echo "CV signal-source dataset not found: $STORE_DIR/dataset.sqlite3" >&2
  echo "Set SIGID_CV_STORE_DIR to the reviewed/synthetic CV dataset root." >&2
  exit 2
fi
if [[ ! -f "$REAL_NOISE_DIR/dataset.sqlite3" ]]; then
  echo "Real-noise dataset not found: $REAL_NOISE_DIR/dataset.sqlite3" >&2
  echo "Set SIGID_REAL_NOISE_STORE_DIR to the dataset containing saved 10-second review samples." >&2
  exit 2
fi
if [[ ! -f "$BASE_MODEL" ]]; then
  echo "YOLO base model not found: $BASE_MODEL" >&2
  echo "Set SIGID_CV_BASE_MODEL to a supported checkpoint." >&2
  exit 2
fi
RESUME_SCENE_BUILD=0
if [[ -e "$OUTPUT_ROOT" ]]; then
  if [[ -f "$SCENES_DIR/cv-scenes.json" && ! -e "$DATASET_DIR" && ! -e "$MODEL_PATH" ]]; then
    RESUME_SCENE_BUILD=1
  else
    echo "Run output exists and is not an incomplete scene-only run: $OUTPUT_ROOT" >&2
    echo "Choose another SIGID_CV_RUN_ID or SIGID_CV_OUTPUT_DIR." >&2
    exit 2
  fi
fi

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT"

# When VIRTUAL_ENV is set, invoke its Python directly. This avoids uv trying to
# create or repair an environment on filesystems that do not support symlinks,
# and avoids syncing over a ROCm-enabled PyTorch build.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_BIN="$(command -v python)"
  if ! "$PYTHON_BIN" -c 'import os, sys; sys.exit(os.path.realpath(sys.prefix) != os.path.realpath(os.environ["VIRTUAL_ENV"]))'; then
    echo "The active python does not belong to VIRTUAL_ENV=$VIRTUAL_ENV" >&2
    echo "Reactivate the ROCm environment, then retry." >&2
    exit 2
  fi
  PYTHON_RUN=("$PYTHON_BIN" -m)
  echo "Python environment: $VIRTUAL_ENV ($PYTHON_BIN; direct invocation)"
else
  PYTHON_RUN=(uv run --project "$PROJECT_DIR" --extra api --extra cv python -m)
  echo "Python environment: project default (.venv)"
fi

echo "Project:       $PROJECT_DIR"
echo "Signal store:  $STORE_DIR"
echo "Real noise:    $REAL_NOISE_DIR"
echo "Run output:    $OUTPUT_ROOT"
if [[ "$RESUME_SCENE_BUILD" == 1 ]]; then
  echo "Resuming the existing scene build."
fi
echo

echo "[1/3] Building 10-second mixed scenes with reviewed real noise"
"${PYTHON_RUN[@]}" sigid.cv_scene_corpus \
  --data-dir "$STORE_DIR" \
  --noise-data-dir "$REAL_NOISE_DIR" \
  --output-dir "$SCENES_DIR" \
  --seed "$SEED" \
  --noise-only-examples "$NOISE_ONLY_EXAMPLES"

echo "[2/3] Exporting reviewed 10-second CV training windows"
"${PYTHON_RUN[@]}" sigid.cv_dataset \
  --data-dir "$STORE_DIR" \
  --also-data-dir "$REAL_NOISE_DIR" \
  --output "$DATASET_DIR" \
  --seed "$SEED"

echo "[3/3] Training the CV detector"
"${PYTHON_RUN[@]}" sigid.cv_training \
  --dataset "$DATASET_DIR" \
  --output-model "$MODEL_PATH" \
  --base-model "$BASE_MODEL" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --seed "$SEED"

echo
echo "CV training complete."
echo "Dataset: $DATASET_DIR"
echo "Model:   $MODEL_PATH"
echo "Report:  $DATASET_DIR/export-report.json"
