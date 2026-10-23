#!/usr/bin/env bash
# Build an isolated synthetic corpus, train the modulation-method baseline,
# and measure sensitivity across lineage-safe split seeds.
#
# Usage:
#   ./build_classifier_corpus.sh
#   SIGID_EXAMPLES_PER_MODE=40 SIGID_RUN_DIR=data/classifier-runs/large-40 ./build_classifier_corpus.sh
#
# Generated captures use a run-local SQLite database under data/ so this does
# not add synthetic rows to the normal live workbench database.

set -Eeuo pipefail

SIGID_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIGID_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
SIGID_RUN_DIR="${SIGID_RUN_DIR:-$SIGID_ROOT/data/classifier-runs/$SIGID_RUN_ID}"
if [[ "$SIGID_RUN_DIR" != /* ]]; then
  SIGID_RUN_DIR="$SIGID_ROOT/$SIGID_RUN_DIR"
fi

SIGID_DATA_DIR="$SIGID_RUN_DIR/dataset"
SIGID_KNOWN_DIR="$SIGID_RUN_DIR/known"
SIGID_OOD_DIR="$SIGID_RUN_DIR/ood"
SIGID_MODEL_DIR="$SIGID_RUN_DIR/models"
SIGID_MODEL_PATH="$SIGID_MODEL_DIR/method.npz"
SIGID_SPLIT_PATH="$SIGID_RUN_DIR/splits.json"

SIGID_EXAMPLES_PER_MODE="${SIGID_EXAMPLES_PER_MODE:-24}"
SIGID_SNR_LEVELS_DB="${SIGID_SNR_LEVELS_DB:-0 3 6 10 20}"
SIGID_OOD_EXAMPLES="${SIGID_OOD_EXAMPLES:-30}"
SIGID_SEED="${SIGID_SEED:-20261023}"
SIGID_OOD_SEED="${SIGID_OOD_SEED:-1717}"
SIGID_EPOCHS="${SIGID_EPOCHS:-1200}"
SIGID_EVALUATION_SEEDS="${SIGID_EVALUATION_SEEDS:-11 23 37 51 79}"
SIGID_FEATURE_VERSION="${SIGID_FEATURE_VERSION:-passband-log-spectrum-envelope-v2}"

if [[ -x "$SIGID_ROOT/.venv/bin/python" ]]; then
  SIGID_PYTHON="${SIGID_PYTHON:-$SIGID_ROOT/.venv/bin/python}"
else
  SIGID_PYTHON="${SIGID_PYTHON:-$(command -v python3 || true)}"
fi
if [[ -z "$SIGID_PYTHON" || ! -x "$SIGID_PYTHON" ]]; then
  echo "Python was not found. Create the project .venv or set SIGID_PYTHON." >&2
  exit 2
fi
if ! "$SIGID_PYTHON" -c 'import numpy, scipy' >/dev/null 2>&1; then
  echo "The selected Python needs NumPy and SciPy. Install the project API dependencies first." >&2
  exit 2
fi

if [[ -e "$SIGID_MODEL_PATH" || -e "$SIGID_MODEL_PATH.json" || -e "$SIGID_MODEL_DIR/evaluation.json" ]]; then
  echo "A model already exists under $SIGID_MODEL_DIR. Choose a fresh SIGID_RUN_DIR to avoid replacing it." >&2
  exit 2
fi

read -r -a SIGID_SNR_ARGS <<< "$SIGID_SNR_LEVELS_DB"
if [[ "${#SIGID_SNR_ARGS[@]}" -eq 0 ]]; then
  echo "SIGID_SNR_LEVELS_DB must contain at least one SNR value." >&2
  exit 2
fi
read -r -a SIGID_EVAL_SEED_ARGS <<< "$SIGID_EVALUATION_SEEDS"

mkdir -p "$SIGID_RUN_DIR" "$SIGID_MODEL_DIR"
cd "$SIGID_ROOT"
export SIG_ID_DATA_DIR="$SIGID_DATA_DIR"

cat > "$SIGID_RUN_DIR/run-config.txt" <<EOF
examples_per_mode=$SIGID_EXAMPLES_PER_MODE
snr_levels_db=$SIGID_SNR_LEVELS_DB
ood_examples=$SIGID_OOD_EXAMPLES
corpus_seed=$SIGID_SEED
ood_seed=$SIGID_OOD_SEED
epochs=$SIGID_EPOCHS
feature_version=$SIGID_FEATURE_VERSION
evaluation_seeds=$SIGID_EVALUATION_SEEDS
EOF

echo "Run directory: $SIGID_RUN_DIR"
echo "Building known-method corpus in isolated data directory: $SIGID_DATA_DIR"
"$SIGID_PYTHON" -m sigid.corpus \
  --output-dir "$SIGID_KNOWN_DIR" \
  --data-dir "$SIGID_DATA_DIR" \
  --examples-per-mode "$SIGID_EXAMPLES_PER_MODE" \
  --snr-db "${SIGID_SNR_ARGS[@]}" \
  --seed "$SIGID_SEED"

echo "Creating separate unknown-family calibration/test corpus"
"$SIGID_PYTHON" -m sigid.ood_corpus \
  --output-dir "$SIGID_OOD_DIR" \
  --examples "$SIGID_OOD_EXAMPLES" \
  --seed "$SIGID_OOD_SEED"

echo "Creating lineage-safe train/validation/test split"
"$SIGID_PYTHON" -m sigid.splits \
  --corpus "$SIGID_KNOWN_DIR/corpus.json" \
  --data-dir "$SIGID_DATA_DIR" \
  --output "$SIGID_SPLIT_PATH" \
  --seed "$SIGID_SEED" \
  --assignment-version 2

echo "Training and evaluating the primary method model"
"$SIGID_PYTHON" -m sigid.training \
  --split "$SIGID_SPLIT_PATH" \
  --data-dir "$SIGID_DATA_DIR" \
  --ood-corpus "$SIGID_OOD_DIR/ood-corpus.json" \
  --output-model "$SIGID_MODEL_PATH" \
  --epochs "$SIGID_EPOCHS" \
  --feature-version "$SIGID_FEATURE_VERSION"

if [[ "${#SIGID_EVAL_SEED_ARGS[@]}" -gt 0 ]]; then
  echo "Evaluating sensitivity across lineage-safe split seeds"
  "$SIGID_PYTHON" -m sigid.seed_evaluation \
    --corpus "$SIGID_KNOWN_DIR/corpus.json" \
    --data-dir "$SIGID_DATA_DIR" \
    --output-dir "$SIGID_RUN_DIR/split-seed-evaluation" \
    --seeds "${SIGID_EVAL_SEED_ARGS[@]}" \
    --epochs "$SIGID_EPOCHS" \
    --ood-corpus "$SIGID_OOD_DIR/ood-corpus.json"
fi

echo
echo "Training complete."
echo "Model: $SIGID_MODEL_PATH"
echo "Single-split evaluation: $SIGID_MODEL_DIR/evaluation.json"
if [[ "${#SIGID_EVAL_SEED_ARGS[@]}" -gt 0 ]]; then
  echo "Split-seed report: $SIGID_RUN_DIR/split-seed-evaluation/multi-seed-evaluation.json"
fi
echo "Set SIG_ID_MODEL_PATH to the model path to load it in the workbench."
