#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTIVE_ENV="${SIGID_ACTIVE_MODELS_ENV:-$ROOT/data/models/active-models.env}"
export SIGID_ACTIVE_MODELS_JSON="${SIGID_ACTIVE_MODELS_JSON:-$ROOT/data/models/active-models.json}"
if [[ ! -f "$ACTIVE_ENV" ]]; then
  echo "No active model configuration found. Run scripts/train_all_models.sh first." >&2
  exit 2
fi
# This file is written by train_all_models.sh from resolved local artifact paths.
source "$ACTIVE_ENV"
cd "$ROOT"
exec uv run --extra api --extra cv sig-id-serve "$@"
