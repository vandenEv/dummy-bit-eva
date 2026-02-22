#!/usr/bin/env bash
# run_train.sh — convenience wrapper for scenario_fail_big_gpu_like
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "[run_train.sh] WARNING: This config is intentionally heavy and will fail the carbon gate."
echo "[run_train.sh] Starting training..."

if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

python "$REPO_ROOT/src/train.py" --config "$REPO_ROOT/configs/config.yaml"
echo "[run_train.sh] Done."
