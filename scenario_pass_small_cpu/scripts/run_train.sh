#!/usr/bin/env bash
# run_train.sh — convenience wrapper for scenario_pass_small_cpu
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "[run_train.sh] Activating virtual environment..."
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

echo "[run_train.sh] Starting training..."
python "$REPO_ROOT/src/train.py" --config "$REPO_ROOT/configs/config.yaml"
echo "[run_train.sh] Done."
