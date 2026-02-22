#!/usr/bin/env bash
# run_before.sh — run the BEFORE (wasteful) version
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
[ -f "$REPO_ROOT/.venv/bin/activate" ] && source "$REPO_ROOT/.venv/bin/activate"
echo "[run_before.sh] Running BEFORE state — expects carbon gate FAILURE."
python "$REPO_ROOT/src/train.py" --config "$REPO_ROOT/configs/config_before.yaml"
