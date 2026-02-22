#!/usr/bin/env bash
# run_after.sh — run the AFTER (fixed) version
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
[ -f "$REPO_ROOT/.venv/bin/activate" ] && source "$REPO_ROOT/.venv/bin/activate"
echo "[run_after.sh] Running AFTER state — expects carbon gate PASS."
python "$REPO_ROOT/src_fixed/train.py" --config "$REPO_ROOT/configs/config_after.yaml"
