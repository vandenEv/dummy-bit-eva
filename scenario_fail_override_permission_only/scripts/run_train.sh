#!/usr/bin/env bash
# run_train.sh — scenario_fail_override_permission_only
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
[ -f "$REPO_ROOT/.venv/bin/activate" ] && source "$REPO_ROOT/.venv/bin/activate"
echo "[run_train.sh] WARNING: This scenario is heavy and WILL fail the carbon gate."
echo "[run_train.sh] It is designed for override demonstrations only."
python "$REPO_ROOT/src/train.py" --config "$REPO_ROOT/configs/config.yaml"
