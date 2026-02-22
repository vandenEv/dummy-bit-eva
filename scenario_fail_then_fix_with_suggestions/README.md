# Scenario: Fail Then Fix With Suggestions

## What This Demonstrates

This scenario shows the **full remediation workflow**:

1. A PR is opened with the "before" state — wasteful patterns cause the carbon-check gate to fail.
2. The bot leaves inline suggestions on the PR.
3. The developer applies **three progressive commits**, each reducing compute.
4. After the third commit, the gate passes.

## Project Structure

```
scenario_fail_then_fix_with_suggestions/
├── configs/
│   ├── config_before.yaml   # BEFORE: heavy config (used by src/)
│   └── config_after.yaml    # AFTER: optimised config (used by src_fixed/)
├── scripts/
│   ├── run_before.sh
│   └── run_after.sh
├── src/                     # BEFORE — wasteful patterns
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── eval.py
│   └── utils.py
├── src_fixed/               # AFTER — all three remediations applied
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── eval.py
│   └── utils.py
├── tests/
│   ├── test_config.py
│   └── test_train_step.py
└── requirements.txt
```

## Three Remediations (Commit-by-Commit)

### Commit 1 — Early stopping + reduce max epochs
```yaml
# config_before.yaml change:
epochs: 40 → epochs: 10
early_stopping: false → early_stopping: true
patience: (add) 3
```
File changed: `configs/config_before.yaml`, `src/train.py`
Commit message: `fix(carbon): enable early stopping, reduce max epochs`

### Commit 2 — Reduce model size
```yaml
# config_before.yaml change:
hidden_dim: 256 → hidden_dim: 128
num_layers: 6 → num_layers: 3
```
File changed: `configs/config_before.yaml`
Commit message: `fix(carbon): reduce model width and depth`

### Commit 3 — Cache preprocessing + FAST_DEV_RUN
```yaml
# config_before.yaml change:
use_cache: false → use_cache: true
fast_dev_run: false → fast_dev_run: true
dataset_size: 8000 → dataset_size: 500
eval_every_n_steps: 1 → eval_every_n_steps: 50
```
Files changed: `configs/config_before.yaml`, `src/data.py`, `src/train.py`
Commit message: `fix(carbon): cache preprocessing, enable fast_dev_run for PR checks`

## Quick Start

### Run BEFORE (expect carbon gate to fail)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --config configs/config_before.yaml
```

### Run AFTER (gate passes)
```bash
python src_fixed/train.py --config configs/config_after.yaml
```

Expected runtime (after): **< 15 seconds** on CPU.

## Running Tests

```bash
pytest tests/ -v
```
