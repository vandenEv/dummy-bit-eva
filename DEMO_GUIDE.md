# Carbon-Check Demo Guide

This guide explains how to use the four demo scenarios to showcase the **carbon-check** GitHub Action in a live GitHub PR demo. Each scenario lives in its own folder and is designed to produce a distinct emissions signature.

---

## System Overview

The `carbon-gate.yml` workflow runs on every PR. It:
1. Scans changed files for compute-intensity signals (epochs, batch size, model depth, dataset size, nested loops).
2. Estimates a relative emissions score.
3. **Blocks the PR** if the score exceeds the threshold — unless the author either applies a suggested remediation or an authorized maintainer uses a permission-based override.

---

## Scenario Summaries

| Scenario | Expected Result | Key Signal |
|---|---|---|
| `scenario_pass_small_cpu` | ✅ Passes gate | Tiny model, few epochs, small dataset |
| `scenario_fail_big_gpu_like` | ❌ Fails gate | Deep transformer, large batch, huge dataset |
| `scenario_fail_then_fix_with_suggestions` | ❌ Fails → ✅ Passes after fixes | Shows 3 progressive remediations |
| `scenario_fail_override_permission_only` | ❌ Fails, override by maintainer | Heavy config, no code fix, role-based bypass |

---

## Demo Flow: Scenario 1 — `scenario_pass_small_cpu`

### What it demonstrates
A small PR that adds a lightweight MLP for a CPU-only workload. The analyzer sees low compute and lets it through immediately.

### How to trigger
1. Create a PR that modifies any file inside `scenario_pass_small_cpu/`.
2. The carbon-check workflow runs and scores the change as **below threshold**.
3. PR is automatically approved by the gate.

### Key config values that keep it green
```yaml
# configs/config.yaml
epochs: 5
batch_size: 32
hidden_dim: 64
num_layers: 2
dataset_size: 500
```

### Local run
```bash
cd scenario_pass_small_cpu
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --config configs/config.yaml
```

---

## Demo Flow: Scenario 2 — `scenario_fail_big_gpu_like`

### What it demonstrates
A PR that introduces a wide transformer-style model with a large synthetic dataset, many epochs, and no optimizations. The gate fires and blocks the merge.

### How to trigger
1. Create a PR modifying files in `scenario_fail_big_gpu_like/`.
2. Carbon-check detects the heavy config and deep model → **exceeds threshold**.
3. The PR is **blocked** with a comment listing the top contributing factors.

### Key wasteful patterns
- `hidden_dim: 512`, `num_layers: 8`, `epochs: 50`
- `dataset_size: 10000`
- Evaluation loop runs every single step (no interval)
- No mixed precision or gradient accumulation
- Redundant preprocessing recomputed each epoch

### What the failure comment highlights
```
Estimated relative compute score: HIGH
Triggers:
  - Large model depth (num_layers=8, hidden_dim=512)
  - No early stopping
  - Per-step evaluation enabled
  - Dataset size: 10000 samples
Suggestion: See scenario_fail_then_fix_with_suggestions for remediation patterns.
```

### Local run
```bash
cd scenario_fail_big_gpu_like
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --config configs/config.yaml
```

---

## Demo Flow: Scenario 3 — `scenario_fail_then_fix_with_suggestions`

### What it demonstrates
A PR starts in a failing state. The carbon-check bot leaves inline suggestions. The developer applies them as separate commits. After each commit the score drops. After all three remediations, the PR passes.

### Before state (initial PR commit)
Files in `src/` — wasteful patterns identical to scenario 2.

```yaml
# configs/config_before.yaml
epochs: 40
batch_size: 256
hidden_dim: 256
num_layers: 6
dataset_size: 8000
eval_every_n_steps: 1      # wasteful
use_cache: false           # wasteful
fast_dev_run: false
```

### Remediation 1 — Early stopping & fewer epochs
**Changed file:** `configs/config_before.yaml` → set `epochs: 10`, add `early_stopping: true`, `patience: 3`
**Also changed:** `src/train.py` — enable early stopping loop

Commit message: `fix(carbon): enable early stopping, reduce max epochs`

### Remediation 2 — Reduce model size
**Changed file:** `configs/config_before.yaml` → set `hidden_dim: 128`, `num_layers: 3`
**Also changed:** `src/model.py` — no structural change needed, config-driven

Commit message: `fix(carbon): reduce model width and depth`

### Remediation 3 — Cache preprocessing + FAST_DEV_RUN for CI
**Changed file:** `configs/config_before.yaml` → set `use_cache: true`, `fast_dev_run: true`, `dataset_size: 500`
**Also changed:** `src/data.py` — add caching path, `src/train.py` — respect `fast_dev_run`

Commit message: `fix(carbon): cache preprocessing, enable fast_dev_run for PR checks`

### After state
Files in `src_fixed/` — all three remediations applied; config is `configs/config_after.yaml`.

```yaml
# configs/config_after.yaml
epochs: 10
batch_size: 64
hidden_dim: 128
num_layers: 3
dataset_size: 500
eval_every_n_steps: 50
use_cache: true
fast_dev_run: true
early_stopping: true
patience: 3
```

### Local run (before)
```bash
cd scenario_fail_then_fix_with_suggestions
python src/train.py --config configs/config_before.yaml
```

### Local run (after)
```bash
python src_fixed/train.py --config configs/config_after.yaml
```

---

## Demo Flow: Scenario 4 — `scenario_fail_override_permission_only`

### What it demonstrates
A PR that fails the carbon gate and has **no code fix**. A maintainer (org member with `write` or `admin` role) bypasses the gate using a permission-based override comment. Contributors without that role cannot override.

### How to trigger the failure
1. Create a PR modifying files in `scenario_fail_override_permission_only/`.
2. Gate fires — PR is blocked.

### How the override works (demo narration only)
- A **maintainer** comments `/carbon-override reason: "scheduled batch job, not interactive"` on the PR.
- The workflow detects the comment, checks the commenter's repository role via the GitHub API, and if they have `write` or `admin` permissions, it marks the gate as passing.
- A **regular contributor** posting the same comment gets a rejection: `Override denied: insufficient permissions`.

### Config flag (demo artifact)
```python
# src/train.py — line near top
ALLOW_OVERRIDE_DEMO = os.getenv("ALLOW_OVERRIDE_DEMO", "false").lower() == "true"
```
This flag does **not** grant any actual override. It only exists for demo narration — the presenter can highlight it to explain that the real gate enforcement lives in the GitHub Actions workflow, not in the application code.

### Local run
```bash
cd scenario_fail_override_permission_only
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --config configs/config.yaml
```

---

## Tests

Each scenario has a `tests/` folder with:
- `test_config.py` — verifies YAML config loads and required keys are present.
- `test_train_step.py` — runs exactly one training step to validate no import or shape errors.

Run all tests from any scenario folder:
```bash
pytest tests/ -v
```

---

## Tips for Live Demo

1. **Use a staging fork** — keep `main` clean; demo on feature branches.
2. **Pre-bake the commits** — stage each remediation commit in advance so the demo is fast.
3. **Show the workflow logs** — the carbon-check action output is the centerpiece; have the Actions tab open.
4. **Contrast scenario 1 vs 2 side-by-side** — the config diff alone tells the story before any code runs.
5. **For scenario 4** — have two browser tabs logged in as different users to show the permission difference live.
