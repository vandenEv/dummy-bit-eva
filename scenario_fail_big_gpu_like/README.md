# Scenario: Fail — Big GPU-Like Model

## What This Demonstrates

This scenario represents a PR that **fails the carbon-check gate** due to obviously excessive compute. It includes a deep transformer-style model, a large synthetic dataset, many epochs, and several wasteful patterns.

The carbon-check analyzer sees:
- Large `hidden_dim` (512) and `num_layers` (8)
- `dataset_size: 10000`
- `epochs: 50` with no early stopping
- Evaluation runs **every single step** (`eval_every_n_steps: 1`)
- Preprocessing is recomputed from scratch each epoch (no caching)
- No mixed precision; no gradient accumulation

→ Estimated emissions score is **above threshold** → PR is **blocked automatically**.

## Wasteful Patterns (Intentional for Demo)

| Pattern | Location | Why it's expensive |
|---|---|---|
| Deep transformer stack | `src/model.py` | 8 layers × 512-dim with multi-head attention |
| Per-step evaluation | `src/train.py` | Runs val loop after every single batch |
| Preprocessing in epoch loop | `src/data.py` | Redundant feature transform recomputed each epoch |
| `epochs: 50` | `configs/config.yaml` | No early stopping; always runs the full 50 |
| `dataset_size: 10000` | `configs/config.yaml` | 20× the size of the passing scenario |
| Nested augmentation loop | `src/data.py` | Applies augmentation N times per sample |

## Project Structure

```
scenario_fail_big_gpu_like/
├── configs/
│   └── config.yaml
├── scripts/
│   └── run_train.sh
├── src/
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

## Quick Start

> **Note:** Default config is intentionally heavier. The run may take 1–2 minutes on CPU.
> To make it faster for testing, reduce `epochs` and `dataset_size` in the config.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --config configs/config.yaml
```

## Key Config Values (all heavy)

```yaml
epochs: 50
batch_size: 256
hidden_dim: 512
num_layers: 8
dataset_size: 10000
eval_every_n_steps: 1    # wasteful: every step
use_cache: false          # wasteful: no caching
```

## Running Tests

```bash
pytest tests/ -v
```
