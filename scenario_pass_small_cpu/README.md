# Scenario: Pass — Small CPU Model

## What This Demonstrates

This scenario represents a lightweight ML change that **passes the carbon-check gate** without any intervention. The model is a tiny two-layer MLP trained on a small synthetic dataset for only a handful of epochs.

The carbon-check analyzer sees:
- Small `hidden_dim` and `num_layers`
- Low `dataset_size`
- Few `epochs`
- No expensive operations (no per-step eval, no redundant recomputation)

→ Estimated emissions score is **below threshold** → PR is automatically approved.

## Project Structure

```
scenario_pass_small_cpu/
├── configs/
│   └── config.yaml       # All compute knobs live here
├── scripts/
│   └── run_train.sh      # Convenience wrapper
├── src/
│   ├── __init__.py
│   ├── data.py           # Synthetic dataset generation
│   ├── model.py          # Tiny MLP definition
│   ├── train.py          # Training loop (entry point)
│   ├── eval.py           # Evaluation helpers
│   └── utils.py          # Seed, logging, config loading
├── tests/
│   ├── test_config.py
│   └── test_train_step.py
└── requirements.txt
```

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training
python src/train.py --config configs/config.yaml
```

Expected runtime: **< 10 seconds** on any modern laptop CPU.

## Key Config Values

```yaml
epochs: 5
batch_size: 32
hidden_dim: 64
num_layers: 2
dataset_size: 500
eval_every_n_steps: 50
```

## Running Tests

```bash
pytest tests/ -v
```
