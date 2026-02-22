# Scenario: Fail — Override (Permission-Based Only)

> **This scenario is for demonstrating permission-based overrides for maintainers only.**
> Contributors without `write` or `admin` role on the repository cannot bypass the gate.

## What This Demonstrates

A PR that **fails the carbon-check gate** and **stays failed from a code perspective**. No remediation code is applied. Instead, an authorized maintainer uses a repository-permission-based override comment to bypass the gate for this specific PR.

This demonstrates that the system has two distinct approval paths:
1. **Code fix** — developer reduces compute, gate passes on next commit.
2. **Permission override** — maintainer acknowledges the cost and explicitly approves despite the flag.

## Override Flow

1. PR is opened touching files in this directory → gate fires → PR is blocked.
2. Maintainer (repo `write` or `admin` role) comments:
   ```
   /carbon-override reason: "scheduled nightly job, not a hot-path PR"
   ```
3. The `carbon-gate.yml` workflow picks up the comment event, calls the GitHub API to check the commenter's repository permissions, and — if authorized — marks the gate status as passing.
4. A contributor without elevated permissions posting the same comment receives:
   ```
   Override denied: your repository role (read) is insufficient.
   Minimum required: write.
   ```

## `ALLOW_OVERRIDE_DEMO` Flag

```python
# src/train.py — top of file
ALLOW_OVERRIDE_DEMO = os.getenv("ALLOW_OVERRIDE_DEMO", "false").lower() == "true"
```

**This flag does NOT grant any override.** It is a demo narration aid only.
The presenter can highlight it to show that override enforcement lives entirely
in the GitHub Actions workflow (via the GitHub API), not in the Python code.
Setting `ALLOW_OVERRIDE_DEMO=true` in your shell changes nothing about PR gating.

## Project Structure

```
scenario_fail_override_permission_only/
├── configs/
│   └── config.yaml          # Heavy config; intentionally kept as-is
├── scripts/
│   └── run_train.sh
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── train.py             # Contains ALLOW_OVERRIDE_DEMO flag
│   ├── eval.py
│   └── utils.py
├── tests/
│   ├── test_config.py
│   └── test_train_step.py
└── requirements.txt
```

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --config configs/config.yaml
```

## Override Demo (local simulation)

```bash
# This does NOT change gate behaviour — demo narration only.
ALLOW_OVERRIDE_DEMO=true python src/train.py --config configs/config.yaml
```

Expected output includes a line:
```
[INFO] ALLOW_OVERRIDE_DEMO=True — demo flag is set (enforcement is in the workflow, not here).
```

## Running Tests

```bash
pytest tests/ -v
```
