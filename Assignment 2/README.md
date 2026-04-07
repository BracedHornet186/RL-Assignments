# RL Assignment 2 — Experiment Run Guide

## Authors
- Yash Purswani (ME22B214)
- Govind S Ashan (ME23B168)
- Abhinand T (ME23B208)
---
This README gives the **correct commands** to run your experiments for:
- Q1: `vanilla_dqn_parallel.py` (+ rendering with `vanilla_dqn_eval.py`)
- Q3: `plot_compare_truncation.py`
- Q4: `hyperparameter_senstivity.py`
- Q5: `dqn_per.py` and `per_plot.py`

> Some usage blocks inside scripts mention old/wrong filenames (for example `dqn_mountaincar.py`, `evaluate.py`, `plot_truncation.py`, `plot_rho_comparison.py`).
> Use the commands in this README.

---

## Setup


Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Q1 — Vanilla DQN training (`vanilla_dqn_parallel.py`)

### Main training run (15 seeds, parallel workers)

```bash
python3 vanilla_dqn_parallel.py --truncation 2000 --replay_factor 1 --workers 8 --log_dir logs
```

### CPU-only run

```bash
python3 vanilla_dqn_parallel.py --device cpu --workers 8 --truncation 2000 --replay_factor 1 --log_dir logs
```

### Single-seed debug

```bash
python3 vanilla_dqn_parallel.py --seeds 0 --workers 1 --truncation 2000 --replay_factor 1 --log_dir logs
```

### Outputs
- CSV logs: `logs/trunc{T}_rho{R}_seed{S}.csv`
- Weights: `logs/weights/..._best.pt` and `..._final.pt`

---

## Q1 Evaluation / Rendering (`vanilla_dqn_eval.py`)

Render a trained checkpoint:

```bash
python3 vanilla_dqn_eval.py --ckpt logs/weights/trunc2000_rho1_seed0_best.pt --episodes 5
```

Headless evaluation:

```bash
python3 vanilla_dqn_eval.py --ckpt logs/weights/trunc2000_rho1_seed0_best.pt --episodes 5 --no_render
```

Slower playback:

```bash
python3 vanilla_dqn_eval.py --ckpt logs/weights/trunc2000_rho1_seed0_best.pt --episodes 3 --slow
```

---

## Q3 — Truncation comparison plot (`plot_compare_truncation.py`)

Assumes logs exist for truncations 200, 1000, 2000 with same `rho`.

```bash
python3 plot_compare_truncation.py --log_dir logs --replay_factor 1 --save --out_dir plots
```

Optional explicit truncation list:

```bash
python3 plot_compare_truncation.py --log_dir logs --replay_factor 1 --truncations 200 1000 2000 --save
```

Output figure:
- `plots/truncation_comparison_rho1.png`

---

## Q4 — Hyperparameter sensitivity (`hyperparameter_senstivity.py`)

Run full sensitivity sweeps:

```bash
python3 hyperparameter_senstivity.py
```

Outputs:
- CSVs in `results/` (batch/target sweeps)
- Plots:
  - `results/sensitivity_batch.png`
  - `results/sensitivity_target.png`

Notes:
- This script is compute-heavy (multiple seeds × multiple settings).
- Filename is intentionally `hyperparameter_senstivity.py` (same typo as in workspace).

---

## Q5 — PER experiments (`dqn_per.py`, `per_plot.py`)

### Train with PER enabled

```bash
python3 dqn_per.py --use_per --log_dir logs_per --truncation 2000 --replay_factor 1 --max_episodes 600 --workers 4
```

For other replay factors:

```bash
python3 dqn_per.py --use_per --log_dir logs_per --truncation 2000 --replay_factor 2 --max_episodes 600 --workers 4
python3 dqn_per.py --use_per --log_dir logs_per --truncation 2000 --replay_factor 4 --max_episodes 600 --workers 4
python3 dqn_per.py --use_per --log_dir logs_per --truncation 2000 --replay_factor 8 --max_episodes 600 --workers 4
```

### (Optional) Uniform baseline from same script

```bash
python3 dqn_per.py --log_dir logs_uniform --truncation 2000 --replay_factor 1 --max_episodes 600 --workers 4
```

### Plot rho comparison for PER logs

```bash
python3 per_plot.py --log_dir logs_per --truncation 2000 --rhos 1 2 4 8 --per --save --out_dir plots
```

Output figure (PER-tagged):
- `plots/rho_comparison_per_trunc2000.png`

---

## Quick command summary

```bash
# Q1 train
python3 vanilla_dqn_parallel.py --truncation 2000 --replay_factor 1 --workers 8 --log_dir logs

# Q1 eval/render
python3 vanilla_dqn_eval.py --ckpt logs/weights/trunc2000_rho1_seed0_best.pt --episodes 5

# Q3 plot truncation comparison
python3 plot_compare_truncation.py --log_dir logs --replay_factor 1 --save

# Q4 sensitivity
python3 hyperparameter_senstivity.py

# Q5 PER training + plot
python3 dqn_per.py --use_per --log_dir logs_per --truncation 2000 --replay_factor 1 --max_episodes 600 --workers 4
python3 per_plot.py --log_dir logs_per --truncation 2000 --rhos 1 2 4 8 --per --save
```