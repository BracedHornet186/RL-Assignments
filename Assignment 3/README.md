# DA6400 PA3 — Section 2.3: Reacher (SAC)

## Setup

```bash
pip install -r requirements.txt
# or with conda:
conda create -n rl_pa3 python=3.10
conda activate rl_pa3
pip install -r requirements.txt
```

## Run Experiments

### Single run (for testing):
```bash
python sac_reacher.py --reward rb --seed 0 --steps 500000
```

### All 15 seeds for one reward type:
```bash
python run_experiments.py --reward ra
python run_experiments.py --reward rb
python run_experiments.py --reward rc
```

### All experiments (sequential):
```bash
python run_experiments.py
```

### All experiments (parallel, one process per seed):
```bash
python run_experiments.py --parallel
```

## Generate Plots
```bash
python plot_results.py --result_dir results --plot_dir plots
```

Produces:
- `plots/fig_A_self_eval_r{a,b,c}.png`  — Q2.3.2
- `plots/fig_B_bar_chart.png`            — Q2.3.3a
- `plots/fig_C_cross_eval_against_r{a,b,c}.png` — Q2.3.3c

## File Overview
| File | Purpose |
|------|---------|
| `sac_reacher.py` | Core SAC + Reacher env + reward functions |
| `run_experiments.py` | Launches training for all seeds/rewards |
| `plot_results.py` | All plotting for Q2.3.2, Q2.3.3a, Q2.3.3c |
| `requirements.txt` | Python dependencies |
