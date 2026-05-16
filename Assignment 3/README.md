# DA6400 — Programming Assignment 3

**Authors**: Yash Purswani (ME22B214), Govind S Ashan (ME23B168), Tiramdas Abhinand (ME23B208)

--- 

Soft Actor-Critic (SAC) on Pendulum, LunarLander, and Reacher, plus PEBBLE
(preference-based RL) on Pendulum and Reacher.

## Folder Structure

```
.
├── Assignment 3.pdf          # Problem statement
├── README.md                 # (this file)
├── requirements.txt          # Python dependencies
│
├── Pendulum/                 # Q2.1 — SAC on modified Pendulum-v1
│   ├── train_pendulum.py     # Main entry point (single run)
│   ├── agent/                # Actor, critic, SAC core
│   ├── envs/                 # Pendulum-with-target-angle wrapper
│   ├── utils/                # Replay buffer, evaluator, logger
│   ├── runs/                 # Multi-seed launchers (sequential + parallel)
│   ├── plotting/             # Plot scripts for Q2.1
│   ├── logs/                 # Training JSON logs
│   └── plots/                # Generated figures
│
├── Lunar Lander/             # Q2.2 — Continuous + discrete SAC, hover variant
│   ├── agents/               # SAC (cont + discrete), DQN
│   ├── envs/                 # LunarLander wrappers (standard / hover)
│   ├── utils/                # Trainer, multi-seed runner, plotting
│   ├── scripts/              # Per-subquestion launchers
│   ├── logs/                 # Per-experiment JSON + PNG
│   └── visualize_policy.py
│
├── Reacher/                  # Q2.3 — SAC on Reacher-Easy with Ra/Rb/Rc rewards
│   ├── sac_reacher.py        # SAC + Reacher env + reward functions
│   ├── run_experiments.py    # All seeds × all rewards
│   ├── plot_results.py       # All plots (Q2.3.2, .3a, .3c)
│   ├── results/              # Per-seed JSON logs
│   └── plots/                # Generated figures
│
└── Pebble/                   # Q3 (Bonus) — PEBBLE on Pendulum + Reacher
    ├── agents/               # SAC, DQN, PEBBLE reward model, trainer
    ├── envs/                 # Env factories
    ├── utils/                # Trainer + plotting
    ├── scripts/
    │   ├── run_q3_pendulum.py     # Q3.1, Q3.2 — Pendulum PEBBLE + budget sweep
    │   ├── run_q3_reacher.py      # Q3.3 — Reacher PEBBLE with 3 teachers
    │   └── visualise_pendulum.py  # Optional: render a saved policy
    └── logs/                 # q3_pebble_pendulum/, q3_pebble_reacher/
```

## Problem Description

- **Q2.1 Pendulum (`Pendulum/`)** — SAC with automated *and* manual temperature
  tuning on a modified Pendulum where the goal is to hold the arm at a target
  angle θ\_target ∈ {0, −10, 30, −60, 90, −90, 120, −150}. Includes reward-scale
  ablations (×0.1, ×10) comparing auto-α vs manual-α.

- **Q2.2 Lunar Lander (`Lunar Lander/`)** —
  1. Continuous SAC with auto-α; analysis of early/intermediate/final policy.
  2. Modified hover-bonus variant ( +200 → −100 reward switch ) comparing
     fixed-α and auto-α: probes the max-entropy formulation.
  3. Discrete SAC vs DQN on the discrete-action version.

- **Q2.3 Reacher (`Reacher/`)** — SAC on Reacher-Easy under three reward
  formulations Rₐ (shaped), R_b (binary in-target), R_c (constant −1 until
  termination). Includes self-evaluation, cross-evaluation, and a steps-to-goal
  / steps-in-target final-policy analysis.

- **Q3 Bonus (`Pebble/`)** —
  1. PEBBLE on Pendulum vs SAC trained on ground truth (Q3.1).
  2. Feedback-budget ablation on Pendulum (Q3.2).
  3. PEBBLE on Reacher with three simulated teachers, one per reward formulation
     (Q3.3).

## Setup

```bash
# Python 3.10 recommended
conda create -n rl_pa3 python=3.10 -y
conda activate rl_pa3
pip install -r requirements.txt
```

`dm_control` (Reacher) needs MuJoCo. On headless machines set
`export MUJOCO_GL=egl` (or `osmesa`) before running anything that creates
Reacher envs.

## How to run

Run each block from inside its own subdirectory.

### Q2.1 Pendulum
```bash
cd Pendulum
python train_pendulum.py                               # single config
python runs/run_parallel_pendulum.py                   # all θ_target, auto-α
python runs/run_parallel_pendulum_manual.py            # manual-α sweep
python plotting/plot_pendulum.py                       # auto-α plots
python plotting/plot_pendulum_manual.py                # manual-α plots
python plotting/plot_pendulum_manual_auto.py           # manual-vs-auto comparison
```

### Q2.2 Lunar Lander
```bash
cd "Lunar Lander"
python scripts/run_2_2_1_continuous.py    # Q2.2 parts 1 & 2
python scripts/run_2_2_3_hover.py         # Q2.2 part 3 (hover variant, reward switch)
python scripts/run_2_2_4_discrete.py      # Q2.2 part 4 (discrete-SAC vs DQN)
```
Per-experiment JSON logs + PNGs land in `logs/<exp_name>/`.

### Q2.3 Reacher
```bash
cd Reacher
python sac_reacher.py --reward rb --seed 0 --steps 500000   # quick single run
python run_experiments.py                                   # all rewards × all seeds (sequential)
python run_experiments.py --parallel                        # one process per seed
python plot_results.py --result_dir results --plot_dir plots
```
Produces `plots/fig_A_self_eval_r{a,b,c}.png`, `plots/fig_B_bar_chart.png`,
`plots/fig_C_cross_eval_against_r{a,b,c}.png`.

### Q3 Bonus (PEBBLE)
```bash
cd Pebble
python scripts/run_q3_pendulum.py        # PEBBLE on Pendulum + budget ablation
python scripts/run_q3_reacher.py         # PEBBLE on Reacher (3 teachers, 5-seed parallel)
# Optional: render a saved policy
python scripts/visualise_pendulum.py --weights <path-to.pt> --theta 90
```
Plots land under `logs/<experiment>/plots/`.

## Notes

- All experiments use **15 seeds** unless stated otherwise.
- Evaluation: 20 offline episodes every 10K env steps. X-axis is always
  environment timesteps (never episodes).
- SAC starts with a 10K-step uniform-random exploration phase.
- Reacher requires the `dm_control` (DeepMind Control Suite) `reacher_easy`
  task.
