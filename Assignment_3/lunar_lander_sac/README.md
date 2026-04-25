# DA6400 PA3 – Q2.2 Lunar Lander (SAC)

## Directory Structure

```
lunar_lander_sac/
├── agents/
│   ├── sac.py          # Continuous SAC + Discrete SAC
│   └── dqn.py          # DQN for comparison (Q2.2.4)
├── envs/
│   └── lunar_lander.py # Env wrappers (standard + hover variants)
├── utils/
│   ├── trainer.py      # Training loop, multi-seed runner
│   └── plotting.py     # Plotting helpers
├── scripts/
│   ├── run_2_2_1_continuous.py   # Q2.2 Parts 1 & 2
│   ├── run_2_2_3_hover.py        # Q2.2 Part 3
│   └── run_2_2_4_discrete.py     # Q2.2 Part 4
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

If `box2d-py` fails, install SWIG first:
```bash
sudo apt-get install swig        # Linux
brew install swig                 # macOS
pip install gymnasium[box2d]
```

## Running Experiments

### Q2.2 Parts 1 & 2 — Continuous SAC
```bash
python scripts/run_2_2_1_continuous.py
```

### Q2.2 Part 3 — Hover Reward (fixed vs auto alpha, reward switching)
```bash
python scripts/run_2_2_3_hover.py
```

### Q2.2 Part 4 — Discrete SAC vs DQN
```bash
python scripts/run_2_2_4_discrete.py
```

Logs are saved to `logs/` as JSON. Plots are saved as PNG inside each log subdirectory.

## Notes
- All experiments run 15 seeds.
- Evaluation: 20 offline episodes every 10K environment timesteps.
- X-axis is always environment timesteps (never episodes).
- Random exploration phase: first 10K steps use uniform random actions.
