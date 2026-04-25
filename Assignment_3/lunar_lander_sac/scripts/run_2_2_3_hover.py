"""
Q2.2 Part 3
Hover-reward LunarLander experiment — runs seeds IN PARALLEL.
  - fixed alpha = 0.01
  - auto alpha
Both trained on:
  Phase 1: hover bonus = +200
  Phase 2 (after switch_step): hover bonus = -100
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────
SEEDS        = list(range(15))
TOTAL_STEPS  = 500_000
SWITCH_STEP  = 250_000
EVAL_EVERY   = 10_000
EVAL_EPS     = 20
RANDOM_STEPS = 10_000
LOG_DIR      = "logs/q2_2_3_hover"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Hard-code dims to avoid importing gymnasium at module level
# (gymnasium triggers OpenGL/CUDA init in every spawned subprocess)
OBS_DIM    = 8
ACTION_DIM = 2


# ── Env factories ────────────────────────────────────────────────
def make_changing_train_env(seed):
    from envs.lunar_lander import ChangingRewardLunarLander
    env = ChangingRewardLunarLander(switch_step=SWITCH_STEP, continuous=True)
    env.reset(seed=seed)
    _PROCESS_ENV[seed] = env
    return env

def make_hover_eval_env():
    from envs.lunar_lander import HoverLunarLander
    return HoverLunarLander(hover_bonus=200.0, continuous=True)


# ── Agent factories ──────────────────────────────────────────────
def make_fixed_alpha_agent(seed):
    from agents.sac import SAC
    return SAC(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        lr=3e-4, gamma=0.99, tau=0.005,
        batch_size=256, buffer_size=600_000,
        hidden=(256, 256),
        auto_alpha=False, alpha=0.01,
        device=DEVICE,
    )

def make_auto_alpha_agent(seed):
    from agents.sac import SAC
    return SAC(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        lr=3e-4, gamma=0.99, tau=0.005,
        batch_size=256, buffer_size=600_000,
        hidden=(256, 256),
        auto_alpha=True,
        device=DEVICE,
    )


# ── Picklable hook ───────────────────────────────────────────────
_PROCESS_ENV = {}

class HoverHook:
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, step):
        env = _PROCESS_ENV.get(self.seed)
        if env is not None:
            env.set_global_step(step)

def make_hook_fn(seed):
    return HoverHook(seed)


if __name__ == "__main__":
    from utils.trainer import run_seeds
    from utils.plotting import plot_curves
    print(f"Using device: {DEVICE}")

    # ── Fixed alpha ──────────────────────────────────────────────
    print("=== Fixed alpha = 0.01 ===")
    ts_fixed, mean_fixed, std_fixed = run_seeds(
        agent_fn         = make_fixed_alpha_agent,
        train_env_fn     = make_changing_train_env,
        eval_env_fn      = make_hover_eval_env,
        seeds            = SEEDS,
        total_steps      = TOTAL_STEPS,
        eval_every       = EVAL_EVERY,
        eval_episodes    = EVAL_EPS,
        random_steps     = RANDOM_STEPS,
        log_dir          = LOG_DIR,
        run_prefix       = "fixed_alpha",
        env_step_hook_fn = make_hook_fn,
        n_workers        = None,
    )

    # ── Auto alpha ───────────────────────────────────────────────
    print("=== Auto alpha ===")
    ts_auto, mean_auto, std_auto = run_seeds(
        agent_fn         = make_auto_alpha_agent,
        train_env_fn     = make_changing_train_env,
        eval_env_fn      = make_hover_eval_env,
        seeds            = SEEDS,
        total_steps      = TOTAL_STEPS,
        eval_every       = EVAL_EVERY,
        eval_episodes    = EVAL_EPS,
        random_steps     = RANDOM_STEPS,
        log_dir          = LOG_DIR,
        run_prefix       = "auto_alpha",
        env_step_hook_fn = make_hook_fn,
        n_workers        = None,
    )

    # ── Plot ─────────────────────────────────────────────────────
    fig, ax = plot_curves(
        [
            {"label": "Fixed α=0.01", "timesteps": ts_fixed, "mean": mean_fixed, "std": std_fixed},
            {"label": "Auto α",       "timesteps": ts_auto,  "mean": mean_auto,  "std": std_auto},
        ],
        title     = "LunarLander Hover Reward: +200 → -100 at 250K steps",
        save_path = f"{LOG_DIR}/plots/hover_comparison.png",
    )
    ax.axvline(x=SWITCH_STEP, color="black", linestyle="--", linewidth=1.5, label="Reward switch")
    ax.legend()
    fig.savefig(f"{LOG_DIR}/plots/hover_comparison.png", dpi=150)
    print("Done.")