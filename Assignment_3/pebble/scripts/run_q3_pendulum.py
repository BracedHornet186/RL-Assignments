"""
Q3 (Bonus) Part 1 & 2 — PEBBLE on modified Pendulum-v1.

Part 1: Compare SAC (GT reward) vs PEBBLE for θ_target ∈ {0,-60,90,120,-150}
Part 2: Different feedback budgets for θ_target = 90
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import json
from pathlib import Path

SEEDS         = list(range(15))
TOTAL_STEPS   = 100_000
EVAL_EVERY    = 10_000
EVAL_EPS      = 20
RANDOM_STEPS  = 10_000
QUERY_EVERY   = 5_000
N_QUERIES     = 10
SEGMENT_LEN   = 50
REWARD_STEPS  = 50
LOG_DIR       = "logs/q3_pebble_pendulum"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

OBS_DIM    = 3   # Pendulum: [cos θ, sin θ, dθ/dt]
ACTION_DIM = 1

TARGET_ANGLES = [0, -60, 90, 120, -150]

# Feedback budgets for Part 2
BUDGETS = [50, 100, 200, 500, 1000]


# ── Env / reward factories ───────────────────────────────────────
def make_pendulum_env(theta_deg):
    def _fn(seed):
        from envs.pendulum import PendulumTargetEnv
        env = PendulumTargetEnv(theta_target_deg=theta_deg)
        env.reset(seed=seed)
        return env
    return _fn

def make_pendulum_eval_env(theta_deg):
    def _fn():
        from envs.pendulum import PendulumTargetEnv
        return PendulumTargetEnv(theta_target_deg=theta_deg)
    return _fn

def make_gt_reward_fn(theta_deg):
    """Returns a function that computes GT return for a segment."""
    from envs.pendulum import PendulumTargetEnv
    env = PendulumTargetEnv(theta_target_deg=theta_deg)
    def gt_fn(obs_seq, act_seq):
        return env.gt_segment_return(obs_seq, act_seq)
    return gt_fn


# ── SAC with GT reward (baseline) ───────────────────────────────
def run_sac_gt(theta_deg, seeds, log_dir):
    """Train SAC with ground-truth reward — the upper-bound baseline."""
    from agents.sac import SAC
    from utils.trainer import run_seeds

    def make_agent(seed):
        return SAC(obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                   lr=3e-4, gamma=0.99, tau=0.005,
                   batch_size=256, buffer_size=200_000,
                   hidden=(256, 256), auto_alpha=True, device=DEVICE)

    ts, mean, std = run_seeds(
        agent_fn     = make_agent,
        train_env_fn = make_pendulum_env(theta_deg),
        eval_env_fn  = make_pendulum_eval_env(theta_deg),
        seeds        = seeds,
        total_steps  = TOTAL_STEPS,
        eval_every   = EVAL_EVERY,
        eval_episodes= EVAL_EPS,
        random_steps = RANDOM_STEPS,
        log_dir      = log_dir,
        run_prefix   = f"sac_gt_theta{theta_deg}",
        n_workers    = None,
    )
    return ts, mean, std


# ── PEBBLE ───────────────────────────────────────────────────────
def run_pebble(theta_deg, seeds, log_dir, budget=500, run_prefix=None):
    from agents.pebble_trainer import run_pebble_seeds

    prefix = run_prefix or f"pebble_theta{theta_deg}_budget{budget}"

    ts, mean, std = run_pebble_seeds(
        env_fn        = make_pendulum_env(theta_deg),
        eval_env_fn   = make_pendulum_eval_env(theta_deg),
        gt_reward_fn  = make_gt_reward_fn(theta_deg),
        obs_dim       = OBS_DIM,
        action_dim    = ACTION_DIM,
        seeds         = seeds,
        total_steps   = TOTAL_STEPS,
        query_budget  = budget,
        query_every   = QUERY_EVERY,
        n_queries     = N_QUERIES,
        segment_len   = SEGMENT_LEN,
        reward_train_steps = REWARD_STEPS,
        eval_every    = EVAL_EVERY,
        eval_episodes = EVAL_EPS,
        random_steps  = RANDOM_STEPS,
        log_dir       = log_dir,
        run_prefix    = prefix,
        device        = DEVICE,
        n_workers     = 8,
    )
    return ts, mean, std


if __name__ == "__main__":
    from utils.plotting import plot_curves
    print(f"Using device: {DEVICE}")

    # ── Part 1: SAC-GT vs PEBBLE for each θ_target ──────────────
    print("\n=== Part 1: SAC-GT vs PEBBLE ===")
    for theta in TARGET_ANGLES:
        print(f"\n--- θ_target = {theta}° ---")
        log_sub = f"{LOG_DIR}/theta_{theta}"

        ts_gt, mean_gt, std_gt = run_sac_gt(theta, SEEDS, log_sub)
        ts_pb, mean_pb, std_pb = run_pebble(theta, SEEDS, log_sub, budget=500)

        plot_curves(
            [
                {"label": "SAC (GT reward)", "timesteps": ts_gt,
                 "mean": mean_gt, "std": std_gt, "color": "#2196F3"},
                {"label": "PEBBLE (budget=500)", "timesteps": ts_pb,
                 "mean": mean_pb, "std": std_pb, "color": "#F44336"},
            ],
            title     = f"SAC-GT vs PEBBLE  |  θ_target = {theta}°",
            ylabel    = "Average Undiscounted Return (GT)",
            save_path = f"{log_sub}/plots/sac_vs_pebble.png",
        )

    # ── Part 2: Different budgets at θ_target = 90 ──────────────
    print("\n=== Part 2: Budget ablation at θ_target = 90° ===")
    theta = 90
    log_sub = f"{LOG_DIR}/budget_ablation"
    curves = []

    COLORS = ["#9C27B0", "#F44336", "#FF9800", "#4CAF50", "#2196F3"]
    for budget, color in zip(BUDGETS, COLORS):
        print(f"\n  Budget = {budget}")
        ts_pb, mean_pb, std_pb = run_pebble(
            theta, SEEDS, log_sub, budget=budget,
            run_prefix=f"pebble_budget{budget}"
        )
        curves.append({
            "label":     f"PEBBLE (budget={budget})",
            "timesteps": ts_pb,
            "mean":      mean_pb,
            "std":       std_pb,
            "color":     color,
        })

    # Add SAC-GT reference
    ts_gt, mean_gt, std_gt = run_sac_gt(theta, SEEDS, log_sub)
    curves.insert(0, {
        "label": "SAC (GT reward)", "timesteps": ts_gt,
        "mean": mean_gt, "std": std_gt, "color": "black",
    })

    plot_curves(
        curves,
        title     = "PEBBLE Budget Ablation  |  θ_target = 90°",
        ylabel    = "Average Undiscounted Return (GT)",
        save_path = f"{log_sub}/plots/budget_ablation.png",
    )

    print("\nDone.")
