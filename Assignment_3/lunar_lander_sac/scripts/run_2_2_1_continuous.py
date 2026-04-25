"""
Q2.2 Part 1 & 2
Train continuous SAC on LunarLander-v3 (continuous actions).
Automated temperature tuning with target entropy.
15 seeds, evaluate every 10K steps using 20 offline episodes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import gymnasium as gym

from agents.sac import SAC
from utils.trainer import run_seeds
from utils.plotting import plot_curves

# ── Config ──────────────────────────────────────────────────────
SEEDS        = list(range(2))
TOTAL_STEPS  = 300_000     # sufficient for LunarLander convergence
EVAL_EVERY   = 10_000
EVAL_EPS     = 20
RANDOM_STEPS = 10_000
LOG_DIR      = "logs/q2_2_1_continuous"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ── Environment factory ──────────────────────────────────────────
def make_train_env(seed):
    env = gym.make("LunarLander-v3", continuous=True)
    env.reset(seed=seed)
    return env

def make_eval_env():
    return gym.make("LunarLander-v3", continuous=True)


# ── Agent factory ────────────────────────────────────────────────
def make_agent(seed):
    env = gym.make("LunarLander-v3", continuous=True)
    obs_dim    = env.observation_space.shape[0]   # 8
    action_dim = env.action_space.shape[0]         # 2
    env.close()

    return SAC(
        obs_dim    = obs_dim,
        action_dim = action_dim,
        lr         = 3e-4,
        gamma      = 0.99,
        tau        = 0.005,
        batch_size = 256,
        buffer_size= 300_000,
        hidden     = (256, 256),
        auto_alpha = True,
        device     = DEVICE,
    )


# ── Run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    ts, mean, std = run_seeds(
        agent_fn      = make_agent,
        train_env_fn  = make_train_env,
        eval_env_fn   = make_eval_env,
        seeds         = SEEDS,
        total_steps   = TOTAL_STEPS,
        eval_every    = EVAL_EVERY,
        eval_episodes = EVAL_EPS,
        random_steps  = RANDOM_STEPS,
        log_dir       = LOG_DIR,
        run_prefix    = "sac_continuous",
        verbose       = True,
    )

    plot_curves(
        [{"label": "SAC (continuous, auto-α)", "timesteps": ts, "mean": mean, "std": std}],
        title     = "SAC on LunarLander-v3 (Continuous)",
        save_path = f"{LOG_DIR}/plots/sac_continuous.png",
    )
    print("Done.")
