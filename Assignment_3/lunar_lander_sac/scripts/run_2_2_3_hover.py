"""
Q2.2 Part 3
Hover-reward LunarLander experiment.
  - fixed alpha = 0.01
  - auto alpha
Both trained on:
  Phase 1: hover bonus = +200
  Phase 2 (after switch_step): hover bonus = -100
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import gymnasium as gym
from pathlib import Path

from agents.sac import SAC
from envs.lunar_lander import HoverLunarLander, ChangingRewardLunarLander
from utils.trainer import run_seeds
from utils.plotting import plot_curves

# ── Config ──────────────────────────────────────────────────────
SEEDS        = list(range(15))
TOTAL_STEPS  = 500_000
SWITCH_STEP  = 250_000    # reward switches here
EVAL_EVERY   = 10_000
EVAL_EPS     = 20
RANDOM_STEPS = 10_000
LOG_DIR      = "logs/q2_2_3_hover"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def get_obs_action_dims():
    env = gym.make("LunarLander-v3", continuous=True)
    od = env.observation_space.shape[0]
    ad = env.action_space.shape[0]
    env.close()
    return od, ad

OBS_DIM, ACTION_DIM = get_obs_action_dims()


# ── Env factories ────────────────────────────────────────────────
def make_changing_train_env(seed):
    env = ChangingRewardLunarLander(switch_step=SWITCH_STEP, continuous=True)
    env.reset(seed=seed)
    return env

def make_hover_eval_env_phase1():
    """Evaluate in the +200 hover env."""
    return HoverLunarLander(hover_bonus=200.0, continuous=True)

def make_hover_eval_env_phase2():
    """Evaluate in the -100 hover env (after switch)."""
    return HoverLunarLander(hover_bonus=-100.0, continuous=True)


# ── Agent factories ──────────────────────────────────────────────
def make_fixed_alpha_agent(seed):
    return SAC(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        lr=3e-4, gamma=0.99, tau=0.005,
        batch_size=256, buffer_size=600_000,
        hidden=(256, 256),
        auto_alpha=False, alpha=0.01,
        device=DEVICE,
    )

def make_auto_alpha_agent(seed):
    return SAC(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        lr=3e-4, gamma=0.99, tau=0.005,
        batch_size=256, buffer_size=600_000,
        hidden=(256, 256),
        auto_alpha=True,
        device=DEVICE,
    )


# ── Hook: update env's global step so reward switches ───────────
def make_hook(train_env):
    def hook(step):
        train_env.set_global_step(step)
    return hook


# ── Custom training loop for this experiment (hooks needed) ─────
from utils.trainer import train as _train

def run_hover_seeds(agent_fn, label, log_prefix):
    all_seed_means = []
    ts_ref = None

    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)

        agent     = agent_fn(seed)
        train_env = ChangingRewardLunarLander(switch_step=SWITCH_STEP, continuous=True)
        eval_env  = HoverLunarLander(hover_bonus=200.0, continuous=True)   # phase-1 eval

        ts, _, rets = _train(
            agent, train_env, eval_env,
            total_steps   = TOTAL_STEPS,
            eval_every    = EVAL_EVERY,
            eval_episodes = EVAL_EPS,
            random_steps  = RANDOM_STEPS,
            seed          = seed,
            log_dir       = LOG_DIR,
            run_name      = f"{log_prefix}_seed{seed}",
            show_pbar     = True,
            env_step_hook = make_hook(train_env),
        )
        seed_means = [np.mean(r) for r in rets]
        all_seed_means.append(seed_means)
        if ts_ref is None:
            ts_ref = ts
        train_env.close()
        eval_env.close()

    arr  = np.array(all_seed_means)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)

    import json
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{LOG_DIR}/{log_prefix}_aggregated.json", "w") as f:
        json.dump({"timesteps": ts_ref, "mean": mean.tolist(), "std": std.tolist()}, f)

    return ts_ref, mean, std


if __name__ == "__main__":
    print("=== Fixed alpha = 0.01 ===")
    ts_fixed, mean_fixed, std_fixed = run_hover_seeds(
        make_fixed_alpha_agent, "Fixed α=0.01", "fixed_alpha"
    )

    print("=== Auto alpha ===")
    ts_auto, mean_auto, std_auto = run_hover_seeds(
        make_auto_alpha_agent, "Auto α", "auto_alpha"
    )

    # Annotate switch point
    switch_x = SWITCH_STEP

    fig, ax = plot_curves(
        [
            {"label": "Fixed α=0.01", "timesteps": ts_fixed, "mean": mean_fixed, "std": std_fixed},
            {"label": "Auto α",       "timesteps": ts_auto,  "mean": mean_auto,  "std": std_auto},
        ],
        title     = "LunarLander Hover Reward: +200 → -100 at 250K steps",
        save_path = f"{LOG_DIR}/plots/hover_comparison.png",
    )
    ax.axvline(x=switch_x, color="black", linestyle="--", linewidth=1.5, label="Reward switch")
    ax.legend()
    fig.savefig(f"{LOG_DIR}/plots/hover_comparison.png", dpi=150)
    print("Done.")