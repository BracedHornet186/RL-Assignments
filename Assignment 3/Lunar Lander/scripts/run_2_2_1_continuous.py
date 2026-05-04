"""
Q2.2 Parts 1 & 2 — Continuous SAC on LunarLander-v3.
Saves checkpoints every 50K steps for later replay/visualisation.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

SEEDS        = list(range(2))
TOTAL_STEPS  = 300_000
EVAL_EVERY   = 10_000
EVAL_EPS     = 20
RANDOM_STEPS = 10_000
SAVE_EVERY   = 50_000   # saves seed0_step50000.pt, seed0_step100000.pt, ...
LOG_DIR      = "logs/q2_2_1_continuous"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

OBS_DIM    = 8
ACTION_DIM = 2


def make_train_env(seed):
    import gymnasium as gym
    env = gym.make("LunarLander-v3", continuous=True)
    env.reset(seed=seed)
    return env

def make_eval_env():
    import gymnasium as gym
    return gym.make("LunarLander-v3", continuous=True)

def make_agent(seed):
    from agents.sac import SAC
    return SAC(
        obs_dim    = OBS_DIM,
        action_dim = ACTION_DIM,
        lr         = 3e-4,
        gamma      = 0.99,
        tau        = 0.005,
        batch_size = 256,
        buffer_size= 300_000,
        hidden     = (256, 256),
        auto_alpha = True,
        device     = DEVICE,
    )


if __name__ == "__main__":
    from utils.trainer import run_seeds
    from utils.plotting import plot_curves
    print(f"Using device: {DEVICE}")

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
        save_every    = SAVE_EVERY,
        n_workers     = None,
    )

    plot_curves(
        [{"label": "SAC (continuous, auto-α)", "timesteps": ts, "mean": mean, "std": std}],
        title     = "SAC on LunarLander-v3 (Continuous)",
        save_path = f"{LOG_DIR}/plots/sac_continuous.png",
    )
    print("Done.")