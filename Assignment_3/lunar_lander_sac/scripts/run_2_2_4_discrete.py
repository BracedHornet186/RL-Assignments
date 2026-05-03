"""
Q2.2 Part 4 — Discrete SAC vs DQN on LunarLander-v3 (discrete).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

SEEDS        = list(range(15))
TOTAL_STEPS  = 500_000
EVAL_EVERY   = 10_000
EVAL_EPS     = 20
RANDOM_STEPS = 10_000
LOG_DIR      = "logs/q2_2_4_discrete_test"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

OBS_DIM   = 8
N_ACTIONS = 4


def make_train_env(seed):
    import gymnasium as gym
    env = gym.make("LunarLander-v3", continuous=False)
    env.reset(seed=seed)
    return env

def make_eval_env():
    import gymnasium as gym
    return gym.make("LunarLander-v3", continuous=False)

def make_discrete_sac(seed):
    from agents.sac import DiscreteSAC
    return DiscreteSAC(
        obs_dim=OBS_DIM, n_actions=N_ACTIONS,
        lr=1e-4, gamma=0.99, tau=0.005,
        batch_size=512, buffer_size=300_000,
        hidden=(256, 256), auto_alpha=True,
        device=DEVICE,
    )

def make_dqn(seed):
    from agents.dqn import DQN
    return DQN(
        obs_dim=OBS_DIM, n_actions=N_ACTIONS,
        lr=3e-4, gamma=0.99, tau=0.005,
        batch_size=256, buffer_size=300_000,
        hidden=(256, 256),
        eps_start=1.0, eps_end=0.05, eps_decay=50_000,
        device=DEVICE,
    )


if __name__ == "__main__":
    from utils.trainer import run_seeds
    from utils.plotting import plot_curves
    print(f"Using device: {DEVICE}")

    print("=== Discrete SAC ===")
    ts_sac, mean_sac, std_sac = run_seeds(
        agent_fn=make_discrete_sac, train_env_fn=make_train_env,
        eval_env_fn=make_eval_env, seeds=SEEDS,
        total_steps=TOTAL_STEPS, eval_every=EVAL_EVERY,
        eval_episodes=EVAL_EPS, random_steps=RANDOM_STEPS,
        log_dir=LOG_DIR, run_prefix="discrete_sac",
        discrete=True, n_workers=None,
    )

    plot_curves(
        [{"label": "Discrete SAC (auto-α)", "timesteps": ts_sac, "mean": mean_sac, "std": std_sac}],
        title     = "Discrete-SAC on LunarLander-v3 (Discrete)",
        save_path = f"{LOG_DIR}/plots/discrete_sac.png",
    )
    print("Done.")