"""
evaluate.py
-----------
Load saved DQN weights and render MountainCar-v0 visually.

Usage
-----
# Run best weights for seed 0
python evaluate.py --ckpt logs/weights/trunc2000_rho1_seed0_best.pt

# Run 5 episodes, show stats
python evaluate.py --ckpt logs/weights/trunc2000_rho1_seed0_best.pt --episodes 5

# Run without rendering (just print stats)
python evaluate.py --ckpt logs/weights/trunc2000_rho1_seed0_best.pt --no_render
"""

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import argparse
import time


# ─────────────────────────────────────────────
# Q-Network  (must match training architecture)
# ─────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────
def evaluate(
    ckpt_path: str,
    n_episodes: int = 3,
    render: bool = True,
    truncation: int = 2000,
    hidden: int = 128,
    seed: int = 42,
    slow: bool = False,       # add delay between frames for easier viewing
):
    # ── Load checkpoint ───────────────────────
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"\nLoaded checkpoint: {ckpt_path}")
    print(f"  Trained seed   : {ckpt.get('seed', 'unknown')}")
    print(f"  Total steps    : {ckpt.get('total_steps', 'unknown')}")
    print(f"  Episodes       : {ckpt.get('episode', 'unknown')}")
    if "avg20" in ckpt:
        print(f"  Best avg20     : {ckpt['avg20']:.1f}")

    # ── Build network and load weights ────────
    obs_dim   = 2   # MountainCar: [position, velocity]
    n_actions = 3   # push-left, no-push, push-right

    q_net = QNetwork(obs_dim, n_actions, hidden)
    q_net.load_state_dict(ckpt["q_net"])
    q_net.eval()

    # ── Create environment with rendering ─────
    render_mode = "human" if render else None
    env = gym.make(
        "MountainCar-v0",
        max_episode_steps=truncation,
        render_mode=render_mode,
    )

    print(f"\nRunning {n_episodes} episode(s) | render={render}\n")
    print(f"{'Episode':>8}  {'Return':>8}  {'Steps':>6}  {'Solved':>6}")
    print("─" * 36)

    returns = []

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        ep_steps  = 0
        solved    = False

        while True:
            # Greedy action — no exploration
            with torch.no_grad():
                obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = int(q_net(obs_t).argmax(dim=1).item())

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            ep_steps  += 1

            if slow:
                time.sleep(0.02)   # ~50 fps, easier to watch

            if terminated:
                solved = True
                break
            if truncated:
                break

        returns.append(ep_return)
        print(f"{ep:>8}  {ep_return:>8.1f}  {ep_steps:>6}  {'✔' if solved else '✘':>6}")

    env.close()

    print("─" * 36)
    print(f"  Mean return : {np.mean(returns):.1f}")
    print(f"  Solved      : {sum(r > -2000 for r in returns)}/{n_episodes} episodes\n")

    return returns


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate saved DQN on MountainCar-v0")
    p.add_argument("--ckpt",       type=str, required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--episodes",   type=int, default=3,
                   help="Number of episodes to run")
    p.add_argument("--truncation", type=int, default=2000,
                   help="Episode truncation length (should match training)")
    p.add_argument("--hidden",     type=int, default=128,
                   help="Hidden layer size (must match training)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--no_render",  action="store_true",
                   help="Disable graphical rendering")
    p.add_argument("--slow",       action="store_true",
                   help="Add frame delay for easier viewing")
    args = p.parse_args()

    evaluate(
        ckpt_path  = args.ckpt,
        n_episodes = args.episodes,
        render     = not args.no_render,
        truncation = args.truncation,
        hidden     = args.hidden,
        seed       = args.seed,
        slow       = args.slow,
    )