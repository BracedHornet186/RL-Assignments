"""
Vanilla DQN for MountainCar-v0
DA6400 Programming Assignment 2

Features:
- Epsilon-greedy exploration with epsilon decay
- Fixed-size replay buffer with uniform random sampling
- Hard target network updates
- Adam optimizer
- Kaiming (He) initialization for ReLU-based Q-network
- Truncation length configurable (default: 2000)
- Logs return per episode and per timestep for plotting
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import os
import csv
from collections import deque


# ──────────────────────────────────────────────
# Q-Network
# ──────────────────────────────────────────────
class QNetwork(nn.Module):
    """
    Two-hidden-layer MLP for Q-value estimation.
    Architecture: 2 → 128 → 128 → 3
    Kaiming (He) uniform initialization for all linear layers.
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────
class ReplayBuffer:
    """Fixed-size circular replay buffer with uniform random sampling."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ──────────────────────────────────────────────
# DQN Agent
# ──────────────────────────────────────────────
class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        # Hyperparameters
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 500,   # hard update every N timesteps
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 50_000,
        replay_factor: int = 1,          # ρ — updates per timestep
        hidden: int = 128,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.replay_factor = replay_factor
        self.device = torch.device(device)

        # Networks
        self.q_net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.total_steps = 0
        self.loss_fn = nn.MSELoss()

    # ── Epsilon schedule ──────────────────────
    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.total_steps / self.eps_decay_steps)
        return self.eps_start + fraction * (self.eps_end - self.eps_start)

    # ── Action selection ──────────────────────
    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.q_net(state_t).argmax(dim=1).item())

    # ── Single gradient update ─────────────────
    def _update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t      = torch.tensor(states,      device=self.device)
        actions_t     = torch.tensor(actions,     device=self.device).unsqueeze(1)
        rewards_t     = torch.tensor(rewards,     device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t       = torch.tensor(dones,       device=self.device)

        # Current Q-values
        q_values = self.q_net(states_t).gather(1, actions_t).squeeze(1)

        # Target Q-values (hard target network, no gradient)
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1).values
            targets = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ── Step: store transition + ρ updates ────
    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

        # ρ gradient updates per timestep
        for _ in range(self.replay_factor):
            self._update()

        # Hard target network update
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def make_env(truncation_length: int = 2000, seed: int = 0):
    """Create MountainCar-v0 with custom truncation length."""
    env = gym.make("MountainCar-v0", max_episode_steps=truncation_length)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def train(
    seed: int = 0,
    total_timesteps: int = 200_000,
    truncation_length: int = 2000,
    replay_factor: int = 1,
    # Hyperparameters
    lr: float = 1e-3,
    gamma: float = 0.99,
    buffer_size: int = 50_000,
    batch_size: int = 64,
    target_update_freq: int = 500,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 50_000,
    hidden: int = 128,
    device: str = "cpu",
    log_dir: str = "logs",
    run_tag: str = "",
):
    """
    Train DQN on MountainCar-v0 and save episode logs.

    Returns
    -------
    episode_returns : list of float
    episode_timesteps : list of int  (global timestep at end of each episode)
    """
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(truncation_length=truncation_length, seed=seed)
    env.reset(seed=seed)

    obs_dim  = env.observation_space.shape[0]   # 2: [position, velocity]
    n_actions = env.action_space.n              # 3: push-left, no-push, push-right

    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        lr=lr,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay_steps=eps_decay_steps,
        replay_factor=replay_factor,
        hidden=hidden,
        device=device,
    )

    episode_returns   = []
    episode_timesteps = []

    state, _ = env.reset()
    ep_return = 0.0
    ep_steps  = 0

    for t in range(1, total_timesteps + 1):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store terminal flag only for true termination (reaching goal),
        # NOT for time-based truncation (bootstrapping continues from truncated states)
        agent.step(state, action, reward, next_state, float(terminated))

        ep_return += reward
        ep_steps  += 1
        state = next_state

        if done:
            episode_returns.append(ep_return)
            episode_timesteps.append(t)
            state, _ = env.reset()
            ep_return = 0.0
            ep_steps  = 0

    env.close()

    # ── Save logs ─────────────────────────────
    os.makedirs(log_dir, exist_ok=True)
    tag = run_tag or f"trunc{truncation_length}_rho{replay_factor}_seed{seed}"
    csv_path = os.path.join(log_dir, f"{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "timestep", "return"])
        for ep_idx, (ret, ts) in enumerate(zip(episode_returns, episode_timesteps)):
            writer.writerow([ep_idx + 1, ts, ret])

    print(f"[seed={seed}] Episodes={len(episode_returns)}  "
          f"Best return={max(episode_returns):.1f}  "
          f"Final 20-ep avg={np.mean(episode_returns[-20:]):.1f}")

    return episode_returns, episode_timesteps


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Vanilla DQN — MountainCar-v0")
    p.add_argument("--seeds",          type=int,   nargs="+", default=list(range(15)),
                   help="Random seeds (default: 0-14)")
    p.add_argument("--total_timesteps",type=int,   default=200_000)
    p.add_argument("--truncation",     type=int,   default=2000,
                   help="Episode truncation length")
    p.add_argument("--replay_factor",  type=int,   default=1,
                   help="ρ: gradient updates per timestep")
    # Hyperparameters
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--gamma",          type=float, default=0.99)
    p.add_argument("--buffer_size",    type=int,   default=50_000)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--target_update",  type=int,   default=500,
                   help="Hard target network update frequency (timesteps)")
    p.add_argument("--eps_start",      type=float, default=1.0)
    p.add_argument("--eps_end",        type=float, default=0.05)
    p.add_argument("--eps_decay",      type=int,   default=50_000,
                   help="Timesteps over which epsilon decays")
    p.add_argument("--hidden",         type=int,   default=128)
    p.add_argument("--device",         type=str,   default="cuda")
    p.add_argument("--log_dir",        type=str,   default="logs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("Vanilla DQN — MountainCar-v0")
    print(f"  truncation  : {args.truncation} steps")
    print(f"  replay_factor ρ : {args.replay_factor}")
    print(f"  seeds       : {args.seeds}")
    print(f"  total_timesteps : {args.total_timesteps}")
    print("=" * 60)

    for seed in args.seeds:
        train(
            seed=seed,
            total_timesteps=args.total_timesteps,
            truncation_length=args.truncation,
            replay_factor=args.replay_factor,
            lr=args.lr,
            gamma=args.gamma,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            target_update_freq=args.target_update,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay_steps=args.eps_decay,
            hidden=args.hidden,
            device=args.device,
            log_dir=args.log_dir,
        )