"""
Vanilla DQN for MountainCar-v0
PA2 - Problem 1: Deep Q-Networks on Mountain Car

Key design choices:
- 2-hidden-layer MLP with ReLU activations (64x64) — not too big, not too small
- Kaiming (He) initialization for all linear layers
- ε-greedy with linear decay from 1.0 → 0.05
- Replay buffer with uniform random sampling
- Hard target network updates (copy every C steps)
- Adam optimizer
- γ = 0.99
- Episode truncation configurable (default 2000)
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import argparse
import os
import json
import concurrent.futures
import multiprocessing as mp

# ─────────────────────────────────────────────
# Q-Network
# ─────────────────────────────────────────────
class QNetwork(nn.Module):
    """
    2-hidden-layer MLP. Hidden size 64 is appropriate for a 2D state space
    with 3 discrete actions — avoids both over- and under-parameterization.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        self._init_weights()

    def _init_weights(self):
        """Kaiming (He) initialization for ReLU networks."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────
class ReplayBuffer:
    """Fixed-size circular buffer with uniform random sampling."""
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


# ─────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────
class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # Hyperparameters — tuned for MountainCar
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 500,   # hard update every C steps
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 50_000,   # linear decay over 50k steps
        replay_factor: int = 1,          # ρ — how many gradient steps per timestep
        device: str = "cuda",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.replay_factor = replay_factor
        self.device = torch.device("cuda")

        # Online and target networks
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        self.total_steps = 0

    @property
    def epsilon(self) -> float:
        """Linear ε decay."""
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax(dim=1).item()

    def update(self):
        """Perform ρ gradient steps per call."""
        if len(self.buffer) < self.batch_size:
            return

        for _ in range(self.replay_factor):
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

            states_t     = torch.FloatTensor(states).to(self.device)
            actions_t    = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards_t    = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states_t= torch.FloatTensor(next_states).to(self.device)
            dones_t      = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

            # Current Q(s,a)
            current_q = self.q_net(states_t).gather(1, actions_t)

            # Target: r + γ * max_a' Q_target(s', a')  (0 if terminal)
            with torch.no_grad():
                max_next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
                target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)

            loss = nn.MSELoss()(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
            self.optimizer.step()

    def step(self, state, action, reward, next_state, done):
        """Store transition and update."""
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
        self.update()

        # Hard target network update
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
def make_env(truncation_length: int = 2000):
    """Create MountainCar-v0 with custom truncation length."""
    env = gym.make("MountainCar-v0", max_episode_steps=truncation_length)
    return env


def train(
    seed: int,
    truncation_length: int = 2000,
    total_timesteps: int = 200_000,
    replay_factor: int = 1,
    # Hyperparameters
    lr: float = 1e-3,
    batch_size: int = 64,
    buffer_capacity: int = 50_000,
    target_update_freq: int = 500,
    eps_decay_steps: int = 50_000,
    device: str = "cuda",
    log_interval: int = 1,       # log every N episodes
):
    # os.environ["OMP_NUM_THREADS"] = "1"
    # torch.set_num_threads(1)

    """
    Train a DQN agent and return per-episode logs.

    Returns:
        episode_returns  : list of undiscounted returns per episode
        episode_steps    : list of cumulative timesteps at end of each episode
    """
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(truncation_length)
    env.reset(seed=seed)

    state_dim  = env.observation_space.shape[0]   # 2 (position, velocity)
    action_dim = env.action_space.n               # 3

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=0.99,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        eps_decay_steps=eps_decay_steps,
        replay_factor=replay_factor,
        device=device,
    )

    episode_returns = []
    episode_steps   = []
    timestep        = 0
    episode         = 0

    state, _ = env.reset()
    ep_return = 0.0
    ep_len    = 0

    while timestep < total_timesteps:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated  # only TRUE termination counts as done for bootstrap
        agent.step(state, action, reward, next_state, done)

        ep_return += reward
        ep_len    += 1
        timestep  += 1
        state      = next_state

        if terminated or truncated:
            episode_returns.append(ep_return)
            episode_steps.append(timestep)
            episode += 1

            # Reset
            state, _ = env.reset()
            ep_return = 0.0
            ep_len    = 0

    env.close()
    return episode_returns, episode_steps


# ─────────────────────────────────────────────
# Multi-seed runner
# ─────────────────────────────────────────────
def run_experiment(
    n_seeds: int = 15,
    truncation_length: int = 2000,
    total_timesteps: int = 200_000,
    replay_factor: int = 1,
    batch_size: int = 64,
    target_update_freq: int = 500,
    save_dir: str = "results",
    tag: str = "dqn",
    device: str = "cuda",
):
    os.makedirs(save_dir, exist_ok=True)
    all_returns = []
    all_steps   = []

    for seed in range(n_seeds):
        print(f"[{tag}] seed {seed+1}/{n_seeds} ...")
        returns, steps = train(
            seed=seed,
            truncation_length=truncation_length,
            total_timesteps=total_timesteps,
            replay_factor=replay_factor,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            device=device,
        )
        all_returns.append(returns)
        all_steps.append(steps)

    # Save raw logs
    results = {"returns": all_returns, "steps": all_steps,
               "tag": tag, "truncation": truncation_length,
               "replay_factor": replay_factor}
    path = os.path.join(save_dir, f"{tag}.json")
    with open(path, "w") as f:
        json.dump(results, f)
    print(f"Saved → {path}")
    return all_returns, all_steps

# ─────────────────────────────────────────────
# Multi-seed runner (Parallel)
# ─────────────────────────────────────────────
def run_experiment_parallel(
    n_seeds: int = 15,
    truncation_length: int = 2000,
    total_timesteps: int = 200_000,
    replay_factor: int = 1,
    batch_size: int = 64,
    target_update_freq: int = 500,
    save_dir: str = "results",
    tag: str = "dqn",
    device: str = "cuda",
    max_workers: int = None,
):
    os.makedirs(save_dir, exist_ok=True)
    
    # Pre-allocate lists to maintain seed order
    all_returns = [None] * n_seeds
    all_steps   = [None] * n_seeds

    # Prepare arguments for each worker
    tasks = []
    for seed in range(n_seeds):
        tasks.append({
            "seed": seed,
            "truncation_length": truncation_length,
            "total_timesteps": total_timesteps,
            "replay_factor": replay_factor,
            "batch_size": batch_size,
            "target_update_freq": target_update_freq,
            "device": device
        })

    print(f"[{tag}] Starting {n_seeds} seeds in parallel using {device}...")

    # Default to using as many workers as seeds, capped by CPU availability if not specified
    if max_workers is None:
        max_workers = min(n_seeds, os.cpu_count() or 1)

    ctx = mp.get_context('spawn')

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        # Submit all tasks and map future to its seed index
        future_to_seed = {executor.submit(train, **task): task["seed"] for task in tasks}
        
        for future in concurrent.futures.as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                returns, steps = future.result()
                all_returns[seed] = returns
                all_steps[seed] = steps
                print(f"[{tag}] ✔ Seed {seed+1}/{n_seeds} completed.")
            except Exception as exc:
                print(f"[{tag}] ❌ Seed {seed+1} generated an exception: {exc}")

    # Save raw logs
    results = {
        "returns": all_returns, 
        "steps": all_steps,
        "tag": tag, 
        "truncation": truncation_length,
        "replay_factor": replay_factor
    }
    path = os.path.join(save_dir, f"{tag}.json")
    with open(path, "w") as f:
        json.dump(results, f)
    print(f"Saved → {path}")
    return all_returns, all_steps


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # CRITICAL: Must use 'spawn' for CUDA multiprocessing
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass # Start method has already been set

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",       type=int, default=15)
    parser.add_argument("--timesteps",   type=int, default=200_000)
    parser.add_argument("--truncation",  type=int, default=2000,
                        help="Episode truncation length (200 | 1000 | 2000)")
    parser.add_argument("--rho",         type=int, default=1,
                        help="Replay factor ρ")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--target_freq", type=int, default=500)
    parser.add_argument("--save_dir",    type=str, default="results")
    parser.add_argument("--tag",         type=str, default="dqn_trunc2000_rho1")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--workers",     type=int, default=15, 
                        help="Number of parallel workers (defaults to min(seeds, cpu_count))")
    args = parser.parse_args()

    run_experiment_parallel(
        n_seeds=args.seeds,
        truncation_length=args.truncation,
        total_timesteps=args.timesteps,
        replay_factor=args.rho,
        batch_size=args.batch_size,
        target_update_freq=args.target_freq,
        save_dir=args.save_dir,
        tag=args.tag,
        device=args.device,
        max_workers=args.workers,
    )