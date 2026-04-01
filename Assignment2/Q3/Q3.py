import numpy as np
import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import time
import os

# =====================
# Hyperparameters
# =====================
GAMMA = 0.99
LR = 5e-4 
BUFFER_SIZE = 100000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 1000
TRAIN_FREQ = 2

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 100000 

HIDDEN_SIZES = [64, 64]  

NUM_EPISODES = 600
NUM_SEEDS = 15
DEVICE = torch.device("cpu")

# =====================
# Replay Buffer
# =====================
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def __len__(self):
        return len(self.buffer)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in indices]
        s, a, r, s_next, done = zip(*batch)

        return (
            torch.stack(s).to(DEVICE),
            torch.tensor(a, dtype=torch.long).to(DEVICE),
            torch.tensor(r, dtype=torch.float32).to(DEVICE),
            torch.stack(s_next).to(DEVICE),
            torch.tensor(done, dtype=torch.float32).to(DEVICE),
        )

# =====================
# Q-Network
# =====================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        layers = []
        prev = state_dim
        for h in HIDDEN_SIZES:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

        self.apply(self.kaiming_init)

    def kaiming_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# =====================
# Train One Seed (NOW TAKES max_steps)
# =====================
def train_dqn(args):
    seed, max_steps = args

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("MountainCar-v0", max_episode_steps=max_steps)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim).to(DEVICE)
    target_net = QNetwork(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    total_steps = 0
    episode_returns = []

    pbar = tqdm(range(NUM_EPISODES), desc=f"Seed {seed} | MaxSteps {max_steps}", leave=False)

    for ep in pbar:
        state, _ = env.reset(seed=seed + ep)
        ep_reward = 0

        for t in range(max_steps):
            epsilon = max(EPS_END, EPS_START - total_steps / EPS_DECAY_STEPS)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.from_numpy(state).float().unsqueeze(0)
                    action = torch.argmax(q_net(state_t)).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated

            buffer.push((
                torch.tensor(state, dtype=torch.float32),
                action,
                reward,
                torch.tensor(next_state, dtype=torch.float32),
                done
            ))

            state = next_state
            ep_reward += reward
            total_steps += 1

            if len(buffer) > BATCH_SIZE and total_steps % TRAIN_FREQ == 0:
                s, a, r, s_next, d = buffer.sample(BATCH_SIZE)

                q_vals = q_net(s).gather(1, a.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    max_next_q = target_net(s_next).max(1)[0]
                    target = r + GAMMA * max_next_q * (1 - d)

                loss = nn.SmoothL1Loss()(q_vals, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(q_net.state_dict())

            if terminated or truncated:
                break

        if ep % 50 == 0:
            pbar.set_postfix({"Return": f"{ep_reward:.1f}", "Eps": f"{epsilon:.3f}"})

        episode_returns.append(ep_reward)

    env.close()
    return episode_returns

# =====================
# Run Experiment
# =====================
def run_experiment(max_steps):
    print(f"\nRunning for max_steps = {max_steps}")

    num_workers = min(8, NUM_SEEDS)
    with Pool(num_workers) as pool:
        all_returns = list(tqdm(
            pool.imap(train_dqn, [(seed, max_steps) for seed in range(NUM_SEEDS)]),
            total=NUM_SEEDS
        ))

    all_returns = np.array(all_returns)

    mean = all_returns.mean(axis=0)
    std = all_returns.std(axis=0)
    ci = 1.96 * std / np.sqrt(NUM_SEEDS)

    return mean, ci

# =====================
# MAIN
# =====================
if __name__ == "__main__":

    mean_200, ci_200 = run_experiment(200)
    mean_1000, ci_1000 = run_experiment(1000)
    summary_df_200 = pd.DataFrame({
        "episode": np.arange(NUM_EPISODES),
        "mean": mean_200,
        "ci_lower": mean_200 - ci_200,
        "ci_upper": mean_200 + ci_200,
    })
    summary_df_200.to_csv("summary_200.csv", index=False)
    summary_df_1000 = pd.DataFrame({
        "episode": np.arange(NUM_EPISODES),
        "mean": mean_1000,
        "ci_lower": mean_1000 - ci_1000,
        "ci_upper": mean_1000 + ci_1000,
    })
    summary_df_1000.to_csv("summary_1000.csv", index=False)
    # =====================
    # Plot Comparison
    # =====================
    plt.figure(figsize=(10, 6))

    plt.plot(mean_200, label="Max Steps = 200")
    plt.fill_between(range(NUM_EPISODES), mean_200 - ci_200, mean_200 + ci_200, alpha=0.2)

    plt.plot(mean_1000, label="Max Steps = 1000")
    plt.fill_between(range(NUM_EPISODES), mean_1000 - ci_1000, mean_1000 + ci_1000, alpha=0.2)

    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.title("DQN on MountainCar: Truncation Length Comparison")
    plt.legend()
    plt.grid(True)

    plt.savefig("comparison_200_vs_1000.png", dpi=300)
    plt.show()