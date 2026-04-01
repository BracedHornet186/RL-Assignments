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
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# =====================
# Hyperparameters
# =====================
start = time.time()

GAMMA = 0.99
LR = 1e-3
BUFFER_SIZE = 50000

BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 500
TRAIN_FREQ = 1

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 100000

HIDDEN_SIZES = [64, 64]

NUM_EPISODES = 600
MAX_STEPS = 2000
NUM_SEEDS = 15

# Sensitivity values
BATCH_SIZE_LIST = [16,32,64,128,256]
TARGET_UPDATE_LIST = [125,250,500,1000,2000]

RHO_VALUES = [1,4]

DEVICE = torch.device("cpu")

os.makedirs("results", exist_ok=True)
torch.set_num_threads(1)
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
# Train Function
# =====================
def train_dqn(args):
    seed, rho, batch_size, target_update_freq = args

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("MountainCar-v0", max_episode_steps=MAX_STEPS)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim).to(DEVICE)
    target_net = QNetwork(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    total_steps = 0
    episode_returns = []

    for ep in range(NUM_EPISODES):
        state, _ = env.reset(seed=seed + ep)
        ep_reward = 0

        for t in range(MAX_STEPS):
            epsilon = max(EPS_END, EPS_START - total_steps / EPS_DECAY_STEPS)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
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

            # Training
            if len(buffer) > batch_size and total_steps % TRAIN_FREQ == 0:
                for _ in range(rho):
                    s, a, r, s_next, d = buffer.sample(batch_size)

                    q_vals = q_net(s).gather(1, a.unsqueeze(1)).squeeze()

                    with torch.no_grad():
                        max_next_q = target_net(s_next).max(1)[0]
                        target = r + GAMMA * max_next_q * (1 - d)

                    loss = nn.SmoothL1Loss()(q_vals, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Target update
            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(q_net.state_dict())

            if terminated or truncated:
                break

        episode_returns.append(ep_reward)

    env.close()
    return episode_returns

# =====================
# Utility
# =====================
def run_config(config_name, jobs):
    print(f"\n===== RUNNING: {config_name} =====")

    num_workers = min(6, NUM_SEEDS)

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(train_dqn, jobs), total=len(jobs)))

    returns = np.array(results)

    final_perf = returns[:, -100:].mean(axis=1)

    mean = final_perf.mean()
    std = final_perf.std()
    ci = 1.96 * std / np.sqrt(NUM_SEEDS)

    print(f"Finished {config_name} → Mean={mean:.2f}, CI={ci:.2f}")

    return returns, mean, ci

# =====================
# MAIN
# =====================
if __name__ == "__main__":

    ##############################
    # BATCH SIZE SENSITIVITY
    ##############################
    batch_results = {rho: [] for rho in RHO_VALUES}

    for rho in RHO_VALUES:
        for bs in BATCH_SIZE_LIST:
            config_name = f"BATCH | rho={rho} | bs={bs}"

            jobs = [(seed, rho, bs, TARGET_UPDATE_FREQ) for seed in range(NUM_SEEDS)]

            returns, mean, ci = run_config(config_name, jobs)

            batch_results[rho].append((bs, mean, ci))

            pd.DataFrame(returns).to_csv(
                f"results/batch_rho{rho}_bs{bs}.csv", index=False
            )

    ##############################
    # TARGET UPDATE SENSITIVITY
    ##############################
    target_results = {rho: [] for rho in RHO_VALUES}

    for rho in RHO_VALUES:
        for tu in TARGET_UPDATE_LIST:
            config_name = f"TARGET | rho={rho} | tu={tu}"

            jobs = [(seed, rho, BATCH_SIZE, tu) for seed in range(NUM_SEEDS)]

            returns, mean, ci = run_config(config_name, jobs)

            target_results[rho].append((tu, mean, ci))

            pd.DataFrame(returns).to_csv(
                f"results/target_rho{rho}_tu{tu}.csv", index=False
            )

    ##############################
    # PLOT: BATCH SIZE
    ##############################
    plt.figure(figsize=(8,6))

    for rho in RHO_VALUES:
        x = [v[0] for v in batch_results[rho]]
        y = [v[1] for v in batch_results[rho]]
        ci = [v[2] for v in batch_results[rho]]

        plt.errorbar(x, y, yerr=ci, marker='o', capsize=4, label=f"ρ = {rho}")

    plt.xscale("log")
    plt.xticks(x, x)
    plt.xlabel("Batch Size")
    plt.ylabel("Performance (last 100 eps)")
    plt.title("Sensitivity: Batch Size")
    plt.legend()
    plt.grid()

    plt.savefig("results/sensitivity_batch.png", dpi=300)
    plt.show()

    ##############################
    # PLOT: TARGET UPDATE
    ##############################
    plt.figure(figsize=(8,6))

    for rho in RHO_VALUES:
        x = [v[0] for v in target_results[rho]]
        y = [v[1] for v in target_results[rho]]
        ci = [v[2] for v in target_results[rho]]

        plt.errorbar(x, y, yerr=ci, marker='o', capsize=4, label=f"ρ = {rho}")

    plt.xscale("log")
    plt.xticks(x, x)
    plt.xlabel("Target Update Frequency")
    plt.ylabel("Performance (last 100 eps)")
    plt.title("Sensitivity: Target Network")
    plt.legend()
    plt.grid()

    plt.savefig("results/sensitivity_target.png", dpi=300)
    plt.show()

    print(f"\nTotal Time: {(time.time() - start)/60:.2f} minutes")