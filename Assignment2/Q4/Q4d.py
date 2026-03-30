import numpy as np
import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
import os
torch.set_default_dtype(torch.float)
# =====================
# SYSTEM SETTINGS
# =====================

DEVICE = torch.device("cpu")

# =====================
# Hyperparameters
# =====================
GAMMA = 0.99
LR = 5e-4
BUFFER_SIZE = 100000
TRAIN_FREQ = 4
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 100000

HIDDEN_SIZES = [64, 64]

NUM_EPISODES = 600
MAX_STEPS = 2000

NUM_ENVS = 4    
NUM_WORKERS = 8
NUM_SEEDS = 15  

# =====================
# REPLAY BUFFER (NUMPY)
# =====================
class ReplayBuffer:
    def __init__(self, size, state_dim):
        self.size = size
        self.ptr = 0
        self.full = False

        self.s = np.zeros((size, state_dim), dtype=float)
        self.a = np.zeros(size, dtype=np.int64)
        self.r = np.zeros(size, dtype=float)
        self.s_next = np.zeros((size, state_dim), dtype=float)
        self.d = np.zeros(size, dtype=float)

    def push(self, s, a, r, s_next, d):
        n = len(s)
        idx = (np.arange(n) + self.ptr) % self.size

        self.s[idx] = s
        self.a[idx] = a
        self.r[idx] = r
        self.s_next[idx] = s_next
        self.d[idx] = d

        self.ptr = (self.ptr + n) % self.size
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size):
        max_idx = self.size if self.full else self.ptr
        idx = np.random.randint(0, max_idx, size=batch_size)

        return (
            torch.from_numpy(self.s[idx]).float(),
            torch.from_numpy(self.a[idx]).long(),
            torch.from_numpy(self.r[idx]).float(),
            torch.from_numpy(self.s_next[idx]).float(),
            torch.from_numpy(self.d[idx]).float(),
        )

# =====================
# Q NETWORK
# =====================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        layers = []
        prev = state_dim
        for h in HIDDEN_SIZES:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =====================
# ENV CREATION
# =====================
def make_env(seed):
    def _init():
        env = gym.make("MountainCar-v0", max_episode_steps=MAX_STEPS)
        env.reset(seed=seed)
        return env
    return _init

# =====================
# TRAIN FUNCTION (PER SEED)
# =====================
def train_single(config):
    seed, rho, batch_size, target_update_freq = config

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.vector.SyncVectorEnv(
        [make_env(seed * 1000 + i) for i in range(NUM_ENVS)]
    )

    state_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n

    q_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE, state_dim)

    total_steps = 0
    returns = []

    obs, _ = env.reset()
    episode_rewards = np.zeros(NUM_ENVS)

    for ep in range(NUM_EPISODES):

        for _ in range(MAX_STEPS):

            epsilon = max(EPS_END, EPS_START - total_steps / EPS_DECAY_STEPS)

            if random.random() < epsilon:
                actions = np.random.randint(0, action_dim, size=NUM_ENVS)
            else:
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs)
                    q_vals = q_net(obs_tensor)
                    actions = torch.argmax(q_vals, dim=1).numpy()

            next_obs, rewards, terms, truncs, _ = env.step(actions)
            dones = terms | truncs

            buffer.push(obs, actions, rewards, next_obs, dones)

            obs = next_obs
            episode_rewards += rewards
            total_steps += 1

            # training
            max_idx = buffer.size if buffer.full else buffer.ptr

            if max_idx > batch_size and total_steps % TRAIN_FREQ == 0:
                for _ in range(rho):
                    s, a, r, s_next, d = buffer.sample(batch_size)

                    q = q_net(s).gather(1, a.unsqueeze(1)).squeeze()

                    with torch.no_grad():
                        max_next = target_net(s_next).max(1)[0]
                        target = r + GAMMA * max_next * (1 - d)

                    loss = nn.SmoothL1Loss()(q, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(q_net.state_dict())

            # done handling
            for i, done in enumerate(dones):
                if done:
                    returns.append(episode_rewards[i])
                    episode_rewards[i] = 0

        obs, _ = env.reset()

    env.close()
    return (rho, batch_size, target_update_freq, seed, returns)

# =====================
# MAIN
# =====================
if __name__ == "__main__":

    os.makedirs("results_sensitivity", exist_ok=True)

    batch_sizes = [64, 128, 256, 512]
    target_freqs = [250, 500, 1000, 2000]
    rhos = [1, 4]

    jobs = []

    for rho in rhos:
        for b in batch_sizes:
            for seed in range(NUM_SEEDS):
                jobs.append((seed, rho, b, 1000))

        for t in target_freqs:
            for seed in range(NUM_SEEDS):
                jobs.append((seed, rho, 256, t))

    print(f"Total jobs: {len(jobs)}")

    results = []

    with Pool(min(cpu_count(), NUM_WORKERS)) as pool:
        for out in tqdm(pool.imap_unordered(train_single, jobs), total=len(jobs)):
            results.append(out)

    # =====================
    # ORGANIZE RESULTS
    # =====================
    results_dict = {}

    for rho, b, t, seed, returns in results:
        key = (rho, b, t)
        if key not in results_dict:
            results_dict[key] = []
        results_dict[key].append(returns)

    summary_rows = []

    # =====================
    # BATCH PLOT
    # =====================
    plt.figure(figsize=(8,6))

    for rho in rhos:
        means, cis = [], []

        for b in batch_sizes:
            arr = np.array(results_dict[(rho, b, 1000)])

            pd.DataFrame(arr).to_csv(
                f"results_sensitivity/all_returns_batch_rho{rho}_b{b}.csv",
                index=False
            )

            final_perf = arr[:, -50:].mean(axis=1)

            mean = final_perf.mean()
            std = final_perf.std()
            ci = 1.96 * std / np.sqrt(NUM_SEEDS)

            means.append(mean)
            cis.append(ci)

            summary_rows.append([rho, b, 1000, mean, std, ci])

        plt.errorbar(batch_sizes, means, yerr=cis, label=f"rho={rho}", marker='o')

    plt.xscale('log')
    plt.title("Sensitivity to Batch Size")
    plt.legend()
    plt.grid(True)
    plt.savefig("results_sensitivity/sensitivity_batch.png", dpi=300)
    plt.show()

    # =====================
    # TARGET PLOT
    # =====================
    plt.figure(figsize=(8,6))

    for rho in rhos:
        means, cis = [], []

        for t in target_freqs:
            arr = np.array(results_dict[(rho, 256, t)])

            pd.DataFrame(arr).to_csv(
                f"results_sensitivity/all_returns_target_rho{rho}_t{t}.csv",
                index=False
            )

            final_perf = arr[:, -50:].mean(axis=1)

            mean = final_perf.mean()
            std = final_perf.std()
            ci = 1.96 * std / np.sqrt(NUM_SEEDS)

            means.append(mean)
            cis.append(ci)

            summary_rows.append([rho, 256, t, mean, std, ci])

        plt.errorbar(target_freqs, means, yerr=cis, label=f"rho={rho}", marker='o')

    plt.xscale('log')
    plt.title("Sensitivity to Target Update Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig("results_sensitivity/sensitivity_target.png", dpi=300)
    plt.show()

    # =====================
    # SAVE SUMMARY
    # =====================
    df = pd.DataFrame(summary_rows,
        columns=["rho", "batch_size", "target_update_freq", "mean", "std", "ci"]
    )
    df.to_csv("results_sensitivity/summary_all.csv", index=False)