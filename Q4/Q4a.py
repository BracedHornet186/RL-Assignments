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
start = time.time()
GAMMA = 0.99
LR = 5e-4 
BUFFER_SIZE = 100000
BATCH_SIZE = 256
TARGET_UPDATE_FREQ = 1000
TRAIN_FREQ = 2

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 100000 

HIDDEN_SIZES = [64, 64]  

NUM_EPISODES = 600
MAX_STEPS = 2000 
NUM_SEEDS = 15

RHO_VALUES = [1, 2, 4, 8]

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
# Train One Seed + RHO
# =====================
def train_dqn(args):
    seed, rho = args

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

    pbar = tqdm(range(NUM_EPISODES), desc=f"Seed {seed}, ρ={rho}", leave=False)

    for ep in pbar:
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

            # =====================
            # Replay Factor Update
            # =====================
            if len(buffer) > BATCH_SIZE and total_steps % TRAIN_FREQ == 0:
                for _ in range(rho):
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
            pbar.set_postfix({
                "Return": f"{ep_reward:.1f}",
                "Eps": f"{epsilon:.3f}"
            })

        episode_returns.append(ep_reward)

    env.close()
    return (rho, episode_returns)

# =====================
# Main
# =====================
if __name__ == "__main__":
    num_workers = min(6, NUM_SEEDS)
    print(f"Using {num_workers} processes...")

    jobs = [(seed, rho) for rho in RHO_VALUES for seed in range(NUM_SEEDS)]

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(train_dqn, jobs), total=len(jobs)))

    # =====================
    # Organize Results
    # =====================
    grouped = {rho: [] for rho in RHO_VALUES}

    for rho, returns in results:
        grouped[rho].append(returns)

    plt.figure(figsize=(12, 7))

    for rho in RHO_VALUES:
        data = np.array(grouped[rho])

        mean_returns = data.mean(axis=0)
        std_returns = data.std(axis=0)
        ci = 1.96 * std_returns / np.sqrt(NUM_SEEDS)

        # Save per-rho CSV
        pd.DataFrame(data).to_csv(f"all_returns_rho_{rho}.csv", index=False)

        summary_df = pd.DataFrame({
            "episode": np.arange(NUM_EPISODES),
            "mean": mean_returns,
            "ci_lower": mean_returns - ci,
            "ci_upper": mean_returns + ci,
            "std": std_returns
        })
        summary_df.to_csv(f"summary_rho_{rho}.csv", index=False)

        # Plot
        plt.plot(mean_returns, label=f"ρ = {rho}")
        plt.fill_between(
            range(NUM_EPISODES),
            mean_returns - ci,
            mean_returns + ci,
            alpha=0.2
        )

    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.title("DQN with Different Replay Factors (MountainCar-v0)")
    plt.legend()
    plt.grid(True)

    plt.savefig("dqn_rho_comparison.png", dpi=300)
    plt.show()

    print(f"Total Time: {(time.time() - start)/60:.2f} minutes")    