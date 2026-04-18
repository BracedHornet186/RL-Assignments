import numpy as np
import random
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
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
# Hyperparameters (FINAL CONFIG)
# =====================
start = time.time()

GAMMA = 0.99
LR = 5e-4
BUFFER_SIZE = 100000

BATCH_SIZE = 256
TARGET_UPDATE_FREQ = 1000
TRAIN_FREQ = 1
RHO = 4

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 100000

HIDDEN_SIZES = [64, 64]

NUM_EPISODES = 600
MAX_STEPS = 2000
NUM_SEEDS = 15

DEVICE = torch.device("cpu")
torch.set_num_threads(1)
os.makedirs("results", exist_ok=True)
os.makedirs("videos", exist_ok=True)

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
# Record Final Policy
# =====================
def record_final_agent(q_net, seed, save_path):
    env = gym.make(
        "MountainCar-v0",
        render_mode="rgb_array",
        max_episode_steps=MAX_STEPS
    )

    env = RecordVideo(
        env,
        video_folder=save_path,
        episode_trigger=lambda ep: True
    )

    state, _ = env.reset(seed=seed)
    done = False

    while not done:
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = torch.argmax(q_net(state_t)).item()

        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()

# =====================
# Train Function
# =====================
def train_dqn(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(
        "MountainCar-v0",
        render_mode="rgb_array",
        max_episode_steps=MAX_STEPS
    )

    # Record only one seed to avoid huge storage
    if seed == 0:
        env = RecordVideo(
            env,
            video_folder=f"videos/train_rho4_bs256_tu1000",
            episode_trigger=lambda ep: ep % 50 == 0
        )

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

            # Training
            if len(buffer) > BATCH_SIZE and total_steps % TRAIN_FREQ == 0:
                for _ in range(RHO):
                    s, a, r, s_next, d = buffer.sample(BATCH_SIZE)

                    q_vals = q_net(s).gather(1, a.unsqueeze(1)).squeeze()

                    with torch.no_grad():
                        max_next_q = target_net(s_next).max(1)[0]
                        target = r + GAMMA * max_next_q * (1 - d)

                    loss = nn.SmoothL1Loss()(q_vals, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Target update
            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(q_net.state_dict())

            if terminated or truncated:
                break

        episode_returns.append(ep_reward)

    # Record final policy
    if seed == 0:
        record_final_agent(q_net, seed, "videos/final_policy")

    env.close()
    return episode_returns

# =====================
# MAIN
# =====================
if __name__ == "__main__":

    print("\n===== FINAL TRAINING (rho=4, B=256, C=1000) =====")

    with Pool(min(6, NUM_SEEDS)) as pool:
        results = list(tqdm(pool.imap(train_dqn, range(NUM_SEEDS)), total=NUM_SEEDS))

    returns = np.array(results)

    # Save results
    pd.DataFrame(returns).to_csv("results/final_returns.csv", index=False)

    # Compute stats
    final_perf = returns[:, -100:].mean(axis=1)
    mean = final_perf.mean()
    std = final_perf.std()
    ci = 1.96 * std / np.sqrt(NUM_SEEDS)

    print(f"\nFinal Performance → Mean={mean:.2f}, CI={ci:.2f}")
    print(f"\nTotal Time: {(time.time() - start)/60:.2f} minutes")