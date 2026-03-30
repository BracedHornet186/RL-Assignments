import numpy as np
import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from multiprocessing import Pool, cpu_count
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
LR = 5e-4 
BUFFER_SIZE = 100000
BATCH_SIZE = 256
TARGET_UPDATE_FREQ = 1000
TRAIN_FREQ = 2

EPS_START = 1.0
EPS_END = 0.05
# Linear decay over 100,000 steps is usually safer for MountainCar
EPS_DECAY_STEPS = 100000 

HIDDEN_SIZES = [64, 64]  

NUM_EPISODES = 600
MAX_STEPS = 2000 
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

        s = torch.stack(s).to(DEVICE)
        a = torch.tensor(a, dtype=torch.long).to(DEVICE)
        r = torch.tensor(r, dtype=torch.float32).to(DEVICE)
        s_next = torch.stack(s_next).to(DEVICE)
        # Note: 'done' should only be True if terminal state reached, 
        # not if truncated (though in standard Gym MC, they often align)
        done = torch.tensor(done, dtype=torch.float32).to(DEVICE)

        return s, a, r, s_next, done

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
        
        # Assignment Requirement: Kaiming Initialization
        self.apply(self.kaiming_init)

    def kaiming_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# =====================
# Train One Seed
# =====================
def train_dqn(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Note: Setting max_episode_steps here to 2000
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
    pbar = tqdm(range(NUM_EPISODES), desc=f"Seed {seed}", position=seed, leave=False)
    for ep in pbar:
        state, _ = env.reset(seed=seed + ep) # Seed per episode for variety
        ep_reward = 0

        for t in range(MAX_STEPS):
            # Linear Epsilon Decay
            epsilon = max(EPS_END, EPS_START - total_steps / EPS_DECAY_STEPS)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
                    action = torch.argmax(q_net(state_t)).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # The environment gives -1 per step. 
            # 'done' for the Bellman equation should only be True if the goal is reached (terminated)
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

                # loss = nn.MSELoss()(q_vals, target)
                loss = nn.SmoothL1Loss()(q_vals, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(q_net.state_dict())

            if terminated or truncated:
                break
        # if ep % 50 == 0:
        #     print(f"[Seed {seed}] Episode {ep}, Return={ep_reward:.1f}, Eps={epsilon:.3f}")
        if ep % 50 == 0:
            pbar.set_postfix({
                "Return": f"{ep_reward:.1f}",
                "Eps": f"{epsilon:.3f}"
            })
        episode_returns.append(ep_reward)

    env.close()
    return episode_returns

if __name__ == "__main__":
    print(os.cpu_count())
    # Standard parallel setup
    num_workers = min(8, NUM_SEEDS) 
    print(f"Using {num_workers} processes for {NUM_SEEDS} seeds...")

    with Pool(num_workers) as pool:
        all_returns = list(tqdm(pool.imap(train_dqn, range(NUM_SEEDS)), total=NUM_SEEDS))

    all_returns = np.array(all_returns)

    # Calculate Mean and 95% Confidence Interval
    mean_returns = all_returns.mean(axis=0)
    std_returns = all_returns.std(axis=0)
    ci = 1.96 * std_returns / np.sqrt(NUM_SEEDS)
    df = pd.DataFrame(all_returns)
    df.to_csv("all_returns.csv", index=False)

    summary_df = pd.DataFrame({
        "episode": np.arange(NUM_EPISODES),
        "mean": mean_returns,
        "ci_lower": mean_returns - ci,
        "ci_upper": mean_returns + ci,
        "std_returns" : std_returns
    })
    summary_df.to_csv("summary.csv", index=False)
    plt.figure(figsize=(10, 6))
    plt.plot(mean_returns, label="Mean Return")
    plt.fill_between(
        range(NUM_EPISODES),
        mean_returns - ci,
        mean_returns + ci,
        alpha=0.3,
        label="95% Confidence Interval"
    )
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.title("Vanilla DQN on MountainCar-v0 (Max Steps = 2000)")
    plt.legend()
    plt.grid(True)
    plt.savefig("dqn_mountaincar.png", dpi=300)
    plt.show()
    print(f"\nTotal Time: {(time.time() - start)/60:.2f} minutes")