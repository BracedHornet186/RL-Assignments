import multiprocessing
import numpy as np
import gymnasium as gym
import pandas as pd
import os
from tqdm import tqdm
from itertools import product

if not os.path.exists("training_data"):
    os.makedirs("training_data")

class AcrobatAgent:
    def __init__(self, env, lr, initial_eps, decay, final_eps, num_bins=10):
        self.env = env
        self.lr = lr
        self.gamma = 0.99
        self.num_bins = num_bins
        
        # Pre-allocate 11x11x11x11x11x11x3 array for speed
        # Using float32 saves 50% memory over float64
        shape = tuple([num_bins + 1] * 6 + [env.action_space.n])
        self.q_values = np.zeros(shape, dtype=np.float32)

        self.low = self.env.observation_space.low
        self.high = self.env.observation_space.high
        self.bin_edges = [np.linspace(self.low[i], self.high[i], num_bins + 1)[1:-1] for i in range(6)]

        self.epsilon = initial_eps
        self.epsilon_decay = decay
        self.final_epsilon = final_eps

    def bin_observation(self, obs):
        return tuple(int(np.digitize(obs[i], self.bin_edges[i])) for i in range(6))

    def get_action(self, obs):
        state = self.bin_observation(obs)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_values[state])

    def update_q(self, state, action, reward, terminated, next_state):
        future_q = (not terminated) * np.max(self.q_values[next_state])
        self.q_values[state][action] += self.lr * (reward + self.gamma * future_q - self.q_values[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

def train_worker(params):
    lr, start_eps = params
    env = gym.make("Acrobot-v1")
    agent = AcrobatAgent(env, lr, start_eps, 0.99, 0.1)
    
    episode_returns = []
    for episode in range(2000): # Using 2k for the grid search
        obs, _ = env.reset()
        state = agent.bin_observation(obs)
        done = False
        total_reward = 0
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.bin_observation(next_obs)
            agent.update_q(state, action, reward, terminated, next_state)
            total_reward += reward
            state = next_state
            obs = next_obs
            done = terminated or truncated
        agent.decay_epsilon()
        episode_returns.append(total_reward)
    
    # Save individual raw data file
    filename = f"training_data/raw_lr_{lr}_eps_{start_eps}.csv"
    pd.DataFrame({"episode": range(len(episode_returns)), "reward": episode_returns}).to_csv(filename, index=False)
    
    env.close()
    return {"lr": lr, "eps": start_eps, "score": np.mean(episode_returns[-100:])}

if __name__ == "__main__":
    lrs = [0.01, 0.1, 0.2, 0.5]
    eps_starts = [0.5, 0.8, 1.0]
    grid = list(product(lrs, eps_starts))
    
    with multiprocessing.Pool(processes=16) as pool:
        results = list(tqdm(pool.imap(train_worker, grid), total=len(grid)))
    
    print(pd.DataFrame(results).sort_values(by="score", ascending=False).head(3))