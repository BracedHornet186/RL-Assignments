import multiprocessing
from itertools import product
import pandas as pd
from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os

# Create a folder for the raw data to keep your directory clean
if not os.path.exists("training_data"):
    os.makedirs("training_data")

class AcrobatAgent:
    def __init__(
            self, 
            env: gym.Env,
            learning_rate: float, 
            initial_epsilon: float, 
            epsilon_decay: float,
            final_epsilon: float, 
            gamma: float = 0.99,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.gamma = gamma

        # Pre-compute bins
        self.low = self.env.observation_space.low
        self.high = self.env.observation_space.high
        self.num_bins = 10
        self.bin_edges = [np.linspace(self.low[i], self.high[i], self.num_bins + 1)[1:-1] for i in range(6)]

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def get_action(self, obs):
        state = self.bin_observation(obs)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[state]))
        
    def bin_observation(self, obs):
        return tuple(int(np.digitize(obs[i], self.bin_edges[i])) for i in range(6))
    
    def update_q(self, state, action, reward, terminated, next_state):
        future_q = (not terminated) * np.max(self.q_values[next_state])
        td_error = (reward + self.gamma * future_q) - self.q_values[state][action]
        self.q_values[state][action] += self.lr * td_error

    # Added SARSA update in case you need to compare later
    def update_sarsa(self, state, action, reward, terminated, next_state, next_action):
        future_q = (not terminated) * self.q_values[next_state][next_action]
        td_error = (reward + self.gamma * future_q) - self.q_values[state][action]
        self.q_values[state][action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

def train_worker(params):
    lr, start_eps = params
    env = gym.make("Acrobot-v1")
    n_episodes = 100000 # Adjusted episodes for clearer tuning
    exp_decay_rate = 0.99 
    
    agent = AcrobatAgent(
        env=env,
        learning_rate=lr,
        initial_epsilon=start_eps,
        epsilon_decay=exp_decay_rate,
        final_epsilon=0.1
    )

    episode_returns = []
    
    for episode in range(n_episodes):
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
            obs, state = next_obs, next_state
            done = terminated or truncated
        
        agent.decay_epsilon()
        episode_returns.append(total_reward)
    
    # --- 1. SAVE RAW DATA TO UNIQUE FILE ---
    # Filename format: raw_lr_0.1_eps_0.5.csv
    filename = f"training_data/raw_lr_{lr}_eps_{start_eps}.csv"
    pd.DataFrame({"episode": range(n_episodes), "reward": episode_returns}).to_csv(filename, index=False)
    
    env.close()
    
    avg_score = np.mean(episode_returns[-100:]) 
    return {"learning_rate": lr, "initial_epsilon": start_eps, "score": avg_score}

if __name__ == "__main__":
    lrs = [0.005, 0.01, 0.05]
    eps_starts = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    grid = list(product(lrs, eps_starts))
    
    print(f"Starting Q-Learning parallel tuning on 16 threads...")
    print(f"Raw data will be saved in the 'training_data' folder.")

    with multiprocessing.Pool(processes=16) as pool:
        results = list(tqdm(pool.imap(train_worker, grid), total=len(grid)))

    df_tuning = pd.DataFrame(results)
    top_three = df_tuning.sort_values(by="score", ascending=False).head(3)
    
    print("\n--- TOP 3 Q-LEARNING HYPERPARAMETER SETTINGS ---")
    print(top_three)
    
    df_tuning.to_csv("q_learning_grid_search_summary.csv", index=False)