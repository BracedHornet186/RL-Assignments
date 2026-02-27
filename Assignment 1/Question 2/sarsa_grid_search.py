import multiprocessing
from itertools import product
import pandas as pd
from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os

# Create a folder for the raw data
output_folder = "sarsa_grid_search_data"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

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
    
    # SARSA Update
    def update_sarsa(self, state, action, reward, terminated, next_state, next_action):
        future_q = (not terminated) * self.q_values[next_state][next_action]
        td_error = (reward + self.gamma * future_q) - self.q_values[state][action]
        self.q_values[state][action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

def train_worker(params):
    lr, start_eps = params
    env = gym.make("Acrobot-v1")
    n_episodes = 100000 
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
        
        # SARSA REQUIRES CHOOSING THE FIRST ACTION BEFORE THE LOOP
        action = agent.get_action(obs) 
        
        done = False
        total_reward = 0
        
        while not done:
            # 1. Take step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.bin_observation(next_obs)
            
            # 2. Choose NEXT action based on current policy (Epsilon-Greedy)
            next_action = agent.get_action(next_obs)
            
            # 3. Update using the SARSA rule
            agent.update_sarsa(state, action, reward, terminated, next_state, next_action)
            
            total_reward += reward
            
            # 4. Roll over state and action to the next step
            obs, state, action = next_obs, next_state, next_action
            
            done = terminated or truncated
        
        agent.decay_epsilon()
        episode_returns.append(total_reward)
    
    # --- SAVE RAW DATA TO UNIQUE FILE ---
    filename = f"{output_folder}/raw_lr_{lr}_eps_{start_eps}.csv"
    pd.DataFrame({"episode": range(n_episodes), "reward": episode_returns}).to_csv(filename, index=False)
    
    env.close()
    
    # Calculate score based on the last 100 episodes
    avg_score = np.mean(episode_returns[-100:]) 
    return {"learning_rate": lr, "initial_epsilon": start_eps, "score": avg_score}

if __name__ == "__main__":
    # Adjusted grid search values appropriate for SARSA
    lrs = [0.005, 0.01, 0.05, 0.1]
    eps_starts = [0.3, 0.5, 0.8, 1.0]
    
    grid = list(product(lrs, eps_starts))
    
    # Use min() to avoid spawning more processes than necessary
    num_processes = min(16, len(grid)) 
    
    print(f"Starting SARSA parallel tuning on {num_processes} processes...")
    print(f"Testing {len(grid)} hyperparameter combinations.")
    print(f"Raw data will be saved in the '{output_folder}' folder.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(train_worker, grid), total=len(grid)))

    df_tuning = pd.DataFrame(results)
    top_three = df_tuning.sort_values(by="score", ascending=False).head(3)
    
    print("\n--- TOP 3 SARSA HYPERPARAMETER SETTINGS ---")
    print(top_three)
    
    df_tuning.to_csv("sarsa_grid_search_summary.csv", index=False)