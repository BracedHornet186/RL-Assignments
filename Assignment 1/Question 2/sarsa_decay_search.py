import os
import numpy as np
import gymnasium as gym
import pandas as pd
from tqdm import tqdm
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

class SarsaAgent:
    def __init__(self, obs_space, action_space, lr, initial_eps, decay, final_eps, num_bins=10):
        self.action_space_n = action_space.n
        self.lr = lr
        self.gamma = 0.99
        self.epsilon = initial_eps
        self.epsilon_decay = decay
        self.final_epsilon = final_eps
        
        # Pre-allocate the Q-table (much faster than defaultdict for grid searches)
        self.q_shape = tuple([num_bins + 1] * obs_space.shape[0] + [self.action_space_n])
        self.q_values = np.zeros(self.q_shape, dtype=np.float32)

        # Pre-compute bin edges
        self.bin_edges = [
            np.linspace(obs_space.low[i], obs_space.high[i], num_bins + 1)[1:-1] 
            for i in range(obs_space.shape[0])
        ]

    def bin_observation(self, obs):
        return tuple(int(np.digitize(obs[i], self.bin_edges[i])) for i in range(len(obs)))

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_n)
        return np.argmax(self.q_values[state])

    def update_sarsa(self, state, action, reward, terminated, next_state, next_action):
        future_q = 0.0 if terminated else self.q_values[next_state + (next_action,)]
        
        state_action = state + (action,)
        current_q = self.q_values[state_action]
        
        # SARSA Update Rule: Uses the actual next action taken (On-Policy)
        self.q_values[state_action] = current_q + self.lr * (reward + self.gamma * future_q - current_q)

    def decay_epsilon(self):
        # Decays epsilon until it hits the final threshold
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)


def train_worker(params):
    lr, decay = params  

    start_eps = 1.0
    final_eps = 0.1
    num_episodes = 10000
    
    # Initialize env and agent inside the worker
    env = gym.make("Acrobot-v1")
    agent = SarsaAgent(env.observation_space, env.action_space, lr, start_eps, decay, final_eps)

    episode_returns = np.zeros(num_episodes, dtype=np.float32)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = agent.bin_observation(obs)
        
        # SARSA requires selecting the first action before the loop begins
        action = agent.get_action(state)
        
        done = False
        total_reward = 0.0
        
        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.bin_observation(next_obs)
            
            # Select the NEXT action based on the current policy
            next_action = agent.get_action(next_state)
            
            # Update Q-values using SARSA
            agent.update_sarsa(state, action, reward, terminated, next_state, next_action)
            
            total_reward += reward
            
            # Roll over state and action
            state = next_state
            action = next_action
            
            done = terminated or truncated
            
        # Decay epsilon at the end of the episode
        agent.decay_epsilon()
        episode_returns[episode] = total_reward
    
    env.close()
    
    # Create the data folder if it doesn't exist
    output_folder = os.path.join("Assignment 1", "Question 2", "sarsa_decay_search_data")
    os.makedirs(output_folder, exist_ok=True)
    
    # Save individual raw data file safely, reflecting both lr and decay
    file_path = os.path.join(output_folder, f"raw_lr_{lr}_decay_{decay}.csv")
    pd.DataFrame({"episode": range(num_episodes), "reward": episode_returns}).to_csv(file_path, index=False)
    
    return {"lr": lr, "decay": decay, "score": np.mean(episode_returns[-100:])}


if __name__ == "__main__":
    # Hyperparameters for the grid search
    lrs = [0.005, 0.01, 0.05, 0.1]
    decays = [0.9, 0.95, 0.99, 0.995, 0.999]

    grid = list(product(lrs, decays))
    results = []
    
    print(f"Starting SARSA Grid Search over {len(grid)} combinations...")
    
    max_workers = min(16, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train_worker, params) for params in grid]
        
        for future in tqdm(as_completed(futures), total=len(grid), desc="Training SARSA Agents"):
            results.append(future.result())
    
    # Summarize results
    df_results = pd.DataFrame(results).sort_values(by="score", ascending=False)
    
    # Save the summary to the root directory
    summary_path = os.path.join("Assignment 1", "Question 2", "sarsa_decay_summary.csv")
    df_results.to_csv(summary_path, index=False)
    print(f"\nFull SARSA decay search results saved to: {summary_path}")
    print(f"Raw episode data saved in: sarsa_decay_search_data/")
    
    # Output top 3 to terminal
    print("\nTop 3 Sarsa Combinations (LR & Decay):")
    print(df_results.head(3).to_string(index=False))