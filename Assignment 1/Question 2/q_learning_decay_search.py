import os
import numpy as np
import gymnasium as gym
import pandas as pd
from tqdm import tqdm
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

class AcrobotAgent:
    def __init__(self, obs_space, action_space, lr, initial_eps, decay, final_eps, num_bins=10):
        self.action_space_n = action_space.n
        self.lr = lr
        self.gamma = 0.99
        self.epsilon = initial_eps
        self.epsilon_decay = decay
        self.final_epsilon = final_eps
        
        # Pre-allocate the Q-table
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

    def update_q(self, state, action, reward, terminated, next_state):
        future_q = 0.0 if terminated else np.max(self.q_values[next_state])
        
        state_action = state + (action,)
        current_q = self.q_values[state_action]
        
        self.q_values[state_action] = current_q + self.lr * (reward + self.gamma * future_q - current_q)

    def decay_epsilon(self):
        # Decays epsilon until it hits the final threshold
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)


def train_worker(params):
    lr, decay = params  # Unpack the learning rate and decay rate

    start_eps = 1.0
    final_eps = 0.1
    
    # Initialize env and agent inside the worker
    env = gym.make("Acrobot-v1")
    agent = AcrobotAgent(env.observation_space, env.action_space, lr, start_eps, decay, final_eps)

    num_episodes = 10000
    episode_returns = np.zeros(num_episodes, dtype=np.float32)
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = agent.bin_observation(obs)
        done = False
        total_reward = 0.0
        
        while not done:
            action = agent.get_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.bin_observation(next_obs)
            
            agent.update_q(state, action, reward, terminated, next_state)
            
            total_reward += reward
            state = next_state
            done = terminated or truncated
            
        agent.decay_epsilon()
        episode_returns[episode] = total_reward
    
    env.close()
    
    # Save individual raw data file safely
    output_folder = os.path.join("Assignment 1", "Question 2", "q_learning_decay_search_data")
    os.makedirs(output_folder, exist_ok=True)
    
    file_path = os.path.join(output_folder, f"raw_lr_{lr}_decay_{decay}.csv")
    pd.DataFrame({"episode": range(num_episodes), "reward": episode_returns}).to_csv(file_path, index=False)
    
    return {"lr": lr, "decay": decay, "score": np.mean(episode_returns[-100:])}


if __name__ == "__main__":
    # Standard parameters to test
    lrs = [0.01, 0.05, 0.1, 0.2]
    decays = [0.9, 0.95, 0.99, 0.995, 0.999]

    grid = list(product(lrs, decays))
    results = []
    
    print(f"Starting grid search over {len(grid)} combinations...")
    
    max_workers = min(16, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train_worker, params) for params in grid]
        
        for future in tqdm(as_completed(futures), total=len(grid), desc="Training Agents"):
            results.append(future.result())
    
    # Summarize results
    df_results = pd.DataFrame(results).sort_values(by="score", ascending=False)
    
    # Define summary path
    summary_dir = os.path.join("Assignment 1", "Question 2")
    os.makedirs(summary_dir, exist_ok=True)
    
    summary_path = os.path.join(summary_dir, "q_learning_decay_summary.csv")
    
    # Save the summary to CSV
    df_results.to_csv(summary_path, index=False)
    print(f"\nFull grid search results saved to: {summary_path}")
    
    # Output top 3 to terminal
    print("\nTop 3 Combinations (LR & Decay):")
    print(df_results.head(3).to_string(index=False))