import os
import numpy as np
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

class RLAgent:
    def __init__(self, obs_space, action_space, algo="q_learning", lr=0.01, num_bins=10):
        self.algo = algo
        self.action_space_n = action_space.n
        self.lr = lr
        self.gamma = 0.99
        
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

    def get_action(self, state, epsilon):
        # Epsilon is passed dynamically now to handle online (decaying) vs offline (greedy = 0.0)
        if np.random.random() < epsilon:
            return np.random.randint(self.action_space_n)
        return np.argmax(self.q_values[state])

    def update(self, state, action, reward, terminated, next_state, next_action=None):
        state_action = state + (action,)
        current_q = self.q_values[state_action]
        
        if terminated:
            future_q = 0.0
        else:
            if self.algo == "q_learning":
                # Off-policy: assumes greedy action for the next state
                future_q = np.max(self.q_values[next_state])
            elif self.algo == "sarsa":
                # On-policy: uses the actual next action taken
                future_q = self.q_values[next_state + (next_action,)]
        
        self.q_values[state_action] = current_q + self.lr * (reward + self.gamma * future_q - current_q)


def run_experiment(params):
    algo, seed = params
    
    # Initialize env with specific seed for reproducibility
    env = gym.make("Acrobot-v1")
    agent = RLAgent(env.observation_space, env.action_space, algo=algo, lr=0.01)
    
    # Epsilon parameters
    epsilon = 1.0
    epsilon_decay = 0.99  # Adjust if it learns too fast/slow
    min_epsilon = 0.1
    
    # Phase 1: Online Performance (Learning)
    online_episodes = 50000
    online_returns = np.zeros(online_episodes, dtype=np.float32)
    
    for episode in range(online_episodes):
        obs, _ = env.reset(seed=seed + episode)
        state = agent.bin_observation(obs)
        action = agent.get_action(state, epsilon)
        
        total_reward = 0.0
        done = False
        
        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.bin_observation(next_obs)
            
            # SARSA needs the next action before the update
            next_action = agent.get_action(next_state, epsilon)
            
            # Update Q-table
            agent.update(state, action, reward, terminated, next_state, next_action)
            
            total_reward += reward
            state = next_state
            action = next_action  # Move to the next action
            done = terminated or truncated
            
        # Decay epsilon to a minimum of 0.1
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        online_returns[episode] = total_reward

    # Phase 2: Offline Performance (Evaluation without exploration)
    offline_episodes = 100
    offline_returns = np.zeros(offline_episodes, dtype=np.float32)
    eval_epsilon = 0.0  # Fully greedy policy
    
    for episode in range(offline_episodes):
        obs, _ = env.reset(seed=seed + 10000 + episode)
        state = agent.bin_observation(obs)
        done = False
        total_reward = 0.0
        
        while not done:
            action = agent.get_action(state, eval_epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            total_reward += reward
            state = agent.bin_observation(next_obs)
            done = terminated or truncated
            
        offline_returns[episode] = total_reward
        
    env.close()
    
    return {
        "algo": algo, 
        "seed": seed, 
        "online_returns": online_returns, 
        "offline_mean": np.mean(offline_returns)
    }

if __name__ == "__main__":
    algorithms = ["q_learning", "sarsa"]
    seeds = [42, 101, 2024] # Running a few seeds to get an average performance
    grid = [(algo, seed) for algo in algorithms for seed in seeds]
    
    results = []
    print("Running Online Training & Offline Evaluation for Q-Learning and SARSA...")
    
    max_workers = min(16, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_experiment, p) for p in grid]
        for future in tqdm(as_completed(futures), total=len(grid)):
            results.append(future.result())

    # Process Data
    df_results = pd.DataFrame(results)
    
    # 1. Compare Offline Performance
    print("\n--- Offline Performance (Greedy Policy, Epsilon=0) ---")
    offline_summary = df_results.groupby("algo")["offline_mean"].agg(['mean', 'std']).reset_index()
    print(offline_summary.to_string(index=False))
    
    # 2. Plot Online Performance
    print("\nPlotting Online Performance...")
    plt.figure(figsize=(10, 6))
    
    for algo in algorithms:
        # Get all online returns for this algorithm across all seeds
        algo_data = df_results[df_results["algo"] == algo]["online_returns"].values
        # Stack and calculate the mean across seeds
        mean_returns = np.mean(np.vstack(algo_data), axis=0)
        
        # Calculate a rolling average for a smoother plot
        rolling_mean = pd.Series(mean_returns).rolling(window=1000, min_periods=1).mean()
        
        plt.plot(rolling_mean, label=f"{algo.upper()} (Rolling Mean)")

    

    plt.title("Online Performance: Q-Learning vs SARSA")
    plt.xlabel("Episode")
    plt.ylabel("Reward (50-Episode Rolling Avg)")
    plt.legend()
    plt.grid(True)
    
    # Save the plot and data
    output_dir = os.path.join("Assignment 1", "Question 2", "SARSA_Q_Learning_Comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, "online_performance_comparison.png"))
    df_results.drop(columns=['online_returns']).to_csv(os.path.join(output_dir, "offline_comparison_summary.csv"), index=False)
    print(f"Plot and summary saved to {output_dir}")