import os
import numpy as np
import gymnasium as gym
import pandas as pd
from tqdm import tqdm
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

# Create the specific folder for the bin experiments
output_folder = os.path.join("Assignment 1", "Question 2", "sarsa_bin_search_data")
os.makedirs(output_folder, exist_ok=True)

class SarsaAgent:
    def __init__(
            self, 
            env: gym.Env,
            learning_rate: float, 
            initial_epsilon: float, 
            epsilon_decay: float,
            final_epsilon: float, 
            num_bins: int = 10,
            gamma: float = 0.99,
    ):
        self.action_space_n = env.action_space.n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Pre-compute bins
        obs_space = env.observation_space
        self.bin_edges = [
            np.linspace(obs_space.low[i], obs_space.high[i], num_bins + 1)[1:-1] 
            for i in range(obs_space.shape[0])
        ]

        # Pre-allocate the Q-table
        self.q_shape = tuple([num_bins + 1] * obs_space.shape[0] + [self.action_space_n])
        self.q_values = np.zeros(self.q_shape, dtype=np.float32)

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
        
        self.q_values[state_action] = current_q + self.lr * (reward + self.gamma * future_q - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)


def train_worker(params):
    seed, num_bins = params
    
    # --- FIXED HYPERPARAMETERS ---
    lr = 0.1
    start_eps = 1.0
    n_episodes = 50000 
    exp_decay_rate = 0.9
    final_eps = 0.1
    
    # Set seeds for NumPy and Gym spaces
    np.random.seed(seed)
    env = gym.make("Acrobot-v1")
    env.action_space.seed(seed)
    
    # Pass num_bins to the agent
    agent = SarsaAgent(
        env=env,
        learning_rate=lr,
        initial_epsilon=start_eps,
        epsilon_decay=exp_decay_rate,
        final_epsilon=final_eps,
        num_bins=num_bins
    )

    episode_returns = np.zeros(n_episodes, dtype=np.float32)
    
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        state = agent.bin_observation(obs)
        
        action = agent.get_action(state)
        
        done = False
        total_reward = 0.0
        
        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.bin_observation(next_obs)
            
            next_action = agent.get_action(next_state)
            
            agent.update_sarsa(state, action, reward, terminated, next_state, next_action)
            
            total_reward += reward
            
            state, action = next_state, next_action
            done = terminated or truncated
        
        agent.decay_epsilon()
        episode_returns[episode] = total_reward
    
    env.close()
    
    # --- SAVE RAW DATA WITH BINS IN FILENAME ---
    filename = os.path.join(output_folder, f"run_seed_{seed}_bins_{num_bins}.csv")
    pd.DataFrame({"episode": range(n_episodes), "reward": episode_returns}).to_csv(filename, index=False)
    
    avg_score = np.mean(episode_returns[-100:]) 
    return {"seed": seed, "num_bins": num_bins, "score": avg_score}


if __name__ == "__main__":
    seeds = [42, 101, 202]
    bins_list = [5, 10, 15, 20]
    
    # Create the grid of (seed, num_bins) combinations
    grid = list(product(seeds, bins_list))
    results = []
    
    print(f"Starting SARSA runs: {len(seeds)} seeds x {len(bins_list)} bin configurations = {len(grid)} total tasks.")
    print(f"Data will be saved in: '{output_folder}'")

    max_workers = min(12, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train_worker, params) for params in grid]
        
        for future in tqdm(as_completed(futures), total=len(grid), desc="Training SARSA Agents"):
            results.append(future.result())

    # --- PROCESS AND DISPLAY RESULTS ---
    df_results = pd.DataFrame(results).sort_values(by=["num_bins", "seed"])
    df_results.to_csv(os.path.join(output_folder, "detailed_summary.csv"), index=False)
    
    print("\n--- DETAILED RUN SUMMARY ---")
    print(df_results.to_string(index=False))
    
    print("\n--- PERFORMANCE BY NUM_BINS (Averaged across seeds) ---")
    summary_by_bins = df_results.groupby("num_bins")["score"].agg(['mean', 'std']).reset_index()
    summary_by_bins = summary_by_bins.rename(columns={"mean": "avg_score", "std": "score_std"})
    print(summary_by_bins.to_string(index=False))
    
    summary_by_bins.to_csv(os.path.join(output_folder, "bins_aggregated_summary.csv"), index=False)