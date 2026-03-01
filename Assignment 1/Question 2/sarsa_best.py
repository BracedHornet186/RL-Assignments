import os
import numpy as np
import gymnasium as gym
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Create the specific folder for the best SARSA runs
output_folder = os.path.join("Assignment 1", "Question 2", "sarsa_best")
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

        # Pre-allocate the Q-table (Fast and memory efficient)
        self.q_shape = tuple([num_bins + 1] * obs_space.shape[0] + [self.action_space_n])
        self.q_values = np.zeros(self.q_shape, dtype=np.float32)

    def bin_observation(self, obs):
        return tuple(int(np.digitize(obs[i], self.bin_edges[i])) for i in range(len(obs)))

    def get_action(self, state):
        # Accepts 'state' directly to avoid double binning
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


def train_worker(seed):
    # --- FIXED HYPERPARAMETERS ---
    lr = 0.1
    start_eps = 1.0
    n_episodes = 100000 
    exp_decay_rate = 0.9
    final_eps = 0.1
    
    # Set seeds for NumPy and Gym spaces to ensure strict reproducibility
    np.random.seed(seed)
    env = gym.make("Acrobot-v1")
    env.action_space.seed(seed)
    
    agent = SarsaAgent(
        env=env,
        learning_rate=lr,
        initial_epsilon=start_eps,
        epsilon_decay=exp_decay_rate,
        final_epsilon=final_eps
    )

    # Pre-allocate numpy array for memory efficiency
    episode_returns = np.zeros(n_episodes, dtype=np.float32)
    
    for episode in range(n_episodes):
        # Seed incrementally to maintain determinism but ensure episode variety
        obs, _ = env.reset(seed=seed + episode)
        state = agent.bin_observation(obs)
        
        # SARSA: Select initial action before the step loop
        action = agent.get_action(state)
        
        done = False
        total_reward = 0.0
        
        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.bin_observation(next_obs)
            
            # SARSA: Select the next action
            next_action = agent.get_action(next_state)
            
            # SARSA: Update Q-values using the actual next action
            agent.update_sarsa(state, action, reward, terminated, next_state, next_action)
            
            total_reward += reward
            
            # Roll over to the next state and action
            state, action = next_state, next_action
            done = terminated or truncated
        
        agent.decay_epsilon()
        episode_returns[episode] = total_reward
    
    env.close()
    
    # --- SAVE RAW DATA TO SEED-SPECIFIC FILE ---
    filename = os.path.join(output_folder, f"run_seed_{seed}.csv")
    pd.DataFrame({"episode": range(n_episodes), "reward": episode_returns}).to_csv(filename, index=False)
    
    avg_score = np.mean(episode_returns[-100:]) 
    return {"seed": seed, "score": avg_score}


if __name__ == "__main__":
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    results = []
    
    print(f"Starting SARSA runs on {len(seeds)} seeds with LR=0.1, deacy=0.9...")
    print(f"Data will be saved in: '{output_folder}'")

    # Run in parallel using modern ProcessPoolExecutor
    max_workers = min(10, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train_worker, seed) for seed in seeds]
        
        for future in tqdm(as_completed(futures), total=len(seeds), desc="Training SARSA Seeds"):
            results.append(future.result())

    df_results = pd.DataFrame(results).sort_values(by="seed")
    
    print("\n--- SARSA BEST RUN SUMMARY ---")
    print(df_results.to_string(index=False))
    print(f"\nAverage Score across all seeds: {df_results['score'].mean():.2f}")
    
    df_results.to_csv(os.path.join(output_folder, "summary_statistics.csv"), index=False)