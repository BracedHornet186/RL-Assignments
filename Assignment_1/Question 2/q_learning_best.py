import multiprocessing
import pandas as pd
from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os

# Create the specific folder for the best runs
output_folder = os.path.join("Assignment 1", "Question 2", "q_learning_best")
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
    
    def update_q(self, state, action, reward, terminated, next_state):
        future_q = (not terminated) * np.max(self.q_values[next_state])
        td_error = (reward + self.gamma * future_q) - self.q_values[state][action]
        self.q_values[state][action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

def train_worker(seed):
    # --- FIXED HYPERPARAMETERS ---
    lr = 0.05
    start_eps = 1.0
    n_episodes = 100000 
    exp_decay_rate = 0.99 
    final_eps = 0.1
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    env = gym.make("Acrobot-v1")
    
    agent = AcrobatAgent(
        env=env,
        learning_rate=lr,
        initial_epsilon=start_eps,
        epsilon_decay=exp_decay_rate,
        final_epsilon=final_eps
    )

    episode_returns = []
    
    # Pass seed to reset for environment reproducibility
    obs, _ = env.reset(seed=seed)
    
    for episode in range(n_episodes):
        obs, _ = env.reset() # subsequent resets don't strictly need seed unless you want identical eps sequences
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
    
    # --- SAVE RAW DATA TO SEED-SPECIFIC FILE ---
    filename = f"{output_folder}/run_seed_{seed}.csv"
    pd.DataFrame({"episode": range(n_episodes), "reward": episode_returns}).to_csv(filename, index=False)
    
    env.close()
    
    avg_score = np.mean(episode_returns[-100:]) 
    return {"seed": seed, "score": avg_score}

if __name__ == "__main__":
    # Generate 10 random seeds
    # You can fix these specific integers if you want strict reproducibility across runs of this script
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    
    print(f"Starting Q-Learning runs on 10 seeds with LR=0.05, Eps=1.0...")
    print(f"Data will be saved in the '{output_folder}' folder.")

    # Using multiprocessing to run the seeds in parallel
    # Adjust processes=10 if you want to run all at once, or fewer if CPU limited
    with multiprocessing.Pool(processes=10) as pool:
        results = list(tqdm(pool.imap(train_worker, seeds), total=len(seeds)))

    df_results = pd.DataFrame(results)
    
    print("\n--- RUN SUMMARY ---")
    print(df_results)
    print(f"\nAverage Score across all seeds: {df_results['score'].mean():.2f}")
    
    df_results.to_csv(f"{output_folder}/summary_statistics.csv", index=False)