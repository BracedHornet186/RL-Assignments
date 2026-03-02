import multiprocessing
import pandas as pd
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Specific folder for the modified reward runs
output_folder = "q_learning_modified_reward_episode_length"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

class AcrobatAgent:
    def __init__(self, env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, num_bins=10):
        self.env = env
        self.lr = learning_rate
        self.gamma = 0.99
        self.num_bins = num_bins

        # PRE-ALLOCATED ARRAY: Fixes the CPU frequency stall
        shape = tuple([num_bins + 1] * 6 + [env.action_space.n])
        self.q_values = np.zeros(shape, dtype=np.float32)

        self.low = self.env.observation_space.low
        self.high = self.env.observation_space.high
        self.bin_edges = [np.linspace(self.low[i], self.high[i], num_bins + 1)[1:-1] for i in range(6)]

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def bin_observation(self, obs):
        return tuple(int(np.digitize(obs[i], self.bin_edges[i])) for i in range(6))

    def get_action(self, obs):
        state = self.bin_observation(obs)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_values[state])
    
    def update_q(self, state, action, reward, terminated, next_state):
        future_q = (not terminated) * np.max(self.q_values[next_state])
        td_error = (reward + self.gamma * future_q) - self.q_values[state][action]
        self.q_values[state][action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

def get_modified_reward(obs, eta=0.5):
    """Calculates reward based on Equation 1 from assignment"""
    # obs: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]
    h = -obs[0] - (obs[0] * obs[2] - obs[1] * obs[3]) 
    
    term1 = (eta * h) / 2
    term2 = np.sign(-1 + (eta * h)) * ((2 - (eta * h)) / 2)
    return term1 + term2

def train_worker(args):
    """Worker function for multiprocessing"""
    seed, eta = args
    
    # FIXED HYPERPARAMETERS
    lr = 0.05
    start_eps = 1.0 
    n_episodes = 100000 
    exp_decay_rate = 0.9 
    
    np.random.seed(seed)
    env = gym.make("Acrobot-v1")
    agent = AcrobatAgent(env, lr, start_eps, exp_decay_rate, 0.1)

    episode_lengths = [] 
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = agent.bin_observation(obs)
        done = False
        steps = 0 
        
        while not done:
            action = agent.get_action(obs)
            next_obs, _, terminated, truncated, _ = env.step(action)
            
            # --- APPLY MODIFIED REWARD HERE WITH SPECIFIC ETA ---
            reward = get_modified_reward(next_obs, eta=eta)
            
            next_state = agent.bin_observation(next_obs)
            agent.update_q(state, action, reward, terminated, next_state)
            
            steps += 1 
            obs, state = next_obs, next_state
            done = terminated or truncated
        
        agent.decay_epsilon()
        episode_lengths.append(steps)
    
    filename = f"{output_folder}/modified_run_seed_{seed}_eta_{eta}.csv"
    
    # Save ONLY the episode lengths to the CSV
    df = pd.DataFrame({"episode": range(n_episodes), "length": episode_lengths})
    df.to_csv(filename, index=False)
    env.close()
    
    return eta, df

if __name__ == "__main__":
    used_seed = 42
    eta_values = [0.5, 1, 2]
    colors = {0.5: 'blue', 1: 'green', 2: 'red'}
    
    # Package arguments for the worker pool
    tasks = [(used_seed, eta) for eta in eta_values]
    results = {}
    
    print(f"Starting parallel training for seed {used_seed} across eta values {eta_values}...")
    
    # Use multiprocessing to run all eta values simultaneously
    # Limit processes to the number of tasks (3) to avoid overhead
    with multiprocessing.Pool(processes=min(len(tasks), multiprocessing.cpu_count())) as pool:
        for eta, df in tqdm(pool.imap_unordered(train_worker, tasks), total=len(tasks), desc="Training Progress"):
            results[eta] = df
    
    print("\nGenerating comparison episode length plot...")
    
    plt.figure(figsize=(12, 7))
    window_size = 4000
    
    # Sort eta values to ensure consistent legend ordering
    for eta in sorted(eta_values):
        df = results[eta]
        
        # Calculate a moving average to smooth out the noisy step-to-step variations
        smoothed_lengths = df['length'].rolling(window=window_size, min_periods=1).mean()

        # Plot raw data in the background (very faded)
        plt.plot(df['episode'], df['length'], alpha=0.1, color=colors[eta])
        
        # Plot smoothed moving average in the foreground
        plt.plot(df['episode'], smoothed_lengths, color=colors[eta], linewidth=2, label=f'eta={eta} (100-ep Moving Avg)')

    plt.title(f'Episode Length vs Number of Training Episodes (Seed {used_seed})', fontsize=14)
    plt.xlabel('Training Episodes', fontsize=12)
    plt.ylabel('Episode Length (Steps)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plot_path = f"{output_folder}/comparison_eta_length_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Comparison plot successfully saved to {plot_path}")