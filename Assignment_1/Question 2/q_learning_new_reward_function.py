import multiprocessing
import pandas as pd
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os

# Specific folder for the modified reward runs
output_folder = "q_learning_modified_reward"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

class AcrobatAgent:
    def __init__(self, env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, num_bins=10):
        self.env = env
        self.lr = learning_rate
        self.gamma = 0.99
        self.num_bins = num_bins

        # PRE-ALLOCATED ARRAY: Fixes the CPU frequency stall (the ~947 MHz issue)
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
    # h = height of the tip relative to the fixed base
    h = -obs[0] - (obs[0] * obs[2] - obs[1] * obs[3]) # -cos(theta1) - cos(theta1+theta2)
    
    term1 = (eta * h) / 2
    term2 = np.sign(-1 + (eta * h)) * ((2 - (eta * h)) / 2)
    return term1 + term2

def train_worker(seed):
    # FIXED HYPERPARAMETERS
    lr = 0.05
    start_eps = 1.0 # High exploration is useful for dense rewards
    n_episodes = 50000 
    exp_decay_rate = 0.999 # Slower decay as recommended for shaped rewards
    
    np.random.seed(seed)
    env = gym.make("Acrobot-v1")
    agent = AcrobatAgent(env, lr, start_eps, exp_decay_rate, 0.1)

    episode_returns = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = agent.bin_observation(obs)
        done = False
        total_reward = 0
        
        while not done:
            action = agent.get_action(obs)
            next_obs, original_reward, terminated, truncated, _ = env.step(action)
            
            # --- APPLY MODIFIED REWARD HERE ---
            reward = get_modified_reward(next_obs, eta=0.5)
            
            next_state = agent.bin_observation(next_obs)
            agent.update_q(state, action, reward, terminated, next_state)
            
            total_reward += reward
            obs, state = next_obs, next_state
            done = terminated or truncated
        
        agent.decay_epsilon()
        episode_returns.append(total_reward)
    
    filename = f"{output_folder}/modified_run_seed_{seed}.csv"
    pd.DataFrame({"episode": range(n_episodes), "reward": episode_returns}).to_csv(filename, index=False)
    env.close()
    return {"seed": seed, "score": np.mean(episode_returns[-100:])}

if __name__ == "__main__":
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    with multiprocessing.Pool(processes=10) as pool:
        results = list(tqdm(pool.imap(train_worker, seeds), total=len(seeds)))
    
    df_results = pd.DataFrame(results)
    print(f"\nAverage Modified Score: {df_results['score'].mean():.2f}")
    df_results.to_csv(f"{output_folder}/modified_summary.csv", index=False)