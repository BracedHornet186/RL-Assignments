import os
import numpy as np
import gymnasium as gym
import pandas as pd
from tqdm import tqdm

# Create the specific folder for the best runs
output_folder = os.path.join("Assignment 1", "Question 2", "q_learning_modified_reward_episode_length")
os.makedirs(output_folder, exist_ok=True)

class AcrobotAgent:
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
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_n)
        return np.argmax(self.q_values[state])
    
    def update_q(self, state, action, reward, terminated, next_state):
        future_q = 0.0 if terminated else np.max(self.q_values[next_state])
        
        state_action = state + (action,)
        current_q = self.q_values[state_action]
        
        self.q_values[state_action] = current_q + self.lr * (reward + self.gamma * future_q - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)


def train_single_seed(seed):
    # --- FIXED HYPERPARAMETERS ---
    lr = 0.1
    start_eps = 1.0
    n_episodes = 50000 
    exp_decay_rate = 0.9 
    final_eps = 0.1
    
    # Set seeds for NumPy and Gym spaces to ensure strict reproducibility
    np.random.seed(seed)
    env = gym.make("Acrobot-v1", render_mode="human")
    env.action_space.seed(seed)
    
    agent = AcrobotAgent(
        env=env,
        learning_rate=lr,
        initial_epsilon=start_eps,
        epsilon_decay=exp_decay_rate,
        final_epsilon=final_eps
    )

    # Track episode lengths instead of rewards
    episode_lengths = np.zeros(n_episodes, dtype=np.int32)
    
    # Use tqdm directly on the loop since we are only running one seed
    for episode in tqdm(range(n_episodes), desc=f"Training Seed {seed}"):
        obs, _ = env.reset(seed=seed + episode)
        state = agent.bin_observation(obs)
        done = False
        step_count = 0
        
        while not done:
            action = agent.get_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.bin_observation(next_obs)
            
            agent.update_q(state, action, reward, terminated, next_state)
            
            state = next_state
            step_count += 1
            done = terminated or truncated
        
        agent.decay_epsilon()
        episode_lengths[episode] = step_count
    
    env.close()
    
    # --- SAVE RAW DATA TO CSV ---
    filename = os.path.join(output_folder, f"original_reward_seed_{seed}.csv")
    pd.DataFrame({"episode": range(n_episodes), "length": episode_lengths}).to_csv(filename, index=False)
    
    avg_length = np.mean(episode_lengths[-100:]) 
    return {"seed": seed, "avg_length_last_100": avg_length, "filepath": filename}


if __name__ == "__main__":
    target_seed = 42
    
    print(f"Starting Q-Learning single run on seed {target_seed}...")
    print(f"Data will be saved in: '{output_folder}'")

    # Run the training directly without multiprocessing
    result = train_single_seed(target_seed)
    
    print("\n--- RUN SUMMARY ---")
    print(f"Seed: {result['seed']}")
    print(f"Average Episode Length (last 100 episodes): {result['avg_length_last_100']:.2f} steps")
    print(f"Data saved to: {result['filepath']}")
