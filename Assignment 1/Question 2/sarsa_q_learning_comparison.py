from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import multiprocessing

class AcrobatAgent:
    def __init__(
            self, 
            env: gym.Env,
            learning_rate: float, 
            initial_epsilon: float, 
            epsilon_decay: float,
            final_epsilon: float, 
            gamma: float =  0.99,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.gamma = gamma

        # Pre-compute bins for speed
        self.low = self.env.observation_space.low
        self.high = self.env.observation_space.high
        self.num_bins = 10
        self.bin_edges = [np.linspace(self.low[i], self.high[i], self.num_bins + 1)[1:-1] for i in range(6)]

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs):
        # Important: get_action should also use the binned state
        state = self.bin_observation(obs)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[state]))
        
    def bin_observation(self, obs):
        # Faster binning using pre-computed edges
        return tuple(int(np.digitize(obs[i], self.bin_edges[i])) for i in range(6))
    
    # def update(
    #         self,
    #         obs: tuple[float, float, float, float, float, float],
    #         action: int,
    #         reward: float,
    #         terminated: bool, 
    #         next_obs: tuple[float, float, float, float, float, float],
    # ):
    #     obs = self.bin_observation(obs)
    #     next_obs = self.bin_observation(next_obs)
        
    #     future_q_value = (not terminated) * np.max(self.q_values[next_obs])
    #     target = reward + self.gamma * future_q_value

    #     temporal_difference = target - self.q_values[obs][action]

    #     self.q_values[obs][action] = self.q_values[obs][action] + self.lr * temporal_difference

    #     self.training_error.append(temporal_difference)

    # Q-Learning Update
    def update_q(self, state, action, reward, terminated, next_state):

        future_q = (not terminated) * np.max(self.q_values[next_state])
        td_error = (reward + self.gamma * future_q) - self.q_values[state][action]
        self.q_values[state][action] += self.lr * td_error

    # SARSA Update
    def update_sarsa(self, state, action, reward, terminated, next_state, next_action):

        future_q = (not terminated) * self.q_values[next_state][next_action]
        td_error = (reward + self.gamma * future_q) - self.q_values[state][action]
        self.q_values[state][action] += self.lr * td_error

    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)


def train_and_get_returns(algo="q_learning"):
    env = gym.make('Acrobot-v1')
    learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
    n_episodes = 50000        # Number of hands to practice
    start_epsilon = 1.0         # Start with 100% random actions
    final_epsilon = 0.1         # Always keep some exploration
    epsilon_decay = np.exp(np.log(final_epsilon / start_epsilon) / n_episodes)

    agent = AcrobatAgent(env, learning_rate, start_epsilon, epsilon_decay, final_epsilon)
    episode_returns = []

    for episode in tqdm(range(n_episodes), desc=f"Training {algo}"):
        obs, info = env.reset()
        state = agent.bin_observation(obs)
        action = agent.get_action(obs)
        
        total_reward = 0
        done = False
        
        while not done:
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = agent.bin_observation(next_obs)
            next_action = agent.get_action(next_obs)
            
            if algo == "q_learning":
                agent.update_q(state, action, reward, terminated, next_state)
            else:
                agent.update_sarsa(state, action, reward, terminated, next_state, next_action)
            
            total_reward += reward
            state, action = next_state, next_action
            done = terminated or truncated
            
        episode_returns.append(total_reward)
        agent.decay_epsilon()
        
    return episode_returns

if __name__ == "__main__":
    with multiprocessing.Pool(processes=2) as pool:
            # map() runs the function with different arguments in parallel
            # Note: we pass the arguments as a list
            results = pool.map(train_and_get_returns, ["q_learning", "sarsa"])

    # 2. Unpack the results
    q_returns, sarsa_returns = results
    import matplotlib.pyplot as plt

    def moving_average(data, window=500):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    import pandas as pd
    df = pd.DataFrame({
        "episode": np.arange(len(q_returns)),
        "q_learning": q_returns,
        "sarsa": sarsa_returns
    })
    df.to_csv("acrobot_results_100k_exponential_decay.csv", index=False)

    plt.figure(figsize=(12, 6))

    # Plot smoothed lines
    plt.plot(moving_average(q_returns), label="Q-Learning", linewidth=2)
    plt.plot(moving_average(sarsa_returns), label="SARSA", linewidth=2)


    plt.title("Acrobot-v1: Q-Learning vs SARSA Returns")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (Return)")
    # plt.axhline(y=-100, color='r', linestyle='--', label='Target Performance') # Example goal threshold
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("acrobot_comparison_100k_exponential_decay.png", dpi=300, bbox_inches='tight')
    plt.show()