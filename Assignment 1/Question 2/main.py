from collections import defaultdict
import gymnasium as gym
import numpy as np

class AcrobatAgent:
    def __init__(
            self, 
            env: gym.Env,
            learning_rate: float, 
            initial_epsilon: float, 
            epsilon_decay: float,
            final_epsilon: float, 
            gamma: float =  0.95,
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
    
    def update(
            self,
            obs: tuple[float, float, float, float, float, float],
            action: int,
            reward: float,
            terminated: bool, 
            next_obs: tuple[float, float, float, float, float, float],
    ):
        obs = self.bin_observation(obs)
        next_obs = self.bin_observation(next_obs)
        
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.gamma * future_q_value

        temporal_difference = target - self.q_values[obs][action]

        self.q_values[obs][action] = self.q_values[obs][action] + self.lr * temporal_difference

        self.training_error.append(temporal_difference)

    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


learning_rate = 0.01
n_episodes = 100000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

env = gym.make("Acrobot-v1")

agent = AcrobatAgent(
    env = env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
) 


from tqdm import tqdm

for episode in tqdm(range(n_episodes)):
    
    obs, info = env.reset()
    done = False

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, terminated, next_obs)

        done = terminated or truncated
        obs = next_obs
    
    agent.decay_epsilon()
