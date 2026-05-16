import gymnasium as gym
import numpy as np

class TargetPendulum(gym.Wrapper):
    def __init__(self, theta_target_deg):
        env = gym.make("Pendulum-v1", max_episode_steps=1000)
        super().__init__(env)
        self.theta_target = np.deg2rad(theta_target_deg)
        
        # Override the action space to match the Tanh output [-1.0, 1.0]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def step(self, action):
        # Scale the [-1, 1] action from the agent to the physics engine's [-2, 2]
        scaled_action = action * 2.0
        
        obs, _, terminated, truncated, info = self.env.step(scaled_action)

        cos_theta, sin_theta, theta_dot = obs
        theta = np.arctan2(sin_theta, cos_theta)

        angle_diff = np.arctan2(
            np.sin(theta - self.theta_target),
            np.cos(theta - self.theta_target)
        )

        # Penalize based on the actual physical torque (scaled_action) applied
        reward = - angle_diff**2 - 0.1 * theta_dot**2 - 0.001 * scaled_action[0]**2

        return obs, reward, terminated, truncated, info