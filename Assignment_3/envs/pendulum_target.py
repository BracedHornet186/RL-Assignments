import gymnasium as gym
import numpy as np

class TargetPendulum(gym.Wrapper):
    def __init__(self, theta_target_deg):
        env = gym.make("Pendulum-v1")
        super().__init__(env)
        self.theta_target = np.deg2rad(theta_target_deg)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        cos_theta, sin_theta, theta_dot = obs
        theta = np.arctan2(sin_theta, cos_theta)

        angle_diff = np.arctan2(
            np.sin(theta - self.theta_target),
            np.cos(theta - self.theta_target)
        )

        reward = - angle_diff**2 - 0.1 * theta_dot**2 - 0.001 * action[0]**2

        return obs, reward, terminated, truncated, info