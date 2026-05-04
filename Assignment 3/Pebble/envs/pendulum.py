"""
Modified Pendulum-v1 with target angle θ_target.
Reward: agent must align pendulum to θ_target and stay there.
"""

import numpy as np
import gymnasium as gym


class PendulumTargetEnv(gym.Wrapper):
    """
    Wraps Pendulum-v1. Replaces the default reward with a custom one
    that rewards alignment to θ_target (in degrees).

    Reward design:
      r = -|θ - θ_target|² / max_angle²    (angle error penalty)
          - 0.1 * |ω|²                      (angular velocity penalty — stay still)
          - 0.001 * |action|²               (action penalty — be efficient)

    All terms are normalised so r ∈ [-1, 0].
    θ_target = 0 → upright (same as default but with custom fn).
    """
    def __init__(self, theta_target_deg: float = 0.0, reward_scale: float = 1.0):
        super().__init__(gym.make("Pendulum-v1"))
        self.theta_target = np.deg2rad(theta_target_deg)
        self.reward_scale = reward_scale

    def _custom_reward(self, obs, action):
        """
        obs = [cos(θ), sin(θ), dθ/dt]
        Recover θ from cos/sin.
        """
        cos_th, sin_th, thdot = obs
        th = np.arctan2(sin_th, cos_th)   # θ in [-π, π]

        # Angle difference, wrapped to [-π, π]
        angle_diff = th - self.theta_target
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        r = -(angle_diff ** 2) - 0.1 * (thdot ** 2) - 0.001 * float(action[0] ** 2)
        return float(r) * self.reward_scale

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self._custom_reward(obs, action)
        return obs, reward, terminated, truncated, info

    def gt_segment_return(self, obs_seq, act_seq):
        """Ground-truth return for a segment — used by SimulatedTeacher."""
        return sum(self._custom_reward(o, a) for o, a in zip(obs_seq, act_seq))


def make_pendulum(theta_target_deg: float, reward_scale: float = 1.0, seed: int = 0):
    env = PendulumTargetEnv(theta_target_deg, reward_scale)
    env.reset(seed=seed)
    return env
