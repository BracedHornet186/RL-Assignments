"""
LunarLander environment wrappers.
  - ContinuousLunarLander : standard continuous action space
  - HoverLunarLander      : adds +200 hover reward (once per episode)
  - ChangingRewardLunarLander : starts with hover +200, switches to -100 mid-training
"""

import gymnasium as gym
import numpy as np


class ContinuousLunarLander(gym.Wrapper):
    """Plain LunarLander-v3 continuous. No modifications."""
    def __init__(self):
        super().__init__(gym.make("LunarLander-v3", continuous=True))


class DiscreteLunarLander(gym.Wrapper):
    """Plain LunarLander-v3 discrete."""
    def __init__(self):
        super().__init__(gym.make("LunarLander-v3", continuous=False))


def _in_hover_box(state):
    """Returns True if lander is inside the hover box: |x|<0.1, 0.4<|y|<0.6"""
    x, y = state[0], state[1]
    return abs(x) < 0.1 and 0.4 < abs(y) < 0.6


class HoverLunarLander(gym.Wrapper):
    """
    LunarLander-v3 (continuous) with an extra hover reward.
    hover_bonus is given AT MOST ONCE per episode when the lander
    enters the hover box (|x|<0.1, 0.4<|y|<0.6).
    """
    def __init__(self, hover_bonus=200.0, continuous=True):
        super().__init__(gym.make("LunarLander-v3", continuous=continuous))
        self.hover_bonus = hover_bonus
        self._given_bonus = False

    def reset(self, **kwargs):
        self._given_bonus = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self._given_bonus and _in_hover_box(obs):
            reward += self.hover_bonus
            self._given_bonus = True
        return obs, reward, terminated, truncated, info


class ChangingRewardLunarLander:
    """
    Wrapper that serves as a training-time reward switcher.
    Phase 1: hover_bonus = +200
    Phase 2 (after switch_step): hover_bonus = -100

    Usage:
        env = ChangingRewardLunarLander(switch_step=200_000)
        env.set_global_step(current_step)  # call before every episode
    """
    def __init__(self, switch_step: int, continuous: bool = True):
        self.switch_step  = switch_step
        self.continuous   = continuous
        self._global_step = 0
        self._env = None
        self._rebuild()

    def _rebuild(self):
        bonus = -100.0 if self._global_step >= self.switch_step else 200.0
        if self._env is not None:
            self._env.close()
        self._env = HoverLunarLander(hover_bonus=bonus, continuous=self.continuous)

    def set_global_step(self, step: int):
        old_phase = self._global_step >= self.switch_step
        self._global_step = step
        new_phase = self._global_step >= self.switch_step
        if old_phase != new_phase:
            self._rebuild()
            print(f"\n[Env] Reward switched at step {step}: hover bonus → -100\n")

    # Delegate the gym API
    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        self._env.close()

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space
