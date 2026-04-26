import gymnasium as gym
import numpy as np

try:
    from dm_control import suite
except ImportError as e:
    raise ImportError(
        "dm_control is required for the Reacher environment. "
        "Install it with `pip install dm-control`."
    ) from e


def _flatten_observation(obs_dict):
    """
    Flattens a dm_control observation dict into a single 1D float32 array.
    """
    parts = []
    for v in obs_dict.values():
        v = np.asarray(v, dtype=np.float32)
        parts.append(v.ravel())
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts, axis=0)


class ReacherEnv(gym.Env):
    """
    Gymnasium-style wrapper for DeepMind Control Reacher (easy) with
    three reward formulations Ra, Rb, Rc as in the assignment sheet.

    - Ra: sparse 0/1 reward, fixed-length episodes (no early termination).
    - Rb: same 0/1 reward, but episode terminates when the target region is reached
          OR when max_episode_steps is hit.
    - Rc: time-to-goal style; reward = -1 each step until the target is reached
          (or time limit). Episodes terminate when the target region is reached
          OR when max_episode_steps is hit.

    We detect “in target” by thresholding the original dm_control reward, which
    is near 1 when the fingertip is inside the target region.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: str = "easy",
        reward_type: str = "Rb",
        max_episode_steps: int = 1000,
        hit_threshold: float = 0.9,
    ):
        """
        Args:
            task: 'easy' (as required by the assignment) or 'hard'.
            reward_type: one of {'Ra', 'Rb', 'Rc'}.
            max_episode_steps: horizon T used for truncation.
            hit_threshold: dm_control base reward above which we
                           consider the target reached.
        """
        assert reward_type in {"Ra", "Rb", "Rc"}, "reward_type must be Ra, Rb or Rc"

        self._env = suite.load(domain_name="reacher", task_name=task)
        self.reward_type = reward_type
        self.max_episode_steps = max_episode_steps
        self.hit_threshold = hit_threshold

        self._step_count = 0

        # ---- Build Gym spaces from dm_control specs ----
        action_spec = self._env.action_spec()
        obs_spec = self._env.observation_spec()

        # dm_control actions are already in [-1, 1] for Reacher
        self.action_space = gym.spaces.Box(
            low=np.asarray(action_spec.minimum, dtype=np.float32),
            high=np.asarray(action_spec.maximum, dtype=np.float32),
            shape=action_spec.shape,
            dtype=np.float32,
        )

        flat_dim = 0
        for v in obs_spec.values():
            v_shape = np.asarray(v.shape, dtype=np.int64)
            flat_dim += int(np.prod(v_shape))

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat_dim,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # dm_control uses numpy RNG seeding via the underlying physics;
            # here we just seed NumPy for reproducibility.
            np.random.seed(seed)

        self._step_count = 0
        time_step = self._env.reset()
        obs = _flatten_observation(time_step.observation)

        # Gymnasium API: return (obs, info)
        return obs, {}

    def step(self, action):
        self._step_count += 1

        # Clip action to dm_control bounds
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        time_step = self._env.step(action)
        base_reward = float(time_step.reward or 0.0)
        obs = _flatten_observation(time_step.observation)

        in_target = base_reward >= self.hit_threshold

        # ----- Reward shaping according to Ra / Rb / Rc -----
        if self.reward_type == "Ra":
            # Sparse: 1 if in target, 0 otherwise; no early termination.
            reward = 1.0 if in_target else 0.0
            terminated = False
        elif self.reward_type == "Rb":
            # Same sparse reward, but terminate when entering the target region.
            reward = 1.0 if in_target else 0.0
            terminated = in_target
        else:  # Rc
            # Time-to-goal style: -1 per step until termination.
            reward = -1.0
            terminated = in_target

        truncated = self._step_count >= self.max_episode_steps
        if truncated:
            # For Ra, Rb, Rc the truncation is solely due to time limit.
            # We do not override 'terminated' here.
            pass

        info = {
            "base_reward": base_reward,
            "in_target": in_target,
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info