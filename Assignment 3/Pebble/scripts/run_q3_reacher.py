"""
Q3 (Bonus) Part 3 — PEBBLE on Reacher (easy) with three simulated teachers.
Each teacher is based on one of the three reward formulations Ra, Rb, Rc.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

SEEDS        = list(range(15))
TOTAL_STEPS  = 300_000
EVAL_EVERY   = 10_000
EVAL_EPS     = 20
RANDOM_STEPS = 10_000
QUERY_EVERY  = 5_000
N_QUERIES    = 10
SEGMENT_LEN  = 50
BUDGET       = 1000
LOG_DIR      = "logs/q3_pebble_reacher"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
N_WORKERS    = 5   # run 5 seeds in parallel

OBS_DIM    = None   # set after env init
ACTION_DIM = None


# ── Reacher reward functions (Ra, Rb, Rc) ───────────────────────
def _in_target(obs):
    """Check if reacher tip is in target. obs structure depends on dm_control."""
    # dm_control reacher obs: [cos q, sin q, target_x, target_y, qvel, finger-target dist]
    # "in target" ≈ dist to target < threshold
    # The last two obs are the vector from finger to target
    dist = np.linalg.norm(obs[-2:]) if len(obs) >= 2 else 1.0
    return dist < 0.05   # threshold (matches dm_control "in_target" check)


def reward_Ra(obs, action):
    """Ra: 1 if in target, else -||dist|| - ||action||²"""
    if _in_target(obs):
        return 1.0
    dist = np.linalg.norm(obs[-2:])
    return -dist - float(np.sum(action ** 2))

def reward_Rb(obs, action):
    """Rb: 1 if in target, else 0"""
    return 1.0 if _in_target(obs) else 0.0

def reward_Rc(obs, action):
    """Rc: -1 until episode terminates at target with near-zero velocity"""
    return -1.0   # episode terminates early on success (handled in env)

def gt_segment_return(reward_fn, obs_seq, act_seq):
    return sum(reward_fn(o, a) for o, a in zip(obs_seq, act_seq))


# ── Environment factories ────────────────────────────────────────
def make_reacher_env(reward_fn_name):
    def _fn(seed):
        from dm_control import suite
        import dm2gym
        env = suite.load("reacher", "easy", task_kwargs={"random": seed})
        # Wrap to gymnasium interface
        wrapped = ReacherWrapper(env, reward_fn_name)
        return wrapped
    return _fn

def make_reacher_eval_env(reward_fn_name):
    def _fn():
        from dm_control import suite
        env = suite.load("reacher", "easy")
        return ReacherWrapper(env, reward_fn_name)
    return _fn


REWARD_FNS = {"Ra": reward_Ra, "Rb": reward_Rb, "Rc": reward_Rc}


class ReacherWrapper:
    """
    Minimal gymnasium-compatible wrapper for dm_control reacher.
    Replaces the dm_control reward with Ra / Rb / Rc.
    """
    def __init__(self, env, reward_fn_name):
        self._env          = env
        self._reward_fn    = REWARD_FNS[reward_fn_name]
        self._reward_name  = reward_fn_name
        self._max_steps    = 1000
        self._step_count   = 0

        # Infer obs/action dims from env
        ts = env.reset()
        flat_obs = self._flatten_obs(ts.observation)
        self._obs_dim    = len(flat_obs)
        self._action_dim = env.action_spec().shape[0]

        # Fake gymnasium spaces for compatibility with SAC
        import numpy as np
        class _Box:
            def __init__(self, shape):
                self.shape = shape
            def sample(self):
                return np.random.uniform(-1, 1, self.shape)
        self.observation_space = _Box((self._obs_dim,))
        self.action_space      = _Box((self._action_dim,))

    def _flatten_obs(self, obs_dict):
        return np.concatenate([v.flatten() for v in obs_dict.values()])

    def reset(self, seed=None, **kwargs):
        ts = self._env.reset()
        self._step_count = 0
        obs = self._flatten_obs(ts.observation)
        return obs, {}

    def step(self, action):
        ts = self._env.step(action)
        obs = self._flatten_obs(ts.observation)
        self._step_count += 1

        reward = self._reward_fn(obs, action)

        # Rc: terminate on reaching target with low velocity
        if self._reward_name == "Rc":
            terminated = _in_target(obs) and np.linalg.norm(obs[4:6]) < 0.01
        else:
            terminated = False

        truncated = self._step_count >= self._max_steps
        return obs, reward, terminated, truncated, {}

    def close(self):
        pass

    def gt_segment_return(self, reward_fn_name, obs_seq, act_seq):
        fn = REWARD_FNS[reward_fn_name]
        return sum(fn(o, a) for o, a in zip(obs_seq, act_seq))


if __name__ == "__main__":
    from agents.pebble_trainer import run_pebble_seeds_parallel
    from utils.plotting import plot_curves
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    print(f"Using device: {DEVICE} | parallel workers: {N_WORKERS}")

    # Get dims from a temporary env
    from dm_control import suite
    _tmp_env = suite.load("reacher", "easy")
    _tmp_w   = ReacherWrapper(_tmp_env, "Rb")
    OBS_DIM    = _tmp_w._obs_dim
    ACTION_DIM = _tmp_w._action_dim
    print(f"Reacher obs_dim={OBS_DIM}, action_dim={ACTION_DIM}")

    COLORS = {"Ra": "#F44336", "Rb": "#2196F3", "Rc": "#4CAF50"}
    curves = []

    for reward_name in ["Ra", "Rb", "Rc"]:
        print(f"\n=== PEBBLE with teacher based on {reward_name} ===")

        # GT reward fn for this teacher
        teacher_reward_fn = REWARD_FNS[reward_name]
        def gt_fn(obs_seq, act_seq, _rfn=teacher_reward_fn):
            return gt_segment_return(_rfn, obs_seq, act_seq)

        ts, mean, std = run_pebble_seeds_parallel(
            env_fn       = make_reacher_env(reward_name),
            eval_env_fn  = make_reacher_eval_env(reward_name),
            gt_reward_fn = gt_fn,
            obs_dim      = OBS_DIM,
            action_dim   = ACTION_DIM,
            seeds        = SEEDS,
            total_steps  = TOTAL_STEPS,
            query_budget = BUDGET,
            query_every  = QUERY_EVERY,
            n_queries    = N_QUERIES,
            segment_len  = SEGMENT_LEN,
            eval_every   = EVAL_EVERY,
            eval_episodes= EVAL_EPS,
            random_steps = RANDOM_STEPS,
            log_dir      = LOG_DIR,
            run_prefix   = f"pebble_{reward_name}",
            device       = DEVICE,
            n_workers    = N_WORKERS,
        )
        curves.append({
            "label":     f"PEBBLE (teacher={reward_name})",
            "timesteps": ts, "mean": mean, "std": std,
            "color":     COLORS[reward_name],
        })

    plot_curves(
        curves,
        title     = "PEBBLE on Reacher-Easy: Three Simulated Teachers",
        ylabel    = "Average Undiscounted Return (GT)",
        save_path = f"{LOG_DIR}/plots/pebble_reacher_comparison.png",
    )
    print("Done.")
