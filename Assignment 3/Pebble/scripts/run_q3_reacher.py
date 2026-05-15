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
N_WORKERS    = 2   # run 2 seeds in parallel (per reward type)

OBS_DIM    = None   # set after env init
ACTION_DIM = None


# ── Reacher reward functions (Ra, Rb, Rc) ───────────────────────
# dm_control reacher observation OrderedDict insertion order is:
#     position(2)  →  to_target(2)  →  velocity(2)
# so after _flatten_obs (no sort, dict-order concat) the slices are:
#     obs[0:2] = position, obs[2:4] = to_target, obs[4:6] = velocity
# IMPORTANT: obs[-2:] is *velocity*, NOT to_target — the original code had
# this swapped, which silently broke the Ra and Rb teachers.
def _to_target(obs):
    return obs[2:4]

def _in_target(obs):
    """True when the finger-to-target distance is below dm_control's threshold."""
    if len(obs) < 4:
        return False
    return float(np.linalg.norm(_to_target(obs))) < 0.05


def reward_Ra(obs, action):
    """Ra: 1 if in target, else -||dist|| - ||action||²"""
    if _in_target(obs):
        return 1.0
    dist = float(np.linalg.norm(_to_target(obs)))
    return -dist - float(np.sum(action ** 2))

def reward_Rb(obs, action):
    """Rb: 1 if in target, else 0"""
    return 1.0 if _in_target(obs) else 0.0

def reward_Rc(obs, action):
    """Rc: constant -1/step. Episode terminates on (dist<0.05 AND |vel|<0.05).
    On timeout (1000 steps without termination) the wrapper adds an extra
    -20 penalty and internally resets the env so the episode CONTINUES
    (no truncation). Identical to Reacher/sac_reacher.py's Rc semantics.
    """
    return -1.0

def gt_segment_return(reward_fn, obs_seq, act_seq):
    return sum(reward_fn(o, a) for o, a in zip(obs_seq, act_seq))


# ── Environment factories (top-level for multiprocessing/spawn pickling) ──
REWARD_FNS = {"Ra": reward_Ra, "Rb": reward_Rb, "Rc": reward_Rc}


class _ReacherEnvFactory:
    """Picklable replacement for the old `make_reacher_env` closure."""
    def __init__(self, reward_name):
        self.reward_name = reward_name

    def __call__(self, seed):
        from dm_control import suite
        env = suite.load("reacher", "easy", task_kwargs={"random": seed})
        return ReacherWrapper(env, self.reward_name)


class _ReacherEvalEnvFactory:
    """Picklable replacement for `make_reacher_eval_env`."""
    def __init__(self, reward_name):
        self.reward_name = reward_name

    def __call__(self):
        from dm_control import suite
        env = suite.load("reacher", "easy")
        return ReacherWrapper(env, self.reward_name)


class _GTRewardFn:
    """Picklable ground-truth segment-return fn keyed by reward name."""
    def __init__(self, reward_name):
        self.reward_name = reward_name

    def __call__(self, obs_seq, act_seq):
        return gt_segment_return(REWARD_FNS[self.reward_name], obs_seq, act_seq)


def make_reacher_env(reward_fn_name):
    return _ReacherEnvFactory(reward_fn_name)

def make_reacher_eval_env(reward_fn_name):
    return _ReacherEvalEnvFactory(reward_fn_name)


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
        ts       = self._env.step(action)
        obs_dict = ts.observation
        obs      = self._flatten_obs(obs_dict)
        self._step_count += 1

        reward = self._reward_fn(obs, action)

        if self._reward_name == "Rc":
            # Mirror Reacher/sac_reacher.py Rc semantics exactly:
            #   terminate on (dist<0.05 AND |velocity|<0.05); never truncate.
            #   On timeout, add -20 penalty and internally reset, but the
            #   episode CONTINUES (the agent is forced to keep going).
            to_target = np.array(obs_dict["to_target"]).flatten()
            velocity  = np.array(obs_dict.get("velocity",
                                              np.zeros(self._action_dim))
                                 ).flatten()
            in_target  = float(np.linalg.norm(to_target)) < 0.05
            terminated = bool(in_target and
                              float(np.linalg.norm(velocity)) < 0.05)
            truncated  = False
            if not terminated and self._step_count >= self._max_steps:
                reward += -20.0
                ts_reset         = self._env.reset()
                obs              = self._flatten_obs(ts_reset.observation)
                self._step_count = 0
        else:
            terminated = False
            truncated  = self._step_count >= self._max_steps

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
    import argparse, json
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    cli = argparse.ArgumentParser()
    cli.add_argument("--rewards", type=str, default="Ra,Rb,Rc",
                     help="Comma-separated subset of {Ra,Rb,Rc} to train. "
                          "Already-aggregated rewards are still loaded into the final plot.")
    cli_args = cli.parse_args()
    target_rewards = [r.strip() for r in cli_args.rewards.split(",") if r.strip()]
    print(f"Using device: {DEVICE} | parallel workers: {N_WORKERS} | "
          f"training rewards: {target_rewards}")

    # Get dims from a temporary env
    from dm_control import suite
    _tmp_env = suite.load("reacher", "easy")
    _tmp_w   = ReacherWrapper(_tmp_env, "Rb")
    OBS_DIM    = _tmp_w._obs_dim
    ACTION_DIM = _tmp_w._action_dim
    print(f"Reacher obs_dim={OBS_DIM}, action_dim={ACTION_DIM}")

    COLORS = {"Ra": "#F44336", "Rb": "#2196F3", "Rc": "#4CAF50"}
    curves_by_name = {}

    for reward_name in target_rewards:
        print(f"\n=== PEBBLE with teacher based on {reward_name} ===")
        gt_fn = _GTRewardFn(reward_name)
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
        curves_by_name[reward_name] = {"timesteps": ts, "mean": mean, "std": std}

    # Always try to load every aggregated curve so the comparison plot is complete.
    for reward_name in ["Ra", "Rb", "Rc"]:
        if reward_name in curves_by_name:
            continue
        agg_path = os.path.join(LOG_DIR, f"pebble_{reward_name}_aggregated.json")
        if os.path.exists(agg_path):
            with open(agg_path) as f:
                d = json.load(f)
            curves_by_name[reward_name] = {
                "timesteps": d["timesteps"], "mean": d["mean"], "std": d["std"],
            }
            print(f"  loaded existing aggregated curve for {reward_name}")
        else:
            print(f"  (no aggregated curve for {reward_name}; skipping in plot)")

    curves = [
        {"label": f"PEBBLE (teacher={r})",
         "timesteps": curves_by_name[r]["timesteps"],
         "mean":      curves_by_name[r]["mean"],
         "std":       curves_by_name[r]["std"],
         "color":     COLORS[r]}
        for r in ["Ra", "Rb", "Rc"] if r in curves_by_name
    ]
    if curves:
        plot_curves(
            curves,
            title     = "PEBBLE on Reacher-Easy: Three Simulated Teachers",
            ylabel    = "Average Undiscounted Return (GT)",
            save_path = f"{LOG_DIR}/plots/pebble_reacher_comparison.png",
        )
    print("Done.")
