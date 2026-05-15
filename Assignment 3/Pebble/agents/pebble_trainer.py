"""
PEBBLE training loop — fixed version.

Key fixes over v1:
  1. Unsupervised pre-training phase: during random_steps, collect preferences
     and pre-train the reward model BEFORE SAC starts using it.
  2. Reward normalization: normalize r_ψ outputs using running mean/std so
     SAC Q-values stay stable regardless of reward model scale.
  3. Conditional relabelling: only relabel after the reward model has been
     trained on at least `min_pref_before_relabel` preferences.
  4. More reward gradient steps early on (200 on first update, 50 thereafter).
  5. Reward model batch_size decoupled from SAC batch_size — use all available
     preferences in early rounds.
"""

import numpy as np
import torch
import os
import json
from pathlib import Path
from tqdm import tqdm

from agents.pebble import RewardModel, PreferenceBuffer, SimulatedTeacher
from agents.sac import SAC, ReplayBuffer


# ─────────────────────────────────────────────
#  Running normalizer for reward model outputs
# ─────────────────────────────────────────────
class RunningNormalizer:
    """Tracks running mean and std to normalize reward model outputs."""
    def __init__(self, eps=1e-8):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = 0
        self.eps   = eps

    def update(self, x):
        """x: numpy array of reward values."""
        batch_mean = np.mean(x)
        batch_var  = np.var(x)
        n = len(x)
        # Welford online update
        new_count = self.count + n
        delta     = batch_mean - self.mean
        self.mean = self.mean + delta * n / new_count
        self.var  = (self.var * self.count + batch_var * n +
                     delta ** 2 * self.count * n / new_count) / new_count
        self.count = new_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + self.eps)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def collect_segment(replay_buffer, segment_len):
    """Sample a random contiguous segment from the replay buffer."""
    if replay_buffer.size < segment_len:
        return None, None
    start = np.random.randint(0, replay_buffer.size - segment_len)
    return (replay_buffer.obs[start: start + segment_len].copy(),
            replay_buffer.action[start: start + segment_len].copy())


def relabel_replay_buffer(replay_buffer, reward_model, normalizer, batch_size=512):
    """
    Re-score all transitions in the replay buffer with the current r_ψ,
    then normalize using the running statistics.
    """
    device = reward_model.device
    n      = replay_buffer.size
    raw_rewards = np.zeros(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        obs_b = torch.FloatTensor(replay_buffer.obs[start:end]).to(device)
        act_b = torch.FloatTensor(replay_buffer.action[start:end]).to(device)
        with torch.no_grad():
            raw_rewards[start:end] = reward_model(obs_b, act_b).cpu().numpy().flatten()

    normalizer.update(raw_rewards)
    normalized = normalizer.normalize(raw_rewards)
    replay_buffer.reward[:n, 0] = normalized


def train_reward_model(reward_model, pref_buffer, n_steps, batch_size, device):
    """Run n_steps gradient updates on the reward model."""
    actual_batch = min(batch_size, len(pref_buffer))
    for _ in range(n_steps):
        s1o, s1a, s2o, s2a, lbl = pref_buffer.sample(actual_batch, device)
        reward_model.update(s1o, s1a, s2o, s2a, lbl)


# ─────────────────────────────────────────────
#  Main PEBBLE training loop
# ─────────────────────────────────────────────
def train_pebble(
    env_fn,
    eval_env_fn,
    gt_reward_fn,
    obs_dim,
    action_dim,
    total_steps            = 100_000,
    seed                   = 0,
    # SAC
    lr                     = 3e-4,
    gamma                  = 0.99,
    tau                    = 0.005,
    batch_size             = 256,
    buffer_size            = int(1e6),
    hidden                 = (256, 256),
    random_steps           = 10_000,
    # PEBBLE
    segment_len            = 50,
    query_every            = 2_000,
    n_queries              = 5,
    reward_train_steps     = 50,
    reward_train_steps_init= 200,   # more steps on very first reward update
    reward_lr              = 3e-4,
    reward_batch_size      = 64,    # separate from SAC batch_size
    max_pref_buffer        = 3_000,
    query_budget           = None,
    min_pref_before_relabel= 10,    # don't relabel until this many prefs collected
    # Eval
    eval_every             = 10_000,
    eval_episodes          = 20,
    # Logging
    log_dir                = "logs/pebble",
    run_name               = "pebble_run",
    show_pbar              = True,
    device                 = "cpu",
):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env      = env_fn(seed)
    eval_env = eval_env_fn()
    teacher  = SimulatedTeacher(gt_reward_fn)

    reward_model = RewardModel(obs_dim, action_dim, hidden=hidden,
                               lr=reward_lr, device=device)
    pref_buffer  = PreferenceBuffer(max_size=max_pref_buffer,
                                    segment_len=segment_len)
    normalizer   = RunningNormalizer()

    agent = SAC(obs_dim, action_dim, lr=lr, gamma=gamma, tau=tau,
                batch_size=batch_size, buffer_size=buffer_size,
                hidden=hidden, auto_alpha=True, device=device)

    obs, _     = env.reset(seed=seed)
    timesteps_list, gt_returns_list, all_rets = [], [], []
    total_queries  = 0
    first_update   = True

    pbar = tqdm(
        total=total_steps, desc=f"PEBBLE {run_name}",
        unit="step", unit_scale=True, dynamic_ncols=True,
        colour="magenta", leave=True, disable=not show_pbar,
    )

    for step in range(1, total_steps + 1):

        # ── Collect transition ───────────────────────────────────
        if step <= random_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, evaluate=False)

        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store 0 reward initially — will be relabelled once reward model is ready
        agent.store(obs, action, 0.0, next_obs, float(terminated))
        obs = next_obs
        if done:
            obs, _ = env.reset()

        # ── SAC update ───────────────────────────────────────────
        # Only update after random phase AND after reward model has been trained
        if step > random_steps and total_queries >= min_pref_before_relabel:
            agent.update()

        # ── Query + reward model update ──────────────────────────
        budget_ok = (query_budget is None or total_queries < query_budget)

        if step % query_every == 0 and agent.replay.size >= segment_len and budget_ok:
            # Collect preference queries
            for _ in range(n_queries):
                s1_obs, s1_act = collect_segment(agent.replay, segment_len)
                s2_obs, s2_act = collect_segment(agent.replay, segment_len)
                if s1_obs is None:
                    continue
                label = teacher.label(s1_obs, s1_act, s2_obs, s2_act)
                pref_buffer.add(s1_obs, s1_act, s2_obs, s2_act, label)
                total_queries += 1
                if query_budget and total_queries >= query_budget:
                    break

            # Train reward model
            if len(pref_buffer) >= reward_batch_size:
                n_steps = reward_train_steps_init if first_update else reward_train_steps
                train_reward_model(reward_model, pref_buffer,
                                   n_steps, reward_batch_size, device)
                first_update = False

                # Relabel replay buffer with normalized rewards
                relabel_replay_buffer(agent.replay, reward_model,
                                      normalizer, batch_size=512)

        # ── GT evaluation ────────────────────────────────────────
        if step % eval_every == 0:
            # Hard cap on eval-episode length: some env wrappers (e.g. the
            # Reacher Rc wrapper) never set `truncated=True`, so without this
            # an unconverged policy makes each eval episode run forever.
            eval_ep_cap = getattr(eval_env, "_max_steps", 1000)
            gt_rets = []
            for ep_i in range(eval_episodes):
                e_obs, _ = eval_env.reset(seed=10000 + ep_i)
                e_done, e_ret, e_steps = False, 0.0, 0
                while not e_done and e_steps < eval_ep_cap:
                    e_act = agent.select_action(e_obs, evaluate=True)
                    e_obs, e_r, e_term, e_trunc, _ = eval_env.step(e_act)
                    e_ret  += e_r
                    e_steps += 1
                    e_done  = e_term or e_trunc
                gt_rets.append(e_ret)

            mean_r = np.mean(gt_rets)
            timesteps_list.append(step)
            gt_returns_list.append(mean_r)
            all_rets.append(gt_rets)
            pbar.set_postfix({
                "GT_ret":  f"{mean_r:+.1f}",
                "queries": total_queries,
            })

        pbar.update(1)

    pbar.close()
    env.close()
    eval_env.close()

    log_path = os.path.join(log_dir, f"{run_name}.json")
    with open(log_path, "w") as f:
        json.dump({
            "timesteps":     timesteps_list,
            "gt_returns":    gt_returns_list,
            "all_returns":   all_rets,
            "total_queries": total_queries,
        }, f)

    agent.save(os.path.join(log_dir, f"{run_name}_sac.pt"))
    reward_model.save(os.path.join(log_dir, f"{run_name}_reward.pt"))

    return timesteps_list, gt_returns_list, all_rets


# ─────────────────────────────────────────────
#  Multi-seed runner (always sequential on GPU)
# ─────────────────────────────────────────────
def run_pebble_seeds(
    env_fn,
    eval_env_fn,
    gt_reward_fn,
    obs_dim,
    action_dim,
    seeds,
    total_steps  = 100_000,
    query_budget = 250,
    log_dir      = "logs/pebble",
    run_prefix   = "pebble",
    device       = "cpu",
    **kwargs,
):
    all_seed_means = []
    ts_ref = None

    seeds_bar = tqdm(total=len(seeds), desc="seeds done",
                     unit="seed", colour="green", leave=True)

    for seed in seeds:
        ts, means, _ = train_pebble(
            env_fn       = env_fn,
            eval_env_fn  = eval_env_fn,
            gt_reward_fn = gt_reward_fn,
            obs_dim      = obs_dim,
            action_dim   = action_dim,
            total_steps  = total_steps,
            seed         = seed,
            query_budget = query_budget,
            log_dir      = log_dir,
            run_name     = f"{run_prefix}_seed{seed}",
            device       = device,
            **kwargs,
        )
        all_seed_means.append(means)
        if ts_ref is None:
            ts_ref = ts
        seeds_bar.update(1)
        seeds_bar.set_postfix({"last_seed": seed})

    seeds_bar.close()

    arr  = np.array(all_seed_means)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)

    agg_path = os.path.join(log_dir, f"{run_prefix}_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump({
            "timesteps": ts_ref,
            "mean":      mean.tolist(),
            "std":       std.tolist(),
            "all_seeds": arr.tolist(),
        }, f)
    tqdm.write(f"✓ Saved → {agg_path}")

    return ts_ref, mean, std

# ─────────────────────────────────────────────
#  Top-level worker (picklable for spawn)
# ─────────────────────────────────────────────
def _pebble_worker(args):
    """Runs one seed of PEBBLE in a subprocess. Returns (seed, timesteps, means)."""
    seed, kwargs = args
    np.random.seed(seed)
    torch.manual_seed(seed)
    ts, means, _ = train_pebble(seed=seed, **kwargs)
    return seed, ts, means


def run_pebble_seeds_parallel(
    env_fn,
    eval_env_fn,
    gt_reward_fn,
    obs_dim,
    action_dim,
    seeds,
    total_steps  = 100_000,
    query_budget = 250,
    log_dir      = "logs/pebble",
    run_prefix   = "pebble",
    device       = "cpu",
    n_workers    = None,    # None = auto (1 if CUDA, else cpu_count)
    **kwargs,
):
    """
    Parallel multi-seed PEBBLE runner.
    On GPU: auto-falls back to sequential (1 GPU can't run N loops).
    On CPU: uses all cores by default.
    """
    import multiprocessing as mp

    if n_workers is None:
        n_workers = 1 if torch.cuda.is_available() else mp.cpu_count()
    n_workers = min(n_workers, len(seeds))

    common = dict(
        env_fn       = env_fn,
        eval_env_fn  = eval_env_fn,
        gt_reward_fn = gt_reward_fn,
        obs_dim      = obs_dim,
        action_dim   = action_dim,
        total_steps  = total_steps,
        query_budget = query_budget,
        log_dir      = log_dir,
        device       = device,
        **kwargs,
    )
    worker_args = [
        (seed, {**common, "run_name": f"{run_prefix}_seed{seed}"})
        for seed in seeds
    ]

    results   = {}
    seeds_bar = tqdm(total=len(seeds), desc="seeds done",
                     unit="seed", colour="green", leave=True)

    if n_workers == 1:
        for args in worker_args:
            seed, ts, means = _pebble_worker(args)
            results[seed]   = (ts, means)
            seeds_bar.update(1)
            seeds_bar.set_postfix({"last_seed": seed})
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for seed, ts, means in pool.imap_unordered(_pebble_worker, worker_args):
                results[seed] = (ts, means)
                seeds_bar.update(1)
                seeds_bar.set_postfix({"last_seed": seed})
                tqdm.write(f"  ✓ seed {seed} done  ({len(results)}/{len(seeds)})")

    seeds_bar.close()

    ts_ref = results[seeds[0]][0]
    arr    = np.array([results[s][1] for s in seeds])
    mean   = arr.mean(axis=0)
    std    = arr.std(axis=0)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    agg_path = os.path.join(log_dir, f"{run_prefix}_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump({
            "timesteps": ts_ref,
            "mean":      mean.tolist(),
            "std":       std.tolist(),
            "all_seeds": arr.tolist(),
        }, f)
    tqdm.write(f"✓ Saved → {agg_path}")

    return ts_ref, mean, std