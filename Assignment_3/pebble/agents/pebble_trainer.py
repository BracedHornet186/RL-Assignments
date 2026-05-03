"""
PEBBLE training loop.

Algorithm (Lee et al. 2021):
  1. Collect transitions using SAC with learned reward r_ψ.
  2. Every `query_every` steps, sample segment pairs from replay buffer,
     query simulated teacher for labels, add to preference buffer.
  3. Every `reward_update_every` steps, update reward model r_ψ using
     preference buffer (Bradley-Terry cross-entropy loss).
  4. Relabel rewards in SAC replay buffer using current r_ψ.
  5. Update SAC policy using relabelled rewards.
  6. Evaluate against GT reward every eval_every steps.
"""

import sys, os
# Ensure the project root is on sys.path so subprocesses (spawn) can
# import agents/, envs/, utils/ regardless of where the script lives.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import os
import json
from pathlib import Path
from tqdm import tqdm

from agents.pebble import RewardModel, PreferenceBuffer, SimulatedTeacher
from agents.sac import SAC, ReplayBuffer


def collect_segment(replay_buffer, segment_len):
    """Sample a contiguous segment of length segment_len from replay buffer."""
    if replay_buffer.size < segment_len:
        return None, None
    start = np.random.randint(0, replay_buffer.size - segment_len)
    obs = replay_buffer.obs[start: start + segment_len]
    act = replay_buffer.action[start: start + segment_len]
    return obs.copy(), act.copy()


def relabel_replay_buffer(replay_buffer, reward_model, batch_size=512):
    """
    Re-score all transitions in replay buffer using current reward model.
    This is the key PEBBLE trick — constantly relabel as reward model improves.
    """
    device = reward_model.device
    n = replay_buffer.size
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        obs_b = torch.FloatTensor(replay_buffer.obs[start:end]).to(device)
        act_b = torch.FloatTensor(replay_buffer.action[start:end]).to(device)
        with torch.no_grad():
            new_r = reward_model(obs_b, act_b).cpu().numpy()   # (B, 1)
        replay_buffer.reward[start:end] = new_r


def train_pebble(
    env_fn,                  # fn(seed) -> PendulumTargetEnv
    eval_env_fn,             # fn() -> env  (for GT evaluation)
    gt_reward_fn,            # fn(obs_seq, act_seq) -> float (ground-truth)
    obs_dim,
    action_dim,
    total_steps      = 100_000,
    seed             = 0,
    # SAC hyperparams
    lr               = 3e-4,
    gamma            = 0.99,
    tau              = 0.005,
    batch_size       = 256,
    buffer_size      = int(1e6),
    hidden           = (256, 256),
    random_steps     = 10_000,
    # PEBBLE hyperparams
    segment_len      = 50,         # length of each preference segment
    query_every      = 5_000,      # query teacher every N env steps
    n_queries        = 10,         # queries per interaction round
    reward_train_steps = 50,       # gradient steps on reward model per round
    reward_lr        = 3e-4,
    max_pref_buffer  = 3_000,
    query_budget     = None,       # max total queries (None = unlimited)
    # Evaluation
    eval_every       = 10_000,
    eval_episodes    = 20,
    # Logging
    log_dir          = "logs/pebble",
    run_name         = "pebble_run",
    show_pbar        = True,
    device           = "cuda",
):
    """
    Full PEBBLE training for one seed.

    Returns
    -------
    timesteps  : list[int]
    gt_returns : list[float]   mean GT return at each eval point
    all_rets   : list[list]
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Init ────────────────────────────────────────────────────
    env      = env_fn(seed)
    eval_env = eval_env_fn()
    teacher  = SimulatedTeacher(gt_reward_fn)

    reward_model = RewardModel(obs_dim, action_dim, hidden=hidden,
                               lr=reward_lr, device=device)
    pref_buffer  = PreferenceBuffer(max_size=max_pref_buffer,
                                    segment_len=segment_len)

    # SAC uses learned reward — we relabel the buffer periodically
    agent = SAC(obs_dim, action_dim, lr=lr, gamma=gamma, tau=tau,
                batch_size=batch_size, buffer_size=buffer_size,
                hidden=hidden, auto_alpha=True, device=device)

    obs, _ = env.reset(seed=seed)
    ep_ret, timesteps_list, gt_returns_list, all_rets = 0.0, [], [], []
    total_queries = 0

    pbar = tqdm(
        total=total_steps, desc=f"PEBBLE {run_name}",
        unit="step", unit_scale=True, dynamic_ncols=True,
        colour="magenta", leave=True, disable=not show_pbar,
    )

    for step in range(1, total_steps + 1):

        # Random exploration phase
        if step <= random_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, evaluate=False)

        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store with learned reward (will be relabelled later)
        learned_r = reward_model.predict_reward(obs, action)
        agent.store(obs, action, learned_r, next_obs, float(terminated))
        obs = next_obs
        if done:
            obs, _ = env.reset()

        # SAC policy update (using learned reward in buffer)
        if step > random_steps:
            agent.update()

        # ── Query teacher ────────────────────────────────────────
        if step % query_every == 0 and step > random_steps:
            budget_ok = (query_budget is None or
                         total_queries < query_budget)
            if budget_ok:
                for _ in range(n_queries):
                    s1_obs, s1_act = collect_segment(agent.replay, segment_len)
                    s2_obs, s2_act = collect_segment(agent.replay, segment_len)
                    if s1_obs is None or s2_obs is None:
                        continue
                    label = teacher.label(s1_obs, s1_act, s2_obs, s2_act)
                    pref_buffer.add(s1_obs, s1_act, s2_obs, s2_act, label)
                    total_queries += 1
                    if query_budget and total_queries >= query_budget:
                        break

                # ── Update reward model ──────────────────────────
                if len(pref_buffer) >= batch_size:
                    for _ in range(reward_train_steps):
                        s1o, s1a, s2o, s2a, lbl = pref_buffer.sample(
                            min(batch_size, len(pref_buffer)), device
                        )
                        reward_model.update(s1o, s1a, s2o, s2a, lbl)

                    # ── Relabel replay buffer ────────────────────
                    relabel_replay_buffer(agent.replay, reward_model)

        # ── GT evaluation ────────────────────────────────────────
        if step % eval_every == 0:
            gt_rets = []
            for ep_i in range(eval_episodes):
                e_obs, _ = eval_env.reset(seed=10000 + ep_i)
                e_done, e_ret = False, 0.0
                while not e_done:
                    e_act = agent.select_action(e_obs, evaluate=True)
                    e_obs, e_r, e_term, e_trunc, _ = eval_env.step(e_act)
                    e_ret  += e_r
                    e_done  = e_term or e_trunc
                gt_rets.append(e_ret)

            mean_r = np.mean(gt_rets)
            std_r  = np.std(gt_rets)
            timesteps_list.append(step)
            gt_returns_list.append(mean_r)
            all_rets.append(gt_rets)
            pbar.set_postfix({
                "GT_ret":   f"{mean_r:+.1f}",
                "±":        f"{std_r:.1f}",
                "queries":  total_queries,
            })

        pbar.update(1)

    pbar.close()
    env.close()
    eval_env.close()

    # Save logs
    log_path = os.path.join(log_dir, f"{run_name}.json")
    with open(log_path, "w") as f:
        json.dump({
            "timesteps":   timesteps_list,
            "gt_returns":  gt_returns_list,
            "all_returns": all_rets,
            "total_queries": total_queries,
        }, f)

    # Save weights
    agent.save(os.path.join(log_dir, f"{run_name}_sac.pt"))
    reward_model.save(os.path.join(log_dir, f"{run_name}_reward.pt"))

    return timesteps_list, gt_returns_list, all_rets



# ─────────────────────────────────────────────
#  Per-seed worker (top-level so pickle works)
# ─────────────────────────────────────────────
def _pebble_seed_worker(args):
    """Runs one PEBBLE seed in a subprocess. Returns (seed, timesteps, means)."""
    (
        seed, env_fn, eval_env_fn, gt_reward_fn,
        obs_dim, action_dim, total_steps, query_budget,
        log_dir, run_prefix, device, kwargs,
    ) = args

    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)

    ts, means, _ = train_pebble(
        env_fn=env_fn, eval_env_fn=eval_env_fn, gt_reward_fn=gt_reward_fn,
        obs_dim=obs_dim, action_dim=action_dim, total_steps=total_steps,
        seed=seed, query_budget=query_budget, log_dir=log_dir,
        run_name=f"{run_prefix}_seed{seed}", show_pbar=True,
        device=device, **kwargs,
    )
    return seed, ts, means


# ─────────────────────────────────────────────
#  Multi-seed runner  (sequential or parallel)
# ─────────────────────────────────────────────
def run_pebble_seeds(
    env_fn, eval_env_fn, gt_reward_fn, obs_dim, action_dim, seeds,
    total_steps=100_000, query_budget=500, log_dir="logs/pebble",
    run_prefix="pebble", device="cpu", n_workers=8, **kwargs,
):
    """
    Run PEBBLE over multiple seeds, sequentially or in parallel.
    n_workers=None: auto (1 if CUDA, else all cores).
    """
    import multiprocessing as mp
    import torch as _torch

    # if n_workers is None:
    #     n_workers = 1 if _torch.cuda.is_available() else mp.cpu_count()
    n_workers = min(n_workers, len(seeds))

    worker_args = [
        (seed, env_fn, eval_env_fn, gt_reward_fn, obs_dim, action_dim,
         total_steps, query_budget, log_dir, run_prefix, device, kwargs)
        for seed in seeds
    ]

    results = {}
    seeds_bar = tqdm(
        total=len(seeds), desc="seeds done", unit="seed",
        dynamic_ncols=True, colour="green", position=0, leave=True,
    )

    if n_workers == 1:
        for args in worker_args:
            seed, ts, means = _pebble_seed_worker(args)
            results[seed] = (ts, means)
            seeds_bar.update(1)
            seeds_bar.set_postfix({"last_seed": seed})
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for seed, ts, means in pool.imap_unordered(_pebble_seed_worker, worker_args):
                results[seed] = (ts, means)
                seeds_bar.update(1)
                seeds_bar.set_postfix({"last_seed": seed})
                tqdm.write(f"  ✓ seed {seed} finished  ({len(results)}/{len(seeds)})")

    seeds_bar.close()

    timesteps_ref  = results[seeds[0]][0]
    all_seed_means = np.array([results[s][1] for s in seeds])
    mean_over      = all_seed_means.mean(axis=0)
    std_over       = all_seed_means.std(axis=0)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    agg_path = os.path.join(log_dir, f"{run_prefix}_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump({
            "timesteps": timesteps_ref,
            "mean":      mean_over.tolist(),
            "std":       std_over.tolist(),
            "all_seeds": all_seed_means.tolist(),
        }, f)
    tqdm.write(f"✓ Saved aggregated → {agg_path}")

    return timesteps_ref, mean_over, std_over