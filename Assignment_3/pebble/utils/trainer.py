"""
Generic training loop + offline evaluation helpers for SAC / DQN.
Supports parallel seed execution via multiprocessing.
Progress bars via tqdm (pip install tqdm).

Two levels of progress bars:
  1. Per-seed bar  : shows step count + live eval return for each seed.
  2. Seeds bar     : outer bar counting how many seeds have finished.
"""

import numpy as np
import torch
import os
import json
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────
def evaluate_policy_env(agent, env, n_episodes=20):
    """Evaluate deterministic policy. Returns list of undiscounted returns."""
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, ep_ret = False, 0.0
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            done = terminated or truncated
        returns.append(ep_ret)
    return returns


# ─────────────────────────────────────────────
#  Training loop  (single seed)
# ─────────────────────────────────────────────
def train(
    agent,
    train_env,
    eval_env,
    total_steps:   int,
    eval_every:    int  = 10_000,
    eval_episodes: int  = 20,
    random_steps:  int  = 10_000,
    seed:          int  = 0,
    log_dir:       str  = "logs",
    run_name:      str  = "run",
    show_pbar:     bool = True,
    discrete:      bool = False,
    env_step_hook       = None,   # fn(global_step) -> None
    save_every:    int  = None,   # save checkpoint every N steps; None = only at end
):
    """
    Main training loop for one seed.

    Returns
    -------
    timesteps : list[int]
    mean_rets : list[float]
    all_rets  : list[list[float]]
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timesteps, mean_rets, all_rets = [], [], []
    obs, _ = train_env.reset(seed=seed)
    ep_ret, ep_steps = 0.0, 0

    # Per-seed tqdm bar — updates every step, postfix shows latest eval return
    pbar = tqdm(
        total        = total_steps,
        desc         = f"seed {seed:>2d}",
        unit         = "step",
        unit_scale   = True,
        dynamic_ncols= True,
        colour       = "cyan",
        leave        = True,
        disable      = not show_pbar,
    )

    for step in range(1, total_steps + 1):

        if env_step_hook is not None:
            env_step_hook(step)

        # Random exploration phase
        action = (train_env.action_space.sample() if step <= random_steps
                  else agent.select_action(obs, evaluate=False))

        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        done = terminated or truncated
        ep_ret += reward
        ep_steps += 1

        agent.store(obs, action, reward, next_obs, float(terminated))
        obs = next_obs

        if done:
            obs, _ = train_env.reset()
            ep_ret, ep_steps = 0.0, 0

        if step > random_steps:
            agent.update()

        # Offline evaluation — update postfix with latest return
        if step % eval_every == 0:
            rets   = evaluate_policy_env(agent, eval_env, n_episodes=eval_episodes)
            mean_r = np.mean(rets)
            std_r  = np.std(rets)
            timesteps.append(step)
            mean_rets.append(mean_r)
            all_rets.append(rets)
            pbar.set_postfix({"ret": f"{mean_r:+.1f}", "±": f"{std_r:.1f}"})

        # Periodic checkpoint
        if save_every is not None and step % save_every == 0:
            ckpt_path = os.path.join(log_dir, f"{run_name}_step{step}.pt")
            agent.save(ckpt_path)

        pbar.update(1)

    pbar.close()

    log_path = os.path.join(log_dir, f"{run_name}.json")
    with open(log_path, "w") as f:
        json.dump(
            {"timesteps": timesteps, "mean_returns": mean_rets, "all_returns": all_rets},
            f,
        )

    # Save final network weights
    weights_path = os.path.join(log_dir, f"{run_name}.pt")
    agent.save(weights_path)

    return timesteps, mean_rets, all_rets


# ─────────────────────────────────────────────
#  Per-seed worker  (top-level so pickle works)
# ─────────────────────────────────────────────
def _seed_worker(args):
    """Runs one seed inside a subprocess. Returns (seed, timesteps, seed_means)."""
    (
        seed,
        agent_fn,
        train_env_fn,
        eval_env_fn,
        total_steps,
        eval_every,
        eval_episodes,
        random_steps,
        log_dir,
        run_prefix,
        show_pbar,
        discrete,
        env_step_hook_fn,
        save_every,
    ) = args

    np.random.seed(seed)
    torch.manual_seed(seed)

    agent     = agent_fn(seed)
    train_env = train_env_fn(seed)
    eval_env  = eval_env_fn()
    hook      = env_step_hook_fn(seed) if env_step_hook_fn else None

    ts, _, rets_per_eval = train(
        agent,
        train_env,
        eval_env,
        total_steps   = total_steps,
        eval_every    = eval_every,
        eval_episodes = eval_episodes,
        random_steps  = random_steps,
        seed          = seed,
        log_dir       = log_dir,
        run_name      = f"{run_prefix}_seed{seed}",
        show_pbar     = show_pbar,
        discrete      = discrete,
        env_step_hook = hook,
        save_every    = save_every,
    )

    seed_means = [np.mean(r) for r in rets_per_eval]
    train_env.close()
    eval_env.close()
    return seed, ts, seed_means


# ─────────────────────────────────────────────
#  Multi-seed runner
# ─────────────────────────────────────────────
def run_seeds(
    agent_fn,
    train_env_fn,
    eval_env_fn,
    seeds,
    total_steps,
    eval_every       = 10_000,
    eval_episodes    = 20,
    random_steps     = 10_000,
    log_dir          = "logs",
    run_prefix       = "run",
    verbose          = True,
    discrete         = False,
    env_step_hook_fn = None,
    n_workers        = None,
    save_every       = None,   # e.g. 50_000 saves a .pt every 50K steps per seed
):
    """
    Run training across multiple seeds, sequentially or in parallel.

    Progress bars
    -------------
    Sequential (n_workers=1):
        Each seed shows its own cyan tqdm bar with live return in the postfix.
        A green outer bar counts seeds completed.

    Parallel (n_workers > 1):
        Each subprocess renders its own cyan bar (they stack in the terminal).
        The main process shows a green "seeds done" bar that ticks as each
        subprocess finishes.

    Parameters
    ----------
    n_workers : int or None
        None  → mp.cpu_count() parallel workers.
        N > 1 → exactly N workers.
        1     → sequential, easiest to debug.

    All fn arguments (agent_fn, etc.) must be picklable — define them at
    module level, not as lambdas or closures over unpicklable objects.
    """
    # Auto-select n_workers:
    # - GPU available → 1 worker (single GPU can't meaningfully run multiple
    #   training loops in parallel; they just serialize on the device)
    # - CPU only → all cores
    if n_workers is None:
        n_workers = 1 if torch.cuda.is_available() else mp.cpu_count()
    n_workers = min(n_workers, len(seeds))

    worker_args = [
        (
            seed,
            agent_fn,
            train_env_fn,
            eval_env_fn,
            total_steps,
            eval_every,
            eval_episodes,
            random_steps,
            log_dir,
            run_prefix,
            True,
            discrete,
            env_step_hook_fn,
            save_every,
        )
        for seed in seeds
    ]

    results = {}   # seed -> (timesteps, seed_means)

    # Outer "seeds done" bar — always visible in main process
    seeds_bar = tqdm(
        total        = len(seeds),
        desc         = "seeds done",
        unit         = "seed",
        dynamic_ncols= True,
        colour       = "green",
        position     = 0,
        leave        = True,
    )

    if n_workers == 1:
        # ── Sequential ───────────────────────────────────────────
        for args in worker_args:
            seed, ts, means = _seed_worker(args)
            results[seed]   = (ts, means)
            seeds_bar.update(1)
            seeds_bar.set_postfix({"last_seed": seed})
    else:
        # ── Parallel (spawn avoids CUDA/OpenGL fork issues) ──────
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for seed, ts, means in pool.imap_unordered(_seed_worker, worker_args):
                results[seed] = (ts, means)
                seeds_bar.update(1)
                seeds_bar.set_postfix({"last_seed": seed})
                tqdm.write(f"  ✓ seed {seed} finished  ({len(results)}/{len(seeds)})")

    seeds_bar.close()

    # ── Aggregate over seeds ─────────────────────────────────────
    timesteps_ref   = results[seeds[0]][0]
    all_seed_means  = np.array([results[s][1] for s in seeds])   # [n_seeds, n_evals]
    mean_over_seeds = all_seed_means.mean(axis=0)
    std_over_seeds  = all_seed_means.std(axis=0)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    agg_path = os.path.join(log_dir, f"{run_prefix}_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump(
            {
                "timesteps": timesteps_ref,
                "mean":      mean_over_seeds.tolist(),
                "std":       std_over_seeds.tolist(),
                "all_seeds": all_seed_means.tolist(),
            },
            f,
        )
    tqdm.write(f"✓ Aggregated results saved to {agg_path}")

    return timesteps_ref, mean_over_seeds, std_over_seeds