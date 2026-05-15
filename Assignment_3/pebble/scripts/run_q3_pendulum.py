"""
Q3 (Bonus) Part 1 & 2 — PEBBLE on modified Pendulum-v1.

Part 1: Compare SAC (GT reward) vs PEBBLE for θ_target ∈ {0,-60,90,120,-150}
Part 2: Different feedback budgets for θ_target = 90
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import json
from pathlib import Path

SEEDS         = list(range(15))
TOTAL_STEPS   = 100_000
EVAL_EVERY    = 10_000
EVAL_EPS      = 20
RANDOM_STEPS  = 10_000
QUERY_EVERY   = 5_000
N_QUERIES     = 10
SEGMENT_LEN   = 50
REWARD_STEPS  = 50
LOG_DIR       = "logs/q3_pebble_pendulum"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

OBS_DIM    = 3   # Pendulum: [cos θ, sin θ, dθ/dt]
ACTION_DIM = 1

TARGET_ANGLES = [0, -60, 90, 120, -150]

# Feedback budgets for Part 2
BUDGETS = [50, 100, 200, 500, 1000]

import functools
# ── Env / reward factories ───────────────────────────────────────
def _create_pendulum_env(theta_deg):
        from envs.pendulum import PendulumTargetEnv
        env = PendulumTargetEnv(theta_target_deg=theta_deg)
        return env

def _create_pendulum_env_with_seed(seed, theta_deg):
    from envs.pendulum import PendulumTargetEnv
    env = PendulumTargetEnv(theta_target_deg=theta_deg)
    env.reset(seed=seed)
    return env

# 2. Bind only theta_deg. The resulting object will expect 'seed' when called.
def make_pendulum_env(theta_deg):
    return functools.partial(_create_pendulum_env_with_seed, theta_deg=theta_deg)

def make_pendulum_eval_env(theta_deg):
    # functools.partial creates a picklable callable 
    # that "remembers" the theta_deg argument.
    return functools.partial(_create_pendulum_env, theta_deg=theta_deg)

def _compute_gt_segment_return(obs_seq, act_seq, theta_deg):
    from envs.pendulum import PendulumTargetEnv
    # Instantiate the env locally within the worker process when called
    env = PendulumTargetEnv(theta_target_deg=theta_deg)
    return env.gt_segment_return(obs_seq, act_seq)


# 2. The factory function now returns a picklable partial object
def make_gt_reward_fn(theta_deg):
    """Returns a picklable function that computes GT return for a segment."""
    return functools.partial(_compute_gt_segment_return, theta_deg=theta_deg)


def make_agent(seed):
    from agents.sac import SAC
    return SAC(obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                lr=3e-4, gamma=0.99, tau=0.005,
                batch_size=256, buffer_size=200_000,
                hidden=(256, 256), auto_alpha=True, device=DEVICE)

def get_missing_seeds(seeds, log_dir, prefix):
    """Checks individual seed JSON files to ensure they reached TOTAL_STEPS."""
    missing = []
    for seed in seeds:
        # Based on your trainer, individual seeds are saved as {prefix}_seed{seed}.json[cite: 3]
        path = os.path.join(log_dir, f"{prefix}_seed{seed}.json")
        if not os.path.exists(path):
            missing.append(seed)
            continue
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            # Check if it hit the exact number of steps[cite: 3]
            if not data.get("timesteps") or data["timesteps"][-1] < TOTAL_STEPS:
                missing.append(seed)
        except (json.JSONDecodeError, KeyError):
            missing.append(seed) # File is corrupted, needs retraining
    return missing

def load_and_aggregate(seeds, log_dir, prefix):
    """Manually aggregates all seeds using the verified JSON keys."""
    all_means = []
    ts_ref = None
    for seed in seeds:
        path = os.path.join(log_dir, f"{prefix}_seed{seed}.json")
        with open(path, 'r') as f:
            data = json.load(f)
            
            # Dynamically grab the rewards whether it's a SAC log or a PEBBLE log
            returns = data.get("mean_returns") or data.get("gt_returns")
            
            if returns is None:
                raise KeyError(
                    f"Could not find the reward array in {path}.\n"
                    f"Keys found: {list(data.keys())}"
                )
                
            all_means.append(returns)
            if ts_ref is None:
                ts_ref = data["timesteps"]
    
    arr = np.array(all_means)
    return ts_ref, arr.mean(axis=0), arr.std(axis=0)

# ── SAC with GT reward (baseline) ───────────────────────────────
def run_sac_gt(theta_deg, seeds, log_dir):
    """Train SAC with ground-truth reward — the upper-bound baseline."""
    prefix = f"sac_gt_theta{theta_deg}"
    
    missing_seeds = get_missing_seeds(seeds, log_dir, prefix)

    if not missing_seeds:
        print(f"⏭️  Skipping {prefix} - all {len(seeds)} seeds complete.")
        return load_and_aggregate(seeds, log_dir, prefix)

    if missing_seeds != seeds:
        print(f"⚠️  Resuming {prefix}. Training missing seeds: {missing_seeds}")
    else:
        print(f"▶️  Starting {prefix}...")

    from agents.sac import SAC
    from utils.trainer import run_seeds

    # Only pass the missing_seeds to save compute time
    run_seeds(
        agent_fn     = make_agent,
        train_env_fn = make_pendulum_env(theta_deg),
        eval_env_fn  = make_pendulum_eval_env(theta_deg),
        seeds        = missing_seeds,
        total_steps  = TOTAL_STEPS,
        eval_every   = EVAL_EVERY,
        eval_episodes= EVAL_EPS,
        random_steps = RANDOM_STEPS,
        log_dir      = log_dir,
        run_prefix   = prefix,
        n_workers    = 8,
    )
    
    # Aggregate ALL seeds (both previously completed and newly finished)
    return load_and_aggregate(seeds, log_dir, prefix)


# ── PEBBLE ───────────────────────────────────────────────────────
def run_pebble(theta_deg, seeds, log_dir, budget=500, run_prefix=None):
    from agents.pebble_trainer import run_pebble_seeds_parallel

    prefix = run_prefix or f"pebble_theta{theta_deg}_budget{budget}"
    
    missing_seeds = get_missing_seeds(seeds, log_dir, prefix)

    if not missing_seeds:
        print(f"⏭️  Skipping {prefix} - all {len(seeds)} seeds complete.")
        return load_and_aggregate(seeds, log_dir, prefix)

    if missing_seeds != seeds:
        print(f"⚠️  Resuming {prefix}. Training missing seeds: {missing_seeds}")
    else:
        print(f"▶️  Starting {prefix}...")

    # Only pass the missing_seeds to save compute time
    run_pebble_seeds_parallel(
        env_fn        = make_pendulum_env(theta_deg),
        eval_env_fn   = make_pendulum_eval_env(theta_deg),
        gt_reward_fn  = make_gt_reward_fn(theta_deg),
        obs_dim       = OBS_DIM,
        action_dim    = ACTION_DIM,
        seeds         = missing_seeds,
        total_steps   = TOTAL_STEPS,
        query_budget  = budget,
        query_every   = QUERY_EVERY,
        n_queries     = N_QUERIES,
        segment_len   = SEGMENT_LEN,
        reward_train_steps = REWARD_STEPS,
        eval_every    = EVAL_EVERY,
        eval_episodes = EVAL_EPS,
        random_steps  = RANDOM_STEPS,
        log_dir       = log_dir,
        run_prefix    = prefix,
        device        = DEVICE,
        n_workers     = 8,
    )
    
    # Aggregate ALL seeds (both previously completed and newly finished)
    return load_and_aggregate(seeds, log_dir, prefix)

if __name__ == "__main__":
    from utils.plotting import plot_curves
    print(f"Using device: {DEVICE}")
 
    COLORS = ["#9C27B0", "#F44336", "#FF9800", "#4CAF50", "#00BCD4"]
 
    # ── Part 2: Budget ablation for ALL θ_target angles ──────────
    print("\n=== Part 2: Budget ablation for all θ_target angles ===")
 
    for theta in TARGET_ANGLES:
        print(f"\n{'='*55}")
        print(f"  θ_target = {theta}°")
        print(f"{'='*55}")
 
        log_sub = f"{LOG_DIR}/budget_ablation_theta{theta}"
        curves  = []
 
        # SAC-GT reference
        ts_gt, mean_gt, std_gt = run_sac_gt(theta, SEEDS, log_sub)
        curves.append({
            "label": "SAC (GT reward)",
            "timesteps": ts_gt, "mean": mean_gt, "std": std_gt,
            "color": "black",
        })
 
        # PEBBLE with each budget
        for budget, color in zip(BUDGETS, COLORS):
            print(f"\n  Budget = {budget}")
            ts_pb, mean_pb, std_pb = run_pebble(
                theta, SEEDS, log_sub, budget=budget,
                run_prefix=f"pebble_budget{budget}",
            )
            curves.append({
                "label":     f"PEBBLE (budget={budget})",
                "timesteps": ts_pb,
                "mean":      mean_pb,
                "std":       std_pb,
                "color":     color,
            })
 
        plot_curves(
            curves,
            title     = f"PEBBLE Budget Ablation  |  θ_target = {theta}°",
            ylabel    = "Average Undiscounted Return (GT)",
            save_path = f"{log_sub}/plots/budget_ablation.png",
        )
        print(f"  ✓ Plot saved for θ={theta}°")
 
    print("\nDone.")
