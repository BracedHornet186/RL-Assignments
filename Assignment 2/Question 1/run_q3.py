"""
run_q3.py  — Runs all experiments for Section 3 of PA2.

Section 3.2: Vanilla DQN with truncation=2000 (15 seeds)
Section 3.3: DQN with truncation=200, 1000, 2000 (15 seeds each)

Usage:
    python run_q3.py [--device cpu|cuda] [--seeds 15] [--timesteps 200000]
"""

import argparse
from dqn import run_experiment_parallel
import multiprocessing as mp

if __name__ == "__main__":

    # CRITICAL: Must be at the very top of the main block
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--device",    type=str, default="cuda")
    parser.add_argument("--seeds",     type=int, default=15)
    parser.add_argument("--workers",   type=int, default=15)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--save_dir",  type=str, default="results")
    args = parser.parse_args()

    TRUNCATIONS = [200, 1000, 2000]

    for trunc in TRUNCATIONS:
        tag = f"dqn_trunc{trunc}_rho1"
        print(f"\n{'='*50}")
        print(f" Running: truncation={trunc}  tag={tag}")
        print(f"{'='*50}")
        run_experiment_parallel(
            n_seeds=args.seeds,
            truncation_length=trunc,
            total_timesteps=args.timesteps,
            replay_factor=1,
            batch_size=64,
            target_update_freq=500,
            save_dir=args.save_dir,
            tag=tag,
            device=args.device,
            max_workers=args.workers
        )

    print("\nAll Q3 experiments complete.")
