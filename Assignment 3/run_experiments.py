"""
run_experiments.py  — FIXED VERSION
Runs SAC-Ra/Rb/Rc for 15 seeds with proper parallelism.

KEY FIX: never launch all 45 processes at once.
Use a process pool of size N_PARALLEL (default=3 — one per reward type).
Each process gets its own GPU context. On a single RTX 3060 with 12GB VRAM,
running 3 processes simultaneously is safe; each uses ~1–2 GB.

Recommended usage:
  # One reward type at a time (safest, most predictable):
  python run_experiments.py --reward ra
  python run_experiments.py --reward rb
  python run_experiments.py --reward rc

  # All 3 reward types in parallel, 5 seeds at a time per type:
  python run_experiments.py --parallel --n_parallel 3

  # Resume (already-done seeds are skipped automatically):
  python run_experiments.py --parallel --n_parallel 3
"""

import os, sys, subprocess, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

REWARD_TYPES = ["ra", "rb", "rc"]
SEEDS        = list(range(15))


def run_one(reward, seed, save_dir, steps):
    """Worker: runs one (reward, seed) in a subprocess."""
    cmd = [sys.executable, "sac_reacher.py",
           "--reward", reward, "--seed", str(seed),
           "--save_dir", save_dir, "--steps", str(steps)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    tag = f"R{reward.upper()}_s{seed}"
    if result.returncode != 0:
        print(f"  [ERROR] {tag}:\n{result.stderr[-400:]}")
    else:
        # Print last line of stdout (progress summary)
        lines = result.stdout.strip().splitlines()
        if lines: print(f"  [DONE] {tag}: {lines[-1]}")
    return tag, result.returncode


def run_sequential(reward_filter, seed_filter, save_dir, steps):
    from sac_reacher import train_sac
    rewards = [reward_filter] if reward_filter else REWARD_TYPES
    seeds   = [seed_filter]   if seed_filter is not None else SEEDS
    for r in rewards:
        for s in seeds:
            print(f"\n{'='*55}\n  SAC-R{r.upper()} seed={s}\n{'='*55}")
            train_sac(r, s, steps, save_dir)


def run_parallel(n_parallel, save_dir, steps):
    """
    Submit all (reward, seed) jobs to a pool of size n_parallel.
    Jobs for already-completed seeds are skipped by train_sac itself.
    """
    jobs = [(r, s) for r in REWARD_TYPES for s in SEEDS]
    print(f"  Submitting {len(jobs)} jobs, max {n_parallel} at a time...")
    print(f"  Already-completed seeds will be skipped automatically.\n")

    done = 0
    with ProcessPoolExecutor(max_workers=n_parallel) as ex:
        futs = {ex.submit(run_one, r, s, save_dir, steps): (r,s)
                for r,s in jobs}
        for fut in as_completed(futs):
            tag, rc = fut.result()
            done += 1
            print(f"  [{done}/{len(jobs)}] {tag} finished (rc={rc})")

    print("\nAll jobs complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward",     type=str, default=None,
                        choices=["ra","rb","rc"])
    parser.add_argument("--seed",       type=int, default=None)
    parser.add_argument("--parallel",   action="store_true",
                        help="Use process pool (throttled)")
    parser.add_argument("--n_parallel", type=int, default=3,
                        help="Max simultaneous processes (default 3)")
    parser.add_argument("--steps",      type=int, default=500_000)
    parser.add_argument("--save_dir",   type=str, default="results")
    args = parser.parse_args()

    if args.parallel:
        run_parallel(args.n_parallel, args.save_dir, args.steps)
    else:
        run_sequential(args.reward, args.seed, args.save_dir, args.steps)
