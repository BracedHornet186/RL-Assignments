"""
run_experiments.py — SAC-Ra/Rb/Rc for 15 seeds with live per-slot tqdm monitoring.

Each parallel "slot" owns its own tqdm bar that is driven by a small progress
file the worker writes every 1K steps. When a slot's job finishes, the slot
picks the next pending (reward, seed) from the queue and its bar resets.

Resumption is automatic: any (reward, seed) whose results JSON already exists
is skipped.

Usage
-----
    # Single job:
    python run_experiments.py --reward rc --seed 0

    # Sequential over all reward types × seeds:
    python run_experiments.py

    # Parallel — 5 simultaneous workers, with one live bar per worker slot:
    python run_experiments.py --parallel --n_parallel 5

    # Restrict parallel run to a single reward type:
    python run_experiments.py --reward rc --parallel --n_parallel 5
"""

import os, sys, time, threading, subprocess, argparse
from queue import Queue, Empty
from tqdm import tqdm

REWARD_TYPES = ["ra", "rb", "rc"]
SEEDS        = list(range(15))

HERE       = os.path.dirname(os.path.abspath(__file__))
SAC_SCRIPT = os.path.join(HERE, "sac_reacher.py")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def result_path(save_dir, reward, seed):
    return os.path.join(save_dir, f"sac_r{reward}_seed{seed}.json")


def read_progress(prog_file):
    """Return current step count from worker, or None on transient failure."""
    try:
        with open(prog_file) as f:
            return int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


# ─────────────────────────────────────────────
# Sequential (no parallelism, no bar magic)
# ─────────────────────────────────────────────

def run_sequential(reward_filter, seed_filter, save_dir, steps):
    from sac_reacher import train_sac
    rewards = [reward_filter] if reward_filter else REWARD_TYPES
    seeds   = [seed_filter]   if seed_filter is not None else SEEDS
    for r in rewards:
        for s in seeds:
            print(f"\n{'='*55}\n  SAC-R{r.upper()} seed={s}\n{'='*55}")
            train_sac(r, s, steps, save_dir)


# ─────────────────────────────────────────────
# Parallel slot-pool with live tqdm
# ─────────────────────────────────────────────

def _slot_worker(slot, job_queue, save_dir, log_dir, prog_dir, steps,
                 done_counter, total_jobs, lock):
    bar = tqdm(
        total=steps,
        position=slot,
        leave=True,
        unit="step",
        unit_scale=False,
        dynamic_ncols=True,
        bar_format=("  {desc:<16}{percentage:5.1f}%|{bar:24}| "
                    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"),
        desc=f"slot{slot} idle",
    )

    while True:
        try:
            r, s = job_queue.get_nowait()
        except Empty:
            break

        tag       = f"R{r.upper()}_s{s:02d}"
        prog_file = os.path.join(prog_dir, f"r{r}_s{s}.txt")
        log_file  = os.path.join(log_dir,  f"r{r}_s{s}.log")
        try: os.remove(prog_file)
        except OSError: pass

        bar.reset(total=steps)
        bar.set_description(tag)
        bar.refresh()

        cmd = [sys.executable, SAC_SCRIPT,
               "--reward",        r,
               "--seed",          str(s),
               "--save_dir",      save_dir,
               "--steps",         str(steps),
               "--progress_file", prog_file]
        with open(log_file, "w") as logf:
            logf.write(f"# CMD: {' '.join(cmd)}\n\n"); logf.flush()
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
            last_n = 0
            while proc.poll() is None:
                cur = read_progress(prog_file)
                if cur is not None and cur > last_n:
                    bar.update(cur - last_n)
                    last_n = cur
                time.sleep(1.0)
            rc_ = proc.wait()

        # Drain the bar to 100% on a clean exit so the visual matches reality.
        if rc_ == 0:
            bar.update(max(0, steps - last_n))
        with lock:
            done_counter[0] += 1
            status = "ok" if rc_ == 0 else f"FAIL(rc={rc_})"
            tqdm.write(f"  [{done_counter[0]}/{total_jobs}] {tag}  {status}  "
                       f"log={log_file}")

    bar.set_description("(slot done)")
    bar.refresh()
    bar.close()


def run_parallel(reward_filter, n_parallel, save_dir, steps):
    log_dir  = os.path.join(save_dir, "logs")
    prog_dir = os.path.join(save_dir, "progress")
    for d in (save_dir, log_dir, prog_dir):
        os.makedirs(d, exist_ok=True)

    rewards = [reward_filter] if reward_filter else REWARD_TYPES
    all_pairs = [(r, s) for r in rewards for s in SEEDS]
    pending   = [(r, s) for (r, s) in all_pairs
                 if not os.path.exists(result_path(save_dir, r, s))]
    skipped   = len(all_pairs) - len(pending)

    if not pending:
        print(f"  All {skipped} jobs already done. Nothing to do.")
        return

    n_slots = min(n_parallel, len(pending))
    print(f"  Pending: {len(pending)} jobs (skipping {skipped} already-done)")
    print(f"  Pool: {n_slots} slots × {steps:,} steps/job")
    print(f"  Logs: {log_dir}/r<reward>_s<seed>.log")
    print(f"  Progress files: {prog_dir}/r<reward>_s<seed>.txt\n")

    q = Queue()
    for j in pending: q.put(j)

    lock         = threading.Lock()
    done_counter = [0]

    threads = [threading.Thread(
                   target=_slot_worker,
                   args=(i, q, save_dir, log_dir, prog_dir, steps,
                         done_counter, len(pending), lock),
                   daemon=True)
               for i in range(n_slots)]
    for t in threads: t.start()
    try:
        for t in threads: t.join()
    except KeyboardInterrupt:
        print("\n  Interrupted — workers will finish their current job.")
        for t in threads: t.join()

    # Move cursor past all the bars before printing the summary.
    sys.stdout.write("\n" * (n_slots + 1)); sys.stdout.flush()
    print(f"All done: {done_counter[0]}/{len(pending)} jobs completed.")


# ─────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward",     type=str, default=None,
                        choices=["ra","rb","rc"])
    parser.add_argument("--seed",       type=int, default=None)
    parser.add_argument("--parallel",   action="store_true",
                        help="Use slot-pool with live tqdm bars")
    parser.add_argument("--n_parallel", type=int, default=5,
                        help="Number of parallel slots (default 5)")
    parser.add_argument("--steps",      type=int, default=500_000)
    parser.add_argument("--save_dir",   type=str, default="results")
    args = parser.parse_args()

    if args.parallel:
        run_parallel(args.reward, args.n_parallel, args.save_dir, args.steps)
    else:
        run_sequential(args.reward, args.seed, args.save_dir, args.steps)
