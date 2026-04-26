# Assignment_3/runs/run_reacher_reward_parallel.py

import torch
import torch.multiprocessing as mp
import os
import sys

# Allow "from train_reacher import train_reacher"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_reacher import train_reacher

torch.set_num_threads(1)


def worker(reward_type, seed, device, show_pbar):
    """
    Worker that trains SAC on Reacher for a single reward formulation and seed.

    reward_type: "Ra", "Rb", or "Rc"
    seed:        integer random seed
    device:      "cpu" or "cuda:<id>"
    show_pbar:   whether this worker prints a tqdm progress bar
    """
    print(f"[START] reward={reward_type}, seed={seed}, device={device}", flush=True)

    # NOTE: make sure train_reacher() internally:
    #   - logs average undiscounted return vs env timesteps
    #   - includes an evaluation at timestep 0 (before any training)
    #   - evaluates every 10K steps using a greedy policy over 20 episodes
    train_reacher(
        reward_type=reward_type,
        task="easy",
        seed=seed,
        device=device,
        show_pbar=show_pbar,
        total_steps=250000,  # adjust if you want longer/shorter training
    )

    print(f"[DONE ] reward={reward_type}, seed={seed}", flush=True)


def run_parallel(reward_types, seeds, max_workers=6):
    """
    Launch multiple Reacher runs in parallel across reward formulations and seeds.

    reward_types: list like ["Ra", "Rb", "Rc"]
    seeds:        list of integer seeds
    max_workers:  maximum number of OS processes to run concurrently
    """
    num_gpus = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)

    jobs = [(r, s) for r in reward_types for s in seeds]

    active = []
    idx = 0  # just for deciding which worker shows the progress bar

    while jobs or active:
        # Launch new workers while we have capacity
        while jobs and len(active) < max_workers:
            reward_type, seed = jobs.pop(0)

            # Use CPU by default; adapt if you want to map to specific GPUs
            device = "cpu"
            # if num_gpus > 0:
            #     # simple round-robin mapping if you want GPUs
            #     gpu_id = idx % num_gpus
            #     device = f"cuda:{gpu_id}"

            # Exactly one progress bar per batch of workers
            show_bar_for_this_worker = (idx % max_workers == 0)

            p = mp.Process(
                target=worker,
                args=(reward_type, seed, device, show_bar_for_this_worker),
            )
            p.start()

            active.append(p)
            idx += 1

        # Remove finished workers
        alive = []
        for p in active:
            if p.is_alive():
                alive.append(p)
            else:
                p.join()
        active = alive

    print("\nAll Reacher reward-formulation runs completed!")


if __name__ == "__main__":
    # SAC-Ra, SAC-Rb, SAC-Rc
    reward_types = ["Ra", "Rb", "Rc"]
    # 15 random seeds as per assignment
    seeds = list(range(15))

    run_parallel(reward_types, seeds, max_workers=8)