import torch
import torch.multiprocessing as mp
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Ensure this import matches your folder structure. 
# Based on your previous file, it was inside the "runs" folder.
from train_pendulum import train

torch.set_num_threads(1)

def worker(job_kwargs, device, show_pbar):
    print(f"[START] {job_kwargs['exp_name']}, device={device}", flush=True)
    # Pass all dictionary arguments directly into train()
    train(
        **job_kwargs, 
        device=device, 
        show_pbar=show_pbar,
        total_steps=120000
    )
    print(f"[DONE ] {job_kwargs['exp_name']}", flush=True)


def run_parallel(jobs, max_workers=4):
    num_gpus = torch.cuda.device_count()

    mp.set_start_method("spawn", force=True)

    active = []
    idx = 0

    while jobs or active:

        # launch new workers
        while jobs and len(active) < max_workers:
            job_kwargs = jobs.pop(0)
            
            
            device = "cpu"
            
            # True for idx 0, max_workers, 2*max_workers, etc. (One bar per batch)
            show_bar_for_this_worker = (idx % max_workers == 0)

            p = mp.Process(target=worker, args=(job_kwargs, device, show_bar_for_this_worker))
            p.start()
            active.append(p)
            idx += 1

        for p in active:
            p.join(0.1)
            if not p.is_alive():
                active.remove(p)


if __name__ == "__main__":
    
    # Using 3 seeds to keep the total compute time reasonable for the grid search.
    # You can increase this to range(5) or range(15) if you have the compute power.
    seeds = range(15)
    jobs = []

    # =========================================================================
    # PART 5(a): Grid search for manual alpha
    # Targets: {-60, 90, 120, -150}
    # =========================================================================
    # targets_5a = [-60, 90, 120, -150]
    # alphas_to_test = [0.01, 0.05, 0.2, 0.5]

    # for theta in targets_5a:
    #     for alpha in alphas_to_test:
    #         for seed in seeds:
    #             jobs.append({
    #                 "theta": theta,
    #                 "seed": seed,
    #                 "learnable_temperature": False,
    #                 "init_temperature": alpha,
    #                 "reward_scale": 1.0,
    #                 "exp_name": f"q5a_theta{theta}_seed{seed}_manual_a{alpha}"
    #             })

    # =========================================================================
    # PART 5(b): Reward scaling for theta = 90
    # Scales: 10x and 0.1x
    # =========================================================================
    target_5b = 90
    scales_5b = [10.0, 0.1]
    
    # # NOTE: You are supposed to analyze the 5(a) plots to find the single best 
    # # manual alpha for theta=90, and plug it in here. 
    # # We are using 0.05 as a placeholder so the script runs completely.
    best_alpha_90 = 0.01

    for scale in scales_5b:
        for seed in seeds:
            # (i) SAC with manually tuned alpha
            jobs.append({
                "theta": target_5b,
                "seed": seed,
                "learnable_temperature": False,
                "init_temperature": best_alpha_90,
                "reward_scale": scale,
                "exp_name": f"q5b_theta{target_5b}_seed{seed}_manual_a{best_alpha_90}_rs{scale}"
            })
            
            # (ii) SAC with automated tuning
            jobs.append({
                "theta": target_5b,
                "seed": seed,
                "learnable_temperature": True,
                "init_temperature": 0.1,  # Starting point doesn't matter much for auto
                "reward_scale": scale,
                "exp_name": f"q5b_theta{target_5b}_seed{seed}_auto_rs{scale}"
            })

    print(f"Total jobs scheduled: {len(jobs)}")
    
    # Adjust max_workers based on your machine's CPU/RAM. 
    # 4 to 6 is usually safe for 16GB of RAM.
    run_parallel(jobs, max_workers=8)