import torch
import torch.multiprocessing as mp
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_pendulum import train

torch.set_num_threads(1)
def worker(theta, seed, device, show_pbar):
    print(f"[START] θ={theta}, seed={seed}, device={device}", flush=True)
    # Pass the show_pbar flag into train()
    train(theta=theta, seed=seed, device=device, show_pbar=show_pbar,total_steps=120000)
    print(f"[DONE ] θ={theta}, seed={seed}", flush=True)


def run_parallel(targets, seeds, max_workers=4):
    num_gpus = torch.cuda.device_count()

    mp.set_start_method("spawn", force=True)

    processes = []
    job_list = [(theta, seed) for theta in targets for seed in seeds]

    active = []
    idx = 0

    while job_list or active:

        # launch new workers
        while job_list and len(active) < max_workers:
            theta, seed = job_list.pop(0)
            device = "cpu"
            
            # True for idx 0, 8, 16, etc... (Exactly one bar per batch!)
            show_bar_for_this_worker = (idx % max_workers == 0)

            p = mp.Process(target=worker, args=(theta, seed, device, show_bar_for_this_worker))
            p.start()

            active.append(p)
            idx += 1

        # remove finished workers
        active = [p for p in active if p.is_alive()]

    print("\nAll runs completed!")

if __name__ == "__main__":
    targets = [0,-10,30, -60, 90, -90, 120, -150]
    #targets = [-10]
    seeds = list(range(15))

    run_parallel(targets, seeds, max_workers=8)