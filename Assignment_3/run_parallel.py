import torch
import torch.multiprocessing as mp
import os
from train import train


def worker(theta, seed, device):
    print(f"[START] θ={theta}, seed={seed}, device={device}", flush=True)
    train(theta=theta, seed=seed, device=device)
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

            p = mp.Process(target=worker, args=(theta, seed, device))
            p.start()

            active.append(p)
            idx += 1

        # remove finished workers
        active = [p for p in active if p.is_alive()]

    print("\nAll runs completed!")


if __name__ == "__main__":
    targets = [0, -10, 30, -60, 90, -90, 120, -150]
    seeds = list(range(15))

    run_parallel(targets, seeds, max_workers=10)