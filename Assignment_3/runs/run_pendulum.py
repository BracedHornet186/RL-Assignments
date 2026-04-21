import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_pendulum import train

targets = [0, -10, 30, -60, 90, -90, 120, -150]
seeds = list(range(15))

for theta in targets:
    for seed in seeds:
        print(f"\nRunning θ={theta}, seed={seed}", flush=True)
        train(theta, seed,device='cuda')