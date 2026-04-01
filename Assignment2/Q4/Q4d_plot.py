import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
NUM_SEEDS = 15

# -----------------------
# Helper: compute mean + CI
# -----------------------
def compute_stats(csv_path):
    data = pd.read_csv(csv_path).values  # shape: (seeds, episodes)

    final_perf = data[:, -100:].mean(axis=1)

    mean = final_perf.mean()
    std = final_perf.std()
    ci = 1.96 * std / np.sqrt(NUM_SEEDS)

    return mean, ci


# -----------------------
# BATCH SIZE PLOT
# -----------------------
def plot_batch():
    plt.figure(figsize=(8, 6))

    files = sorted([f for f in os.listdir(RESULTS_DIR) if f.startswith("batch")])

    rho_groups = {}

    for f in files:
        # Example: batch_rho4_bs256.csv
        parts = f.replace(".csv", "").split("_")
        rho = int(parts[1].replace("rho", ""))
        bs = int(parts[2].replace("bs", ""))

        mean, ci = compute_stats(os.path.join(RESULTS_DIR, f))

        if rho not in rho_groups:
            rho_groups[rho] = []

        rho_groups[rho].append((bs, mean, ci))

    # Plot
    for rho in rho_groups:
        rho_groups[rho].sort()
        x = [v[0] for v in rho_groups[rho]]
        y = [v[1] for v in rho_groups[rho]]
        ci = [v[2] for v in rho_groups[rho]]

        plt.errorbar(x, y, yerr=ci, marker='o', capsize=4, label=f"ρ = {rho}")

    plt.xscale("log")
    plt.xticks(x, x)
    plt.xlabel("Batch Size")
    plt.ylabel("Performance (last 50 episodes)")
    plt.title("Sensitivity: Batch Size")
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(RESULTS_DIR, "sensitivity_batch_q.png"), dpi=300)
    plt.show()


# -----------------------
# TARGET UPDATE PLOT
# -----------------------
def plot_target():
    plt.figure(figsize=(8, 6))

    files = sorted([f for f in os.listdir(RESULTS_DIR) if f.startswith("target")])

    rho_groups = {}

    for f in files:
        # Example: target_rho4_tu1000.csv
        parts = f.replace(".csv", "").split("_")
        rho = int(parts[1].replace("rho", ""))
        tu = int(parts[2].replace("tu", ""))

        mean, ci = compute_stats(os.path.join(RESULTS_DIR, f))

        if rho not in rho_groups:
            rho_groups[rho] = []

        rho_groups[rho].append((tu, mean, ci))

    # Plot
    for rho in rho_groups:
        rho_groups[rho].sort()
        x = [v[0] for v in rho_groups[rho]]
        y = [v[1] for v in rho_groups[rho]]
        ci = [v[2] for v in rho_groups[rho]]

        plt.errorbar(x, y, yerr=ci, marker='o', capsize=4, label=f"ρ = {rho}")

    plt.xscale("log")
    plt.xticks(x, x)
    plt.xlabel("Target Update Frequency")
    plt.ylabel("Performance (last 50 episodes)")
    plt.title("Sensitivity: Target Network")
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(RESULTS_DIR, "sensitivity_target_q.png"), dpi=300)
    plt.show()


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    plot_batch()
    plot_target()