"""
plot_q3.py — Generates all plots for Section 3.

Produces:
  - q3_2_dqn_trunc2000.pdf   : Return vs timesteps for vanilla DQN (trunc=2000)
  - q3_3_truncation_compare.pdf : Comparison of trunc=200, 1000, 2000

Smoothing: 20-episode rolling mean per seed before computing mean/CI across seeds.
CI: 95% confidence interval  (mean ± 1.96 * SEM)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
})

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────
def load(tag: str):
    path = os.path.join(RESULTS_DIR, f"{tag}.json")
    with open(path) as f:
        data = json.load(f)
    return data["returns"], data["steps"]


def smooth(arr: list, window: int = 20) -> np.ndarray:
    """Rolling mean over a 1-D array."""
    arr = np.array(arr, dtype=np.float32)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def interpolate_to_common_grid(all_returns, all_steps, n_points=200):
    """
    Smooth each seed's curve then interpolate onto a common timestep grid
    so we can compute mean/std across seeds.
    Returns (grid, matrix) where matrix[i] is seed i's interpolated curve.
    """
    # Determine common x range
    max_step = min(max(s[-1] for s in all_steps), int(2e5))
    grid = np.linspace(0, max_step, n_points)

    matrix = []
    for returns, steps in zip(all_returns, all_steps):
        s = np.array(steps, dtype=np.float64)
        r = smooth(returns, window=20)
        # smooth() shortens by (window-1), align steps accordingly
        s = s[len(s)-len(r):]
        # prepend origin so interpolation covers x=0
        s = np.concatenate([[0], s])
        r = np.concatenate([[r[0]], r])
        interp = np.interp(grid, s, r)
        matrix.append(interp)

    return grid, np.array(matrix)


def plot_mean_ci(ax, grid, matrix, label, color):
    mean = matrix.mean(axis=0)
    sem  = matrix.std(axis=0) / np.sqrt(len(matrix))
    ci   = 1.96 * sem
    ax.plot(grid, mean, label=label, color=color, lw=2)
    ax.fill_between(grid, mean - ci, mean + ci, alpha=0.25, color=color)


# ─────────────────────────────────────────────
# Plot 1: Q3.2 — Vanilla DQN, trunc=2000
# ─────────────────────────────────────────────
def plot_q3_2():
    returns, steps = load("dqn_trunc2000_rho1")
    grid, matrix   = interpolate_to_common_grid(returns, steps)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_mean_ci(ax, grid, matrix, label="DQN (ρ=1, trunc=2000)", color="#1f77b4")

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Return (per episode)")
    ax.set_title("Vanilla DQN on MountainCar-v0\n(mean ± 95% CI, 15 seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "q3_2_dqn_trunc2000.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


# ─────────────────────────────────────────────
# Plot 2: Q3.3 — Truncation comparison
# ─────────────────────────────────────────────
def plot_q3_3():
    configs = [
        ("dqn_trunc200_rho1",  "trunc=200",  "#d62728"),
        ("dqn_trunc1000_rho1", "trunc=1000", "#ff7f0e"),
        ("dqn_trunc2000_rho1", "trunc=2000", "#1f77b4"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, label, color in configs:
        returns, steps = load(tag)
        grid, matrix   = interpolate_to_common_grid(returns, steps)
        plot_mean_ci(ax, grid, matrix, label=label, color=color)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Return (per episode)")
    ax.set_title("DQN on MountainCar-v0: Effect of Truncation Length\n(mean ± 95% CI, 15 seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "q3_3_truncation_compare.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    plot_q3_2()
    plot_q3_3()
    print("All Q3 plots generated.")
