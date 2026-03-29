"""
plot_truncation.py
------------------
Plots return vs Episodes and return vs Timesteps for all three
truncation lengths (200, 1000, 2000) on the same axes, with 95% CI.

Usage
-----
python plot_truncation.py --log_dir logs --replay_factor 1
python plot_truncation.py --log_dir logs --replay_factor 1 --save
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════
TRUNCATIONS = [200, 1000, 2000]

COLORS = {
    200:  "#e74c3c",   # red
    1000: "#f39c12",   # orange
    2000: "#2c7bb6",   # blue
}


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════
def load_seed_csvs(log_dir: str, truncation: int, replay_factor: int):
    pattern = os.path.join(log_dir, f"trunc{truncation}_rho{replay_factor}_seed*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found for truncation={truncation}:\n  {pattern}"
        )
    return [pd.read_csv(f) for f in files]


# ═══════════════════════════════════════════════════════════════
# Interpolation
# ═══════════════════════════════════════════════════════════════
def interpolate_to_grid(dfs, x_col, y_col, n_points=500):
    x_min  = max(df[x_col].min() for df in dfs)
    x_max  = min(df[x_col].max() for df in dfs)
    x_grid = np.linspace(x_min, x_max, n_points)
    matrix = [np.interp(x_grid, df[x_col].values, df[y_col].values) for df in dfs]
    return x_grid, np.array(matrix)


# ═══════════════════════════════════════════════════════════════
# 95% CI  (t-distribution, correct for small n)
# ═══════════════════════════════════════════════════════════════
def mean_and_ci(matrix, confidence=0.95):
    n    = matrix.shape[0]
    mean = matrix.mean(axis=0)
    se   = matrix.std(axis=0, ddof=1) / np.sqrt(n)
    t    = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean, t * se


# ═══════════════════════════════════════════════════════════════
# Draw one curve + CI onto an existing axis
# ═══════════════════════════════════════════════════════════════
def draw_curve(ax, x_grid, mean, ci, color, label):
    ax.plot(x_grid, mean, color=color, linewidth=2.0, label=label)
    ax.fill_between(x_grid, mean - ci, mean + ci, alpha=0.18, color=color)


# ═══════════════════════════════════════════════════════════════
# Format x-axis  (e.g. 50000 → "50k")
# ═══════════════════════════════════════════════════════════════
def fmt_k(ax):
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
        )
    )


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def make_plot(log_dir, replay_factor, truncations, n_points=500, save=False, out_dir="plots"):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"DQN on MountainCar-v0 — Effect of Truncation Length  (ρ={replay_factor})",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for trunc in truncations:
        try:
            dfs     = load_seed_csvs(log_dir, trunc, replay_factor)
            n_seeds = len(dfs)
            color   = COLORS.get(trunc, "#555555")
            label   = f"trunc={trunc}  (n={n_seeds})"
            print(f"  truncation={trunc:>4}  seeds={n_seeds}")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        # ── Return vs Episodes ─────────────────────────────────
        ep_grid, ep_mat = interpolate_to_grid(dfs, "episode",  "return", n_points)
        ep_mean, ep_ci  = mean_and_ci(ep_mat)
        draw_curve(axes[0], ep_grid, ep_mean, ep_ci, color, label)

        # ── Return vs Timesteps ────────────────────────────────
        ts_grid, ts_mat = interpolate_to_grid(dfs, "timestep", "return", n_points)
        ts_mean, ts_ci  = mean_and_ci(ts_mat)
        draw_curve(axes[1], ts_grid, ts_mean, ts_ci, color, label)

    # ── Axis formatting ───────────────────────────────────────
    for ax, xlabel, title in zip(
        axes,
        ["Episodes", "Timesteps"],
        ["Return vs Episodes", "Return vs Timesteps"],
    ):
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Return", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fmt_k(ax)

    plt.tight_layout()

    if save:
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"truncation_comparison_rho{replay_factor}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved → {fname}")
    else:
        plt.show()

    return fig


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare DQN across truncation lengths")
    p.add_argument("--log_dir",       type=str, default="logs")
    p.add_argument("--replay_factor", type=int, default=1)
    p.add_argument("--truncations",   type=int, nargs="+", default=TRUNCATIONS,
                   help="Truncation lengths to compare (default: 200 1000 2000)")
    p.add_argument("--n_points",      type=int, default=500)
    p.add_argument("--save",          action="store_true")
    p.add_argument("--out_dir",       type=str, default="plots")
    args = p.parse_args()

    print(f"Comparing truncations: {args.truncations}  ρ={args.replay_factor}\n")
    make_plot(
        log_dir       = args.log_dir,
        replay_factor = args.replay_factor,
        truncations   = args.truncations,
        n_points      = args.n_points,
        save          = args.save,
        out_dir       = args.out_dir,
    )