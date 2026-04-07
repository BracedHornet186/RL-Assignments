"""
per_plot.py
-----------
Loads all trunc2000_rho{R}_seed*.csv files and plots mean return
vs Episodes and vs Timesteps for all ρ values on the same axes.

Since all runs are for the same number of episodes (600), the episode
axis is stacked directly. The timestep axis uses interpolation since
episode lengths vary across seeds.

Usage
-----
python per_plot.py --log_dir logs_per
python per_plot.py --log_dir logs_per --save
python per_plot.py --log_dir logs_per --rhos 1 2 4 8
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
COLORS = {
    1:  "#2c7bb6",
    2:  "#f39c12",
    4:  "#27ae60",
    8:  "#e74c3c",
    16: "#8e44ad",
}


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════
def load_rho(log_dir: str, truncation: int, rho: int):
    pattern = os.path.join(log_dir, f"trunc{truncation}_rho{rho}_seed*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        return None
    dfs = [pd.read_csv(f) for f in files]
    ep_counts = [len(d) for d in dfs]
    print(f"  ρ={rho:<2}  seeds={len(dfs):>2}  episodes per seed: "
          f"min={min(ep_counts)} max={max(ep_counts)}")
    return dfs


# ═══════════════════════════════════════════════════════════════
# Direct stack for episode axis (all seeds same length)
# ═══════════════════════════════════════════════════════════════
def stack_episodes(dfs, y_col, window=10):
    """
    Stack return arrays directly — valid when all seeds have same
    number of episodes. Applies a rolling mean smoothing.
    Returns (episodes array, matrix of shape (n_seeds, n_episodes)).
    """
    min_eps = min(len(df) for df in dfs)
    matrix  = np.array([
        pd.Series(df[y_col].values[:min_eps]).rolling(window, min_periods=1).mean().values
        for df in dfs
    ])
    episodes = np.arange(1, min_eps + 1)
    return episodes, matrix


# ═══════════════════════════════════════════════════════════════
# Interpolation for timestep axis (episode lengths vary per seed)
# ═══════════════════════════════════════════════════════════════
def interp_timesteps(dfs, y_col, n_points=500):
    x_min  = max(df["timestep"].min() for df in dfs)
    x_max  = min(df["timestep"].max() for df in dfs)
    x_grid = np.linspace(x_min, x_max, n_points)
    matrix = np.array([
        np.interp(x_grid, df["timestep"].values, df[y_col].values)
        for df in dfs
    ])
    return x_grid, matrix


# ═══════════════════════════════════════════════════════════════
# 95% CI  (t-distribution)
# ═══════════════════════════════════════════════════════════════
def mean_ci(matrix, confidence=0.95):
    n    = matrix.shape[0]
    mean = matrix.mean(axis=0)
    se   = matrix.std(axis=0, ddof=1) / np.sqrt(n)
    t    = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean, t * se


# ═══════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════
def draw(ax, x, mean, ci, color, label):
    ax.plot(x, mean, color=color, linewidth=2.0, label=label)
    ax.fill_between(x, mean - ci, mean + ci, alpha=0.15, color=color)


def style_ax(ax, xlabel, title, k_fmt=False):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Return", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if k_fmt:
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
            )
        )


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def make_plot(log_dir, truncation, rhos, n_points=500,
              smooth=10, save=False, out_dir="plots", per=False):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tag_str   = "PER" if per else "Uniform"
    fig.suptitle(
        f"DQN on MountainCar-v0 — Replay Factor ρ Comparison  "
        f"[{tag_str}, truncation={truncation}]",
        fontsize=14, fontweight="bold",
    )

    print(f"\nLoading from: {os.path.abspath(log_dir)}/")
    any_loaded = False

    for rho in rhos:
        dfs = load_rho(log_dir, truncation, rho)
        if dfs is None:
            print(f"  ρ={rho:<2}  [SKIP — no files found]")
            continue

        color      = COLORS.get(rho, "#555555")
        label      = f"ρ={rho}  (n={len(dfs)})"
        any_loaded = True

        # ── Left panel: return vs episodes (direct, smoothed) ─
        ep_x, ep_mat   = stack_episodes(dfs, "return", window=smooth)
        ep_mean, ep_ci = mean_ci(ep_mat)
        draw(axes[0], ep_x, ep_mean, ep_ci, color, label)

        # ── Right panel: return vs timesteps (interpolated) ───
        ts_x, ts_mat   = interp_timesteps(dfs, "return", n_points)
        ts_mean, ts_ci = mean_ci(ts_mat)
        draw(axes[1], ts_x, ts_mean, ts_ci, color, label)

    if not any_loaded:
        print("\nNo data found! Check --log_dir.")
        return None

    style_ax(axes[0], f"Episodes  (smoothed window={smooth})", "Return vs Episodes")
    style_ax(axes[1], "Timesteps", "Return vs Timesteps", k_fmt=True)

    plt.tight_layout()

    if save:
        os.makedirs(out_dir, exist_ok=True)
        suffix = "per" if per else "uniform"
        fname  = os.path.join(out_dir, f"rho_comparison_{suffix}_trunc{truncation}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved → {fname}")
    else:
        plt.show()

    return fig


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare DQN ρ values")
    p.add_argument("--log_dir",    type=str, default="logs_per")
    p.add_argument("--truncation", type=int, default=2000)
    p.add_argument("--rhos",       type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--n_points",   type=int, default=500,
                   help="Grid points for timestep interpolation")
    p.add_argument("--smooth",     type=int, default=10,
                   help="Rolling window for episode axis smoothing")
    p.add_argument("--save",       action="store_true")
    p.add_argument("--out_dir",    type=str, default="plots")
    p.add_argument("--per",        action="store_true",
                   help="Label the plot as PER experiment")
    args = p.parse_args()

    make_plot(
        log_dir    = args.log_dir,
        truncation = args.truncation,
        rhos       = args.rhos,
        n_points   = args.n_points,
        smooth     = args.smooth,
        save       = args.save,
        out_dir    = args.out_dir,
        per        = args.per,
    )
