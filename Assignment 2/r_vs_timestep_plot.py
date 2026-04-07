"""
r_vs_timestep_plot.py
---------------------
Reads all CSV logs for a given experiment config and plots:
  - Mean return vs Episodes
  - Mean return vs Timesteps
with 95% confidence intervals across seeds.

Usage
-----
# After running: python vanilla_dqn_parallel.py --truncation 2000 --replay_factor 1
python r_vs_timestep_plot.py --log_dir logs --truncation 2000 --replay_factor 1

# Save figure instead of showing
python r_vs_timestep_plot.py --log_dir logs --truncation 2000 --replay_factor 1 --save
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
# Data loading
# ═══════════════════════════════════════════════════════════════
def load_seed_csvs(log_dir: str, truncation: int, replay_factor: int):
    """
    Load all CSV files matching trunc{T}_rho{R}_seed*.csv
    Returns a list of DataFrames, one per seed.
    """
    pattern = os.path.join(log_dir, f"trunc{truncation}_rho{replay_factor}_seed*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found for pattern:\n  {pattern}\n"
            f"Make sure you have run the training first."
        )

    print(f"Found {len(files)} seed files:")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  {os.path.basename(f):50s}  episodes={len(df)}")

    return dfs


# ═══════════════════════════════════════════════════════════════
# Interpolation helpers
# ═══════════════════════════════════════════════════════════════
def interpolate_to_grid(dfs: list, x_col: str, y_col: str, n_points: int = 500):
    """
    Interpolate each seed's curve onto a common x-grid, then stack.
    This handles seeds having different numbers of episodes/timesteps.

    Returns
    -------
    x_grid : (n_points,)
    matrix : (n_seeds, n_points)  — each row is one seed's interpolated curve
    """
    # Common x range: min of max-x across seeds (so all seeds have data there)
    x_min = max(df[x_col].min() for df in dfs)
    x_max = min(df[x_col].max() for df in dfs)
    x_grid = np.linspace(x_min, x_max, n_points)

    matrix = []
    for df in dfs:
        y_interp = np.interp(x_grid, df[x_col].values, df[y_col].values)
        matrix.append(y_interp)

    return x_grid, np.array(matrix)   # (n_seeds, n_points)


# ═══════════════════════════════════════════════════════════════
# 95% Confidence Interval
# ═══════════════════════════════════════════════════════════════
def mean_and_ci(matrix: np.ndarray, confidence: float = 0.95):
    """
    Given (n_seeds, n_points), return mean and CI half-width per point.
    Uses t-distribution (correct for small n).
    """
    n    = matrix.shape[0]
    mean = matrix.mean(axis=0)
    se   = matrix.std(axis=0, ddof=1) / np.sqrt(n)
    t    = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    ci   = t * se
    return mean, ci


# ═══════════════════════════════════════════════════════════════
# Single panel plot
# ═══════════════════════════════════════════════════════════════
def plot_panel(ax, x_grid, mean, ci, x_label, title, color="#2c7bb6"):
    ax.plot(x_grid, mean, color=color, linewidth=2.0, label="Mean Return")
    ax.fill_between(
        x_grid,
        mean - ci,
        mean + ci,
        alpha=0.25,
        color=color,
        label="95% Confidence Interval",
    )
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Return", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
    ))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ═══════════════════════════════════════════════════════════════
# Main plot function
# ═══════════════════════════════════════════════════════════════
def make_plot(log_dir, truncation, replay_factor, n_points=500, save=False, out_dir="plots"):

    dfs = load_seed_csvs(log_dir, truncation, replay_factor)
    n_seeds = len(dfs)

    # ── Interpolate onto common grids ─────────────────────────
    ep_grid,  ep_matrix  = interpolate_to_grid(dfs, "episode",   "return", n_points)
    ts_grid,  ts_matrix  = interpolate_to_grid(dfs, "timestep",  "return", n_points)

    ep_mean,  ep_ci  = mean_and_ci(ep_matrix)
    ts_mean,  ts_ci  = mean_and_ci(ts_matrix)

    # ── Figure: two side-by-side panels ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Vanilla DQN on MountainCar-v0  "
        f"(truncation={truncation}, ρ={replay_factor}, {n_seeds} seeds)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plot_panel(
        axes[0], ep_grid, ep_mean, ep_ci,
        x_label="Episodes",
        title="Return vs Episodes",
    )
    plot_panel(
        axes[1], ts_grid, ts_mean, ts_ci,
        x_label="Timesteps",
        title="Return vs Timesteps",
    )

    plt.tight_layout()

    if save:
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"dqn_trunc{truncation}_rho{replay_factor}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved → {fname}")
    else:
        plt.show()

    return fig


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot DQN learning curves")
    p.add_argument("--log_dir",       type=str, default="logs")
    p.add_argument("--truncation",    type=int, default=2000)
    p.add_argument("--replay_factor", type=int, default=1)
    p.add_argument("--n_points",      type=int, default=500,
                   help="Number of points on the interpolation grid")
    p.add_argument("--save",          action="store_true",
                   help="Save figure to --out_dir instead of showing")
    p.add_argument("--out_dir",       type=str, default="plots")
    args = p.parse_args()

    make_plot(
        log_dir       = args.log_dir,
        truncation    = args.truncation,
        replay_factor = args.replay_factor,
        n_points      = args.n_points,
        save          = args.save,
        out_dir       = args.out_dir,
    )
