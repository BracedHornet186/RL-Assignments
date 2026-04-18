"""
plot_aggregate.py
-----------------
Generates two plots from the assignment:

  (b) Comparison Plot 2 — aggregate performance per ρ variant,
      plotted as mean ± 95% CI with a connecting line.

  (c) Sensitivity Plot — aggregate performance vs hyperparameter value
      for ρ=1 and ρ=4, two lines on the same axes.

Aggregate performance = AUC of the return curve, scaled by number of
episodes (i.e. mean episode return across the run).

Usage
-----
# Comparison plot 2 (rho variants)
python plot_aggregate.py --mode comparison --log_dir logs_per --save

# Sensitivity: batch size
python plot_aggregate.py --mode sensitivity --param batch_size \
    --log_dir logs_sensitivity --save

# Sensitivity: target update freq
python plot_aggregate.py --mode sensitivity --param target_update \
    --log_dir logs_sensitivity --save
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════
COLORS = {
    1: "#2c7bb6",   # blue
    4: "#e74c3c",   # red
    2: "#f39c12",
    8: "#27ae60",
}

# Expected CSV naming:
#   Comparison : trunc2000_rho{R}_seed{S}.csv
#   Sensitivity: trunc2000_rho{R}_{param}{V}_seed{S}.csv
#   e.g.        trunc2000_rho1_batch32_seed0.csv


# ═══════════════════════════════════════════════════════════════
# Core: compute aggregate performance for one seed
# AUC scaled by n_episodes = mean episode return over the run
# ═══════════════════════════════════════════════════════════════
def aggregate(df: pd.DataFrame) -> float:
    """Area under return curve / num episodes = mean return per run."""
    returns = df["return"].values
    return float(np.mean(returns))


def load_aggregates(log_dir: str, pattern: str):
    """
    Load all CSVs matching pattern, compute aggregate per seed.
    Returns array of shape (n_seeds,).
    """
    files = sorted(glob.glob(os.path.join(log_dir, pattern)))
    if not files:
        return None
    values = np.array([aggregate(pd.read_csv(f)) for f in files])
    return values


# ═══════════════════════════════════════════════════════════════
# 95% CI using t-distribution
# ═══════════════════════════════════════════════════════════════
def ci95(values: np.ndarray):
    n    = len(values)
    mean = values.mean()
    se   = values.std(ddof=1) / np.sqrt(n)
    t    = stats.t.ppf(0.975, df=n - 1)
    return mean, t * se


# ═══════════════════════════════════════════════════════════════
# (b) Comparison Plot 2
#     x-axis: variant labels (ρ=1, ρ=2, ρ=4, ρ=8)
#     y-axis: aggregate performance
# ═══════════════════════════════════════════════════════════════
def plot_comparison(log_dir, truncation, rhos, save, out_dir):

    means, cis, labels, colors = [], [], [], []

    print(f"\nComparison Plot — loading from {log_dir}/")
    for rho in rhos:
        pattern = f"trunc{truncation}_rho{rho}_seed*.csv"
        vals    = load_aggregates(log_dir, pattern)
        if vals is None:
            print(f"  ρ={rho}  [SKIP]")
            continue
        m, c = ci95(vals)
        means.append(m)
        cis.append(c)
        labels.append(f"ρ = {rho}")
        colors.append(COLORS.get(rho, "#555555"))
        print(f"  ρ={rho:<2}  n={len(vals)}  mean={m:.1f}  CI=±{c:.1f}")

    if not means:
        print("No data found.")
        return

    x    = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 5))

    # Connecting line (grey, behind points)
    ax.plot(x, means, color="grey", linewidth=1.2, zorder=1)

    # Error bars + markers per variant
    for i, (m, c, col, lbl) in enumerate(zip(means, cis, colors, labels)):
        ax.errorbar(
            x[i], m, yerr=c,
            fmt="o", color=col,
            markersize=9,
            capsize=6, capthick=1.8, elinewidth=1.8,
            zorder=2, label=lbl,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Variant", fontsize=12)
    ax.set_ylabel("Aggregate performance", fontsize=12)
    ax.set_title("Comparison plot 2 (95% confidence intervals)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save_or_show(fig, save, out_dir, "comparison_plot2.png")


# ═══════════════════════════════════════════════════════════════
# (c) Sensitivity Plot
#     x-axis: hyperparameter values (log scale)
#     y-axis: aggregate performance
#     two lines: ρ=1 and ρ=4
#
# CSV naming convention:
#   trunc2000_rho{R}_{param_tag}{V}_seed{S}.csv
#   e.g. trunc2000_rho1_batch16_seed0.csv
#        trunc2000_rho4_target100_seed0.csv
# ═══════════════════════════════════════════════════════════════
def plot_sensitivity(log_dir, truncation, param, param_values,
                     rhos, save, out_dir):
    """
    param        : short tag used in filename, e.g. "batch" or "target"
    param_values : list of numeric values, e.g. [16, 32, 64, 128, 256]
    rhos         : list of ρ values to plot, default [1, 4]
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    print(f"\nSensitivity Plot — param={param}  log_dir={log_dir}/")

    for rho in rhos:
        means, cis = [], []
        valid_vals  = []

        for v in param_values:
            pattern = f"trunc{truncation}_rho{rho}_{param}{v}_seed*.csv"
            vals    = load_aggregates(log_dir, pattern)
            if vals is None:
                print(f"  ρ={rho}  {param}={v}  [SKIP]")
                continue
            m, c = ci95(vals)
            means.append(m)
            cis.append(c)
            valid_vals.append(v)
            print(f"  ρ={rho}  {param}={v:<6}  n={len(vals)}  "
                  f"mean={m:.1f}  CI=±{c:.1f}")

        if not means:
            continue

        color  = COLORS.get(rho, "#555555")
        x_plot = np.arange(len(valid_vals))

        # Connecting line
        ax.plot(x_plot, means, color=color, linewidth=1.8,
                label=f"ρ = {rho}", zorder=1)

        # Error bars
        ax.errorbar(
            x_plot, means, yerr=cis,
            fmt="o", color=color,
            markersize=7,
            capsize=5, capthick=1.5, elinewidth=1.5,
            zorder=2,
        )

    ax.set_xticks(np.arange(len(param_values)))
    ax.set_xticklabels([str(v) for v in param_values], fontsize=10)
    ax.set_xlabel("Hyperparameter values", fontsize=12)
    ax.set_ylabel("Aggregate performance", fontsize=12)
    ax.set_title("Sensitivity plot (95% confidence intervals)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save_or_show(fig, save, out_dir, f"sensitivity_{param}.png")


# ═══════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════
def _save_or_show(fig, save, out_dir, fname):
    if save:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved → {path}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode",        type=str, required=True,
                   choices=["comparison", "sensitivity"],
                   help="Which plot to generate")
    p.add_argument("--log_dir",     type=str, default="logs")
    p.add_argument("--truncation",  type=int, default=2000)
    p.add_argument("--save",        action="store_true")
    p.add_argument("--out_dir",     type=str, default="plots")

    # Comparison plot args
    p.add_argument("--rhos",        type=int, nargs="+", default=[1, 2, 4, 8])

    # Sensitivity plot args
    p.add_argument("--param",       type=str, default="batch",
                   help="Param tag used in CSV filename, e.g. 'batch' or 'target'")
    p.add_argument("--param_values",type=float, nargs="+",
                   default=[16, 32, 64, 128, 256],
                   help="Hyperparameter values to sweep over")
    p.add_argument("--sens_rhos",   type=int, nargs="+", default=[1, 4],
                   help="ρ values to compare in sensitivity plot")
    args = p.parse_args()

    if args.mode == "comparison":
        plot_comparison(
            log_dir    = args.log_dir,
            truncation = args.truncation,
            rhos       = args.rhos,
            save       = args.save,
            out_dir    = args.out_dir,
        )
    else:
        plot_sensitivity(
            log_dir      = args.log_dir,
            truncation   = args.truncation,
            param        = args.param,
            param_values = [int(v) if v == int(v) else v for v in args.param_values],
            rhos         = args.sens_rhos,
            save         = args.save,
            out_dir      = args.out_dir,
        )