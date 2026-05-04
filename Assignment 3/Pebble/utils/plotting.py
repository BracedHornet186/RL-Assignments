"""
Plotting helpers for RL training curves.
All plots use environment timesteps on x-axis (never episodes).
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_aggregated(log_dir, run_prefix):
    path = os.path.join(log_dir, f"{run_prefix}_aggregated.json")
    with open(path) as f:
        d = json.load(f)
    return np.array(d["timesteps"]), np.array(d["mean"]), np.array(d["std"])


def plot_curves(
    curves,          # list of dict: {label, timesteps, mean, std, color(opt)}
    title="",
    xlabel="Environment Timesteps",
    ylabel="Average Undiscounted Return",
    save_path=None,
    figsize=(8, 5),
    ci_alpha=0.15,
):
    fig, ax = plt.subplots(figsize=figsize)
    for i, c in enumerate(curves):
        color = c.get("color", COLORS[i % len(COLORS)])
        ts    = np.array(c["timesteps"])
        mean  = np.array(c["mean"])
        std   = np.array(c["std"])
        ax.plot(ts, mean, label=c["label"], color=color, linewidth=2)
        ax.fill_between(ts, mean - std, mean + std, alpha=ci_alpha, color=color)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    return fig, ax


def plot_bar_chart(
    categories,       # list of str (x-tick labels)
    values,           # list of float (bar heights)
    errors,           # list of float (CI)
    ylabel="",
    title="",
    save_path=None,
    figsize=(7, 4),
    colors=None,
):
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(categories))
    bars = ax.bar(x, values, yerr=errors, capsize=5,
                  color=colors or COLORS[:len(categories)],
                  alpha=0.85, error_kw={"linewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    return fig, ax
