"""
plot_results.py
Generates all required plots for Section 2.3:

  Figure A : Q2.3.2  — Three separate plots: SAC-Ri vs Ri  (self-evaluation)
  Figure B : Q2.3.3a — Bar chart: steps_to_goal & steps_in_target
  Figure C : Q2.3.3c — Three plots: SAC-Ra/Rb/Rc vs each reward formulation

Run AFTER all experiments have completed:
    python plot_results.py --result_dir results
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REWARD_TYPES  = ["ra", "rb", "rc"]
REWARD_LABELS = {"ra": r"$\mathcal{R}_a$",
                 "rb": r"$\mathcal{R}_b$",
                 "rc": r"$\mathcal{R}_c$"}
AGENT_COLORS  = {"ra": "#E74C3C", "rb": "#2980B9", "rc": "#27AE60"}
SEEDS         = list(range(15))


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_results(result_dir):
    """
    Returns nested dict:
      data[reward_type][seed] = log_dict
    """
    data = {r: {} for r in REWARD_TYPES}
    for rtype in REWARD_TYPES:
        for seed in SEEDS:
            fpath = os.path.join(result_dir,
                                 f"sac_r{rtype}_seed{seed}.json")
            if os.path.exists(fpath):
                with open(fpath) as f:
                    data[rtype][seed] = json.load(f)
            else:
                print(f"  [WARNING] Missing: {fpath}")
    return data


def aggregate(data, rtype, eval_key):
    """
    Aggregate a metric across all seeds for a given agent type.
    Returns (timesteps, means, stds).
    """
    all_vals  = []
    timesteps = None
    for seed, log in data[rtype].items():
        if eval_key in log and log[eval_key]:
            all_vals.append(log[eval_key])
            if timesteps is None:
                timesteps = log["timesteps"]
    if not all_vals:
        return None, None, None
    arr  = np.array(all_vals)              # shape: (n_seeds, n_evals)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    return np.array(timesteps), mean, std


# ─────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────

def plot_curve(ax, ts, mean, std, label, color, alpha_fill=0.2):
    ax.plot(ts, mean, label=label, color=color, linewidth=2)
    ax.fill_between(ts, mean - std, mean + std,
                    color=color, alpha=alpha_fill)


def save_fig(fig, path, dpi=150):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────
# FIGURE A  (Q2.3.2): SAC-Ri vs Ri
# ─────────────────────────────────────────────

def plot_self_eval(data, out_dir):
    """
    Three separate figures: one for each (agent=Ri, eval=Ri).
    """
    for rtype in REWARD_TYPES:
        fig, ax = plt.subplots(figsize=(8, 5))
        eval_key = f"eval_{rtype}"
        ts, mean, std = aggregate(data, rtype, eval_key)
        if ts is None:
            print(f"  No data for SAC-R{rtype.upper()} vs R{rtype.upper()}, skipping.")
            continue

        plot_curve(ax, ts, mean, std,
                   label=f"SAC-{REWARD_LABELS[rtype]}",
                   color=AGENT_COLORS[rtype])

        ax.set_xlabel("Environment Timesteps", fontsize=13)
        ax.set_ylabel(f"Avg Undiscounted Return ({REWARD_LABELS[rtype]})",
                      fontsize=13)
        ax.set_title(f"SAC-{REWARD_LABELS[rtype]} trained & evaluated on "
                     f"{REWARD_LABELS[rtype]}", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        save_fig(fig, os.path.join(out_dir,
                                   f"fig_A_self_eval_r{rtype}.png"))


# ─────────────────────────────────────────────
# FIGURE B  (Q2.3.3a): Bar chart
# ─────────────────────────────────────────────

def plot_bar_chart(data, out_dir):
    """
    Two bar charts side-by-side:
      Left:  steps_to_goal for Ra, Rb, Rc
      Right: steps_in_target for Ra, Rb, Rc
    """
    means_stg, stds_stg   = [], []
    means_sit, stds_sit   = [], []
    labels                 = []

    for rtype in REWARD_TYPES:
        all_stg, all_sit = [], []
        for seed, log in data[rtype].items():
            if "final_steps_to_goal" in log:
                all_stg.extend(log["final_steps_to_goal"])
                all_sit.extend(log["final_steps_in_target"])
        if all_stg:
            means_stg.append(np.mean(all_stg))
            stds_stg.append(np.std(all_stg) / np.sqrt(len(all_stg)))  # SE
            means_sit.append(np.mean(all_sit))
            stds_sit.append(np.std(all_sit) / np.sqrt(len(all_sit)))
            labels.append(REWARD_LABELS[rtype])

    if not labels:
        print("  No final evaluation data found, skipping bar chart.")
        return

    x    = np.arange(len(labels))
    cmap = [AGENT_COLORS[r] for r in REWARD_TYPES if
            REWARD_LABELS[r] in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x, means_stg, yerr=stds_stg, color=cmap,
            capsize=6, edgecolor="black", linewidth=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=12)
    ax1.set_ylabel("Steps to Goal (mean ± SE)", fontsize=12)
    ax1.set_title("Steps to First Reach Target\n(lower is better)", fontsize=13)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    ax2.bar(x, means_sit, yerr=stds_sit, color=cmap,
            capsize=6, edgecolor="black", linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=12)
    ax2.set_ylabel("Steps in Target (mean ± SE)", fontsize=12)
    ax2.set_title("Steps Staying Inside Target\n(higher is better)", fontsize=13)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Final Policy Evaluation: 500 Episodes × 5000 Steps",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "fig_B_bar_chart.png"))


# ─────────────────────────────────────────────
# FIGURE C  (Q2.3.3c): Cross-evaluation
# ─────────────────────────────────────────────

def plot_cross_eval(data, out_dir):
    """
    Three figures. Figure i: agent SAC-Ra/Rb/Rc evaluated on Ri.
    Each figure has three curves (one per agent).
    """
    for eval_rtype in REWARD_TYPES:
        fig, ax = plt.subplots(figsize=(9, 5))
        eval_key = f"eval_{eval_rtype}"

        for agent_rtype in REWARD_TYPES:
            ts, mean, std = aggregate(data, agent_rtype, eval_key)
            if ts is None:
                continue
            plot_curve(ax, ts, mean, std,
                       label=f"SAC-{REWARD_LABELS[agent_rtype]}",
                       color=AGENT_COLORS[agent_rtype])

        ax.set_xlabel("Environment Timesteps", fontsize=13)
        ax.set_ylabel(f"Avg Undiscounted Return ({REWARD_LABELS[eval_rtype]})",
                      fontsize=13)
        ax.set_title(f"All agents evaluated on {REWARD_LABELS[eval_rtype]}",
                     fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        save_fig(fig, os.path.join(out_dir,
                                   f"fig_C_cross_eval_against_r{eval_rtype}.png"))


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--plot_dir",   type=str, default="plots")
    args = parser.parse_args()

    print(f"\nLoading results from '{args.result_dir}'...")
    data = load_results(args.result_dir)

    print("\nGenerating Figure A (Q2.3.2: self-evaluation)...")
    plot_self_eval(data, args.plot_dir)

    print("\nGenerating Figure B (Q2.3.3a: bar chart)...")
    plot_bar_chart(data, args.plot_dir)

    print("\nGenerating Figure C (Q2.3.3c: cross-evaluation)...")
    plot_cross_eval(data, args.plot_dir)

    print("\nAll plots saved!")
