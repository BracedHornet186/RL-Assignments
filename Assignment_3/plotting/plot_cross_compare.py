# plotting/plot_cross_compare.py

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

LOG_DIR = os.path.join("logs", "reacher")
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

TRAIN_REWARDS = ["Ra", "Rb", "Rc"]
EVAL_REWARDS = ["Ra", "Rb", "Rc"]
COLORS = {"Ra": "tab:blue", "Rb": "tab:orange", "Rc": "tab:green"}


def load_runs(train_reward, eval_reward):
    """
    Load all CSV logs for a given training reward formulation, and pick
    the column corresponding to the evaluation reward.

    Expected filename pattern:
        logs/reacher/reacher_easy_<train_reward>_seed*.csv

    Expected columns (example for Ra-training):
        step, return_Ra, return_Rb, return_Rc
    """
    pattern = os.path.join(LOG_DIR, f"reacher_easy_{train_reward}_seed*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No logs found for pattern {pattern}")

    col_name = f"return_{eval_reward}"

    all_steps = []
    all_returns = []

    for f in files:
        df = pd.read_csv(f)
        df = df.sort_values("step")
        if col_name not in df.columns:
            raise KeyError(f"{col_name} not found in {f}")
        all_steps.append(df["step"].to_numpy())
        all_returns.append(df[col_name].to_numpy())

    steps = all_steps[0]
    returns = np.stack(all_returns, axis=0)  # [num_seeds, num_steps]
    mean = returns.mean(axis=0)
    stderr = returns.std(axis=0) / np.sqrt(returns.shape[0])

    return steps, mean, stderr, len(files)


def plot_for_eval_reward(eval_reward):
    """
    For a fixed evaluation reward R_i, plot curves for agents trained with
    Ra, Rb, Rc, using the corresponding return_Ri column.
    """
    plt.figure(figsize=(7, 5))

    for train_reward in TRAIN_REWARDS:
        try:
            steps, mean_ret, stderr_ret, n_seeds = load_runs(
                train_reward, eval_reward
            )
        except (FileNotFoundError, KeyError) as e:
            print(e)
            continue

        label = f"SAC-{train_reward} (avg {n_seeds} seeds)"
        color = COLORS[train_reward]

        plt.plot(steps, mean_ret, label=label, color=color)
        plt.fill_between(
            steps,
            mean_ret - stderr_ret,
            mean_ret + stderr_ret,
            color=color,
            alpha=0.2,
        )

    plt.xlabel("Environment Timesteps", fontsize=12)
    plt.ylabel(f"Average Return (evaluated as R{eval_reward[-1]})", fontsize=12)
    plt.title(
        f"Q2.3.3(c): Average Return vs Timesteps\nEvaluation Reward = {eval_reward}",
        fontsize=13,
    )
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f"cross_compare_eval_{eval_reward}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    for eval_reward in EVAL_REWARDS:
        plot_for_eval_reward(eval_reward)