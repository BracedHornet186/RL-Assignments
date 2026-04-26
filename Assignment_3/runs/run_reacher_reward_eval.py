import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import multiprocessing as mp

# Ensure imports work regardless of where the script is called from
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.sac import SACAgent
from envs.reacher_env import ReacherEnv


def worker_eval_chunk(args):
    """
    Worker function to evaluate a chunk of episodes.

    args: (reward_type, model_path, start_seed, num_eps, max_steps, device)
    """
    reward_type, model_path, start_seed, num_episodes, max_steps, device = args

    env = ReacherEnv(task="easy", reward_type=reward_type)
    env.max_episode_steps = max_steps

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize agent skeleton
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_range=[-1, 1],
        device=device,
        critic_cfg=dict(
            _target_="agent.critic.DoubleQCritic",
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            hidden_depth=2,
        ),
        actor_cfg=dict(
            _target_="agent.actor.DiagGaussianActor",
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            hidden_depth=2,
            log_std_bounds=[-5, 2],
        ),
        discount=0.99,
        init_temperature=0.1,
        learnable_temperature=True,
        alpha_lr=3e-4,
        alpha_betas=(0.9, 0.999),
        actor_lr=3e-4,
        actor_betas=(0.9, 0.999),
        actor_update_frequency=1,
        critic_lr=3e-4,
        critic_betas=(0.9, 0.999),
        critic_tau=0.005,
        critic_target_update_frequency=2,
        batch_size=256,
    )

    # Load the best weights
    agent.actor.load_state_dict(
        torch.load(model_path, map_location=torch.device(device))
    )
    agent.actor.eval()

    steps_to_goal_list = []
    steps_in_target_list = []

    for ep in range(num_episodes):
        seed = start_seed + ep
        obs, _ = env.reset(seed=seed)

        steps_taken = 0
        hit_target_yet = False
        steps_to_first_hit = max_steps  # default if never hits
        total_steps_inside = 0

        env._step_count = 0

        for _ in range(max_steps):
            with torch.no_grad():
                action = agent.act(obs, sample=False)

            obs, reward, terminated, truncated, info = env.step(action)
            steps_taken += 1

            in_target = info.get("in_target", False)

            if in_target:
                total_steps_inside += 1
                if not hit_target_yet:
                    hit_target_yet = True
                    steps_to_first_hit = steps_taken

            if terminated or truncated or steps_taken >= max_steps:
                break

        steps_to_goal_list.append(steps_to_first_hit)
        steps_in_target_list.append(total_steps_inside)

    return steps_to_goal_list, steps_in_target_list


def run_evaluation_parallel(
    reward_type,
    model_path,
    seed=0,
    num_episodes=500,
    max_steps=5000,
    device="cpu",
    num_workers=8,
):
    """
    Parallel evaluation over num_episodes using num_workers processes,
    with a tqdm progress bar over total episodes.
    """
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return [], []

    print(
        f"Evaluating SAC-{reward_type} on {num_episodes} episodes "
        f"of length {max_steps} using {num_workers} workers..."
    )

    # Split episodes across workers
    eps_per_worker = num_episodes // num_workers
    remainder = num_episodes % num_workers

    pool_args = []
    current_seed = seed

    for i in range(num_workers):
        n_eps = eps_per_worker + (1 if i < remainder else 0)
        if n_eps == 0:
            continue
        pool_args.append(
            (reward_type, model_path, current_seed, n_eps, max_steps, device)
        )
        current_seed += n_eps

    all_steps_to_goal = []
    all_steps_in_target = []

    with mp.Pool(processes=num_workers) as pool:
        with tqdm(
            total=num_episodes, desc=f"Eval {reward_type}", leave=False
        ) as pbar:
            # imap_unordered yields results as workers finish
            for steps_to_goal_list, steps_in_target_list in pool.imap_unordered(
                worker_eval_chunk, pool_args
            ):
                all_steps_to_goal.extend(steps_to_goal_list)
                all_steps_in_target.extend(steps_in_target_list)
                pbar.update(len(steps_to_goal_list))

    return all_steps_to_goal, all_steps_in_target


def plot_results(results_dict, output_dir="plots"):
    """
    Plots the bar charts with confidence intervals for both metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    formulations = list(results_dict.keys())

    # --- Metric 1: Steps to Goal ---
    means_goal = [np.mean(results_dict[f]["goal"]) for f in formulations]
    stds_goal = [np.std(results_dict[f]["goal"]) for f in formulations]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        formulations,
        means_goal,
        yerr=stds_goal,
        capsize=10,
        color=["blue", "orange", "green"],
        alpha=0.7,
        edgecolor="black",
    )
    plt.title("Q2.3.3(a): Steps to Reach Goal (Lower is Better)", fontsize=14)
    plt.ylabel("Timesteps to First Hit", fontsize=12)
    plt.xlabel("Reward Formulation Trained On", fontsize=12)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 10,
            f"{yval:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eval_steps_to_goal.png"), dpi=300)
    plt.show()
    plt.close()

    # --- Metric 2: Steps In Target ---
    means_in = [np.mean(results_dict[f]["inside"]) for f in formulations]
    stds_in = [np.std(results_dict[f]["inside"]) for f in formulations]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        formulations,
        means_in,
        yerr=stds_in,
        capsize=10,
        color=["blue", "orange", "green"],
        alpha=0.7,
        edgecolor="black",
    )
    plt.title(
        "Q2.3.3(a): Total Steps Spent Inside Target (Higher is Better)", fontsize=14
    )
    plt.ylabel("Timesteps Hovering Inside Target", fontsize=12)
    plt.xlabel("Reward Formulation Trained On", fontsize=12)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 10,
            f"{yval:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eval_steps_in_target.png"), dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    base_model_dir = "models/reacher"

    model_paths = {
        "Ra": os.path.join(base_model_dir, "easy_Ra", "actor_easy_Ra_seed0_best.pth"),
        "Rb": os.path.join(base_model_dir, "easy_Rb", "actor_easy_Rb_seed0_best.pth"),
        "Rc": os.path.join(base_model_dir, "easy_Rc", "actor_easy_Rc_seed0_best.pth"),
    }

    results = {}

    # Adjust paths if script is run from inside runs/ folder
    if not os.path.exists("models") and os.path.exists("../models"):
        base_model_dir = "../models/reacher"
        model_paths = {
            "Ra": os.path.join(
                base_model_dir, "easy_Ra", "actor_easy_Ra_seed0_best.pth"
            ),
            "Rb": os.path.join(
                base_model_dir, "easy_Rb", "actor_easy_Rb_seed0_best.pth"
            ),
            "Rc": os.path.join(
                base_model_dir, "easy_Rc", "actor_easy_Rc_seed0_best.pth"
            ),
        }
        out_dir = "../plots"
    else:
        out_dir = "plots"

    for r_type, path in model_paths.items():
        if os.path.exists(path):
            goals, insides = run_evaluation_parallel(
                reward_type=r_type,
                model_path=path,
                seed=42,
                num_episodes=500,
                max_steps=5000,
                device="cpu",
                num_workers=8,  # parallelization here
            )
            results[r_type] = {"goal": goals, "inside": insides}
        else:
            print(f"Skipping {r_type}: Model not found at {path}")

    if len(results) > 0:
        plot_results(results, output_dir=out_dir)
        print("Evaluation complete and plots saved successfully!")
    else:
        print("No models were evaluated. Please check the paths in the script.")