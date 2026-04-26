# Assignment_3/train_reacher.py

import torch
torch.set_num_threads(1)
import numpy as np
import csv
import os
from tqdm import tqdm

from agent.sac import SACAgent
from utils.replay_buffer import ReplayBuffer
from utils.eval import evaluate
from utils.logger import Logger
from envs.reacher_env import ReacherEnv


def train_reacher(
    reward_type="Ra",          # one of {"Ra", "Rb", "Rc"}
    task="easy",               # "easy" as in the assignment
    seed=0,
    total_steps=500000,        # Increased from 250k to allow Ra to converge
    device=None,
    show_pbar=True,
    learnable_temperature=True,
    init_temperature=0.1,
    exp_name=None,
):
    """
    Train SAC on DeepMind Control Reacher with a chosen reward formulation.

    reward_type: "Ra", "Rb", or "Rc" (defines env reward/termination logic)
    task:        "easy" (required by assignment) or "hard"
    """

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- ENV ---
    env = ReacherEnv(task=task, reward_type=reward_type)
    
    # Per Assignment 2.3.1: Evaluate against ALL THREE reward formulations
    eval_env_Ra = ReacherEnv(task=task, reward_type="Ra")
    eval_env_Rb = ReacherEnv(task=task, reward_type="Rb")
    eval_env_Rc = ReacherEnv(task=task, reward_type="Rc")

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # --- AGENT ---
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
        init_temperature=init_temperature,
        learnable_temperature=learnable_temperature,
        alpha_lr=5e-4,
        alpha_betas=(0.9, 0.999),
        actor_lr=5e-4,
        actor_betas=(0.9, 0.999),
        actor_update_frequency=1,
        critic_lr=5e-4,
        critic_betas=(0.9, 0.999),
        critic_tau=0.005,
        critic_target_update_frequency=2,
        batch_size=256,
    )

    logger = Logger()
    buffer = ReplayBuffer(obs_dim, action_dim, 1_000_000, device)

    obs, _ = env.reset()
    returns = []

    # Naming: SAC-Ra, SAC-Rb, SAC-Rc
    mode = f"{task}_{reward_type}"  # e.g., "easy_Ra"
    if exp_name is None:
        exp_name = f"{mode}_seed{seed}"

    # --- Directories: keep Reacher separate from Pendulum ---
    logs_root = os.path.join("logs", "reacher")
    models_root = os.path.join("models", "reacher")

    os.makedirs(logs_root, exist_ok=True)
    os.makedirs(models_root, exist_ok=True)
    os.makedirs(os.path.join(models_root, mode), exist_ok=True)

    # -------- Evaluation at timestep 0 (random / untrained policy) --------
    eval_ret_Ra = evaluate(agent, eval_env_Ra, episodes=20)
    eval_ret_Rb = evaluate(agent, eval_env_Rb, episodes=20)
    eval_ret_Rc = evaluate(agent, eval_env_Rc, episodes=20)
    returns.append((0, eval_ret_Ra, eval_ret_Rb, eval_ret_Rc))
    
    # Track the "best" return based on the specific formulation we are training on
    best_return = eval_ret_Ra if reward_type == "Ra" else (eval_ret_Rb if reward_type == "Rb" else eval_ret_Rc)
    
    torch.save(
        agent.actor.state_dict(),
        os.path.join(models_root, mode, f"actor_{exp_name}_step0.pth"),
    )

    # ---------------- Main training loop ----------------
    pbar = tqdm(
        range(1, total_steps + 1),
        desc=f"{exp_name}",
        leave=False,
        disable=not show_pbar,
    )

    for step in pbar:

        # Initial random exploration for better coverage
        if step < 10_000:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, sample=True)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.add(obs, action, reward, next_obs, done=done, done_no_max=terminated)
        obs = next_obs

        # Start learning after the initial exploration phase
        if step >= 10_000:
            agent.update(buffer, logger=logger, step=step)

        if done:
            obs, _ = env.reset()

        # Offline evaluation every 10k steps (20 episodes) across all 3 formulations
        if step % 10_000 == 0:
            eval_ret_Ra = evaluate(agent, eval_env_Ra, episodes=20)
            eval_ret_Rb = evaluate(agent, eval_env_Rb, episodes=20)
            eval_ret_Rc = evaluate(agent, eval_env_Rc, episodes=20)
            
            returns.append((step, eval_ret_Ra, eval_ret_Rb, eval_ret_Rc))

            # Check if this is the best model for the current training objective
            current_eval_ret = eval_ret_Ra if reward_type == "Ra" else (eval_ret_Rb if reward_type == "Rb" else eval_ret_Rc)

            if current_eval_ret > best_return:
                best_return = current_eval_ret
                torch.save(
                    agent.actor.state_dict(),
                    os.path.join(models_root, mode, f"actor_{exp_name}_best.pth"),
                )

            # tqdm postfix
            pbar.set_postfix(
                {
                    f"ret_{reward_type}": round(current_eval_ret, 2),
                    "alpha": round(agent.alpha.item(), 3),
                }
            )

    # Save last policy snapshot
    torch.save(
        agent.actor.state_dict(),
        os.path.join(models_root, mode, f"actor_{exp_name}_last.pth"),
    )

    # Save eval curve: step vs mean undiscounted return for all 3 formulations
    log_file = os.path.join(logs_root, f"reacher_{exp_name}.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "return_Ra", "return_Rb", "return_Rc"])
        for step_idx, r_a, r_b, r_c in returns:
            writer.writerow([step_idx, r_a, r_b, r_c])

    return returns