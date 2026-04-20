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
from envs.pendulum_target import TargetPendulum


def train(theta, seed=0, total_steps=100000, device=None, workers=None, show_pbar=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- ENV ---
    env = TargetPendulum(theta)
    eval_env = TargetPendulum(theta)   # separate eval env

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # --- AGENT ---
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        # IMPORTANT: Keep this at [-1, 1] to match the Tanh output.
        # The environment wrapper handles scaling it to [-2, 2] for the physics engine.
        action_range=[-1, 1], 
        device=device,
        critic_cfg=dict(_target_="agent.critic.DoubleQCritic",
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        hidden_dim=64,
                        hidden_depth=2),
        actor_cfg=dict(_target_="agent.actor.DiagGaussianActor",
                       obs_dim=obs_dim,
                       action_dim=action_dim,
                       hidden_dim=64,
                       hidden_depth=2,
                       log_std_bounds=[-5, 2]),
        discount=0.99, # Reverted to standard 0.99 
        init_temperature=0.1,
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
        learnable_temperature=True
    )

    logger = Logger()
    buffer = ReplayBuffer(obs_dim, action_dim, 100000, device)

    obs, _ = env.reset()
    returns = []
    
    # Track the best return for model saving
    best_return = -float('inf')

    # Create directories for logs and models
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    pbar = tqdm(
        range(1, total_steps + 1), 
        desc=f"θ={theta}, seed={seed}", 
        leave=False, 
        disable=not show_pbar
    )
    
    for step in pbar:

        # --- 10K random exploration --- (As mandated by the assignment)
        if step < 10000:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, sample=True)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # CRITICAL FIX: Scaled reward by 0.1 to prevent Q-value explosion and entropy collapse!
        buffer.add(obs, action, reward , next_obs, done=done, done_no_max=terminated)
        obs = next_obs

        # --- update ---
        if step >= 10000: # Network updates also start after the 10k exploration phase
            agent.update(buffer, logger=logger, step=step)

        if done:
            obs, _ = env.reset()

        # --- evaluation every 10K ---
        if step % 10000 == 0:
            eval_ret = evaluate(agent, eval_env, episodes=20)
            returns.append((step, eval_ret))

            # --- SAVE BEST MODEL ---
            if eval_ret > best_return:
                best_return = eval_ret
                # Save just the actor state dict (policy) since that's all we need for inference
                torch.save(agent.actor.state_dict(), f"models/actor_theta{theta}_seed{seed}_best.pth")

            pbar.set_postfix({
                "return": round(eval_ret, 2),
                "alpha": round(agent.alpha.item(), 3)
            })

    # --- SAVE LAST MODEL ---
    torch.save(agent.actor.state_dict(), f"models/actor_theta{theta}_seed{seed}_last.pth")

    # --- SAVE CSV ---
    filename = f"logs/pendulum_theta{theta}_seed{seed}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "return"])
        for s, r in returns:
            writer.writerow([s, r])

    return returns