"""
Visualise a saved SAC policy on the modified Pendulum environment.

Usage
-----
# Live render
python scripts/visualise_pendulum.py \
    --weights logs/q3_pebble_pendulum/theta_90/sac_gt_theta90_seed0.pt \
    --theta 90

# Headless (no display) — just print returns
python scripts/visualise_pendulum.py \
    --weights logs/.../sac_gt_theta90_seed0.pt \
    --theta 90 --no-render

# Record to MP4
python scripts/visualise_pendulum.py \
    --weights logs/.../sac_gt_theta90_seed0.pt \
    --theta 90 --record --out videos/pendulum_theta90

# Sweep multiple checkpoints to see policy evolution
python scripts/visualise_pendulum.py \
    --weights logs/.../sac_gt_theta90_seed0_step50000.pt \
              logs/.../sac_gt_theta90_seed0_step100000.pt \
    --theta 90 --no-render
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from tqdm import tqdm


OBS_DIM    = 3
ACTION_DIM = 1


def load_agent(weights_path, device):
    from agents.sac import SAC
    agent = SAC(
        obs_dim    = OBS_DIM,
        action_dim = ACTION_DIM,
        hidden     = (256, 256),
        auto_alpha = True,
        device     = device,
    )
    agent.load(weights_path)
    agent.actor.eval()
    return agent


def make_env(theta_deg, render_mode, record_dir=None):
    from envs.pendulum import PendulumTargetEnv
    if record_dir:
        base = PendulumTargetEnv(theta_target_deg=theta_deg)
        # Unwrap to get the raw gym env for RecordVideo
        env = RecordVideo(
            base,
            video_folder     = record_dir,
            episode_trigger  = lambda _: True,
            disable_logger   = True,
        )
    else:
        env = PendulumTargetEnv(theta_target_deg=theta_deg)
        # Patch render_mode on the inner env
        env.env = gym.make("Pendulum-v1", render_mode=render_mode)
    return env


def angle_from_obs(obs):
    """Recover θ in degrees from [cos θ, sin θ, dθ/dt]."""
    cos_th, sin_th, _ = obs
    return np.degrees(np.arctan2(sin_th, cos_th))


def run_episodes(agent, env, theta_deg, n_episodes, verbose=True):
    returns, angle_errors = [], []

    for ep in tqdm(range(n_episodes), desc="Episodes"):
        obs, _ = env.reset(seed=ep)
        done, ep_ret, steps = False, 0.0, 0
        ep_errors = []

        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret  += reward
            steps   += 1
            done     = terminated or truncated

            # Track angle error
            current_angle = angle_from_obs(obs)
            error = abs(current_angle - theta_deg)
            error = min(error, 360 - error)   # wrap to [0, 180]
            ep_errors.append(error)

        returns.append(ep_ret)
        angle_errors.append(np.mean(ep_errors))

        if verbose:
            tqdm.write(
                f"  ep {ep+1:>2d}: return={ep_ret:+8.1f} | "
                f"mean_angle_error={np.mean(ep_errors):5.1f}°  ({steps} steps)"
            )

    return returns, angle_errors


def print_summary(label, returns, angle_errors, theta_deg):
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"  θ_target = {theta_deg}°")
    print(f"  Mean return      : {np.mean(returns):+.1f} ± {np.std(returns):.1f}")
    print(f"  Mean angle error : {np.mean(angle_errors):.2f}°")
    print(f"  Min angle error  : {np.min(angle_errors):.2f}°")
    print(f"{'─'*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualise saved SAC policy on modified Pendulum"
    )
    parser.add_argument(
        "--weights", nargs="+", required=True,
        help="Path(s) to .pt checkpoint file(s). Multiple = sweep comparison."
    )
    parser.add_argument(
        "--theta", type=float, required=True,
        help="Target angle θ_target in degrees (must match training)"
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of episodes per checkpoint (default: 5)"
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Disable rendering (headless / server use)"
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Record episodes to MP4 instead of live render"
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory for recorded videos (default: videos/pendulum_theta<N>)"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device   : {device}")
    print(f"θ_target : {args.theta}°")
    print(f"Weights  : {args.weights}")

    render_mode = None if args.no_render else "human"

    for weights_path in args.weights:
        label = os.path.basename(weights_path)
        print(f"\n{'='*50}")
        print(f"Loading: {label}")

        agent = load_agent(weights_path, device)

        if args.record:
            record_dir = args.out or f"videos/pendulum_theta{int(args.theta)}"
            os.makedirs(record_dir, exist_ok=True)
            env = make_env(args.theta, "rgb_array", record_dir=record_dir)
            print(f"Recording to: {record_dir}/")
        else:
            env = make_env(args.theta, render_mode)

        returns, angle_errors = run_episodes(
            agent, env, args.theta, args.episodes, verbose=True
        )
        env.close()

        print_summary(label, returns, angle_errors, args.theta)


if __name__ == "__main__":
    main()