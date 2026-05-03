"""
Visualise a saved policy by rendering episodes in the gymnasium window.

Usage examples
--------------
# Continuous SAC (Q2.2.1)
python scripts/visualise_policy.py --mode continuous --weights logs/q2_2_1_continuous/sac_continuous_seed0.pt

# Hover env, fixed alpha (Q2.2.3)
python scripts/visualise_policy.py --mode hover --weights logs/q2_2_3_hover/fixed_alpha_seed0.pt --hover-bonus 200

# Hover env after reward switch
python scripts/visualise_policy.py --mode hover --weights logs/q2_2_3_hover/fixed_alpha_seed0.pt --hover-bonus -100

# Discrete SAC (Q2.2.4)
python scripts/visualise_policy.py --mode discrete-sac --weights logs/q2_2_4_discrete/discrete_sac_seed0.pt

# DQN (Q2.2.4)
python scripts/visualise_policy.py --mode dqn --weights logs/q2_2_4_discrete/dqn_seed0.pt

Options
-------
--episodes   Number of episodes to render (default: 5)
--no-render  Skip rendering, just print episode returns (useful on headless servers)
--record     Save episodes as an MP4 video instead of rendering live
--out        Output video path (default: videos/<mode>_policy.mp4)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from tqdm import tqdm


# ── Hard-coded dims (no gym needed at import time) ──────────────
OBS_DIM    = 8
ACTION_DIM = 2   # continuous
N_ACTIONS  = 4   # discrete


# ── Agent loaders ────────────────────────────────────────────────
def load_continuous_sac(weights_path, device):
    from agents.sac import SAC
    agent = SAC(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        hidden=(256, 256), auto_alpha=True, device=device,
    )
    agent.load(weights_path)
    agent.actor.eval()
    return agent

def load_discrete_sac(weights_path, device):
    from agents.sac import DiscreteSAC
    agent = DiscreteSAC(
        obs_dim=OBS_DIM, n_actions=N_ACTIONS,
        hidden=(256, 256), auto_alpha=True, device=device,
    )
    agent.load(weights_path)
    agent.actor.eval()
    return agent

def load_dqn(weights_path, device):
    from agents.dqn import DQN
    agent = DQN(
        obs_dim=OBS_DIM, n_actions=N_ACTIONS,
        hidden=(256, 256), device=device,
    )
    agent.load(weights_path)
    agent.qnet.eval()
    return agent


# ── Environment builders ─────────────────────────────────────────
def make_env(mode, render_mode, hover_bonus=200.0, record_dir=None):
    if mode in ("continuous", "hover"):
        if record_dir:
            base = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
            env  = RecordVideo(base, video_folder=record_dir,
                               episode_trigger=lambda _: True, disable_logger=True)
        else:
            env = gym.make("LunarLander-v3", continuous=True, render_mode=render_mode)

        if mode == "hover":
            from envs.lunar_lander import HoverLunarLander
            # Wrap manually — RecordVideo already wraps, so patch bonus on inner env
            env = _HoverWrapper(env, hover_bonus)

    else:  # discrete-sac or dqn
        if record_dir:
            base = gym.make("LunarLander-v3", continuous=False, render_mode="rgb_array")
            env  = RecordVideo(base, video_folder=record_dir,
                               episode_trigger=lambda _: True, disable_logger=True)
        else:
            env = gym.make("LunarLander-v3", continuous=False, render_mode=render_mode)

    return env


class _HoverWrapper(gym.Wrapper):
    """Lightweight hover-bonus wrapper that works on top of any env."""
    def __init__(self, env, hover_bonus):
        super().__init__(env)
        self.hover_bonus   = hover_bonus
        self._given_bonus  = False

    def reset(self, **kwargs):
        self._given_bonus = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x, y = obs[0], obs[1]
        if not self._given_bonus and abs(x) < 0.1 and 0.4 < abs(y) < 0.6:
            reward += self.hover_bonus
            self._given_bonus = True
        return obs, reward, terminated, truncated, info


# ── Run episodes ─────────────────────────────────────────────────
def run_episodes(agent, env, n_episodes, desc="Evaluating"):
    returns = []
    for ep in tqdm(range(n_episodes), desc=desc):
        obs, _ = env.reset(seed=ep)
        done, ep_ret, steps = False, 0.0, 0
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            steps  += 1
            done = terminated or truncated
        returns.append(ep_ret) 
        tqdm.write(f"  Episode {ep+1:>2d}: return = {ep_ret:+.1f}  ({steps} steps)")
    return returns


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Visualise a saved LunarLander policy")
    parser.add_argument("--mode",    required=True,
                        choices=["continuous", "hover", "discrete-sac", "dqn"],
                        help="Which agent/env to visualise")
    parser.add_argument("--weights", required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes (default: 5)")
    parser.add_argument("--hover-bonus", type=float, default=200.0,
                        help="Hover bonus for hover mode (200 or -100)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering (headless servers)")
    parser.add_argument("--record", action="store_true",
                        help="Record episodes to MP4 instead of live render")
    parser.add_argument("--out", default=None,
                        help="Output directory for recorded videos")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Mode   : {args.mode}")
    print(f"Weights: {args.weights}")

    # ── Load agent ───────────────────────────────────────────────
    if args.mode == "continuous":
        agent = load_continuous_sac(args.weights, device)
    elif args.mode == "hover":
        agent = load_continuous_sac(args.weights, device)
    elif args.mode == "discrete-sac":
        agent = load_discrete_sac(args.weights, device)
    elif args.mode == "dqn":
        agent = load_dqn(args.weights, device)

    # ── Build env ────────────────────────────────────────────────
    if args.record:
        record_dir = args.out or f"videos/{args.mode}_policy"
        os.makedirs(record_dir, exist_ok=True)
        render_mode = "rgb_array"
        print(f"Recording to: {record_dir}/")
    elif args.no_render:
        render_mode = None
        record_dir  = None
        print("Rendering disabled.")
    else:
        render_mode = "human"
        record_dir  = None

    env = make_env(args.mode, render_mode,
                   hover_bonus=args.hover_bonus,
                   record_dir=record_dir if args.record else None)

    # ── Run ──────────────────────────────────────────────────────
    returns = run_episodes(agent, env, args.episodes)
    env.close()

    print(f"\n{'─'*40}")
    print(f"Mean return : {np.mean(returns):+.1f}")
    print(f"Std         : {np.std(returns):.1f}")
    print(f"Min / Max   : {min(returns):+.1f} / {max(returns):+.1f}")
    if args.record:
        print(f"Videos saved to: {record_dir}/")


if __name__ == "__main__":
    main()