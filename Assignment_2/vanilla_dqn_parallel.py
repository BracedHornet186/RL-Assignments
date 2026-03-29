"""
Vanilla DQN for MountainCar-v0  —  Parallel seed runner
DA6400 Programming Assignment 2

Parallelism strategy
--------------------
Each seed runs as an independent subprocess (torch.multiprocessing).
GPU assignment: seeds are round-robin distributed across available GPUs.
  - 1 GPU  → all workers share it (CUDA is fine with concurrent contexts)
  - N GPUs → worker i uses GPU i % N
  - No GPU → all workers use CPU

Each worker is fully independent: own env, own networks, own replay buffer.
No shared memory / gradients between workers (correct for seed-level parallelism).

Usage
-----
# Run 15 seeds, 8 at a time (auto-detects GPUs)
python dqn_mountaincar.py --truncation 2000 --replay_factor 1 --workers 8

# Force CPU-only
python dqn_mountaincar.py --device cpu --workers 8

# Single seed debug (no subprocess overhead)
python dqn_mountaincar.py --seeds 0 --workers 1
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import random
import argparse
import os
import csv
import time
import queue as queue_module
from collections import deque
from typing import List
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════
# Q-Network
# ═══════════════════════════════════════════════════════════════
class QNetwork(nn.Module):
    """
    2 → 128 → 128 → 3  MLP with ReLU activations.
    Kaiming (He) uniform init on every Linear layer.
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self._kaiming_init()

    def _kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════
# Replay Buffer
# ═══════════════════════════════════════════════════════════════
class ReplayBuffer:
    """Fixed-size circular buffer; uniform random sampling."""

    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(s2, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# ═══════════════════════════════════════════════════════════════
# DQN Agent
# ═══════════════════════════════════════════════════════════════
class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float               = 1e-3,
        gamma: float            = 0.99,
        buffer_size: int        = 50_000,
        batch_size: int         = 64,
        target_update_freq: int = 500,
        eps_start: float        = 1.0,
        eps_end: float          = 0.05,
        eps_decay_steps: int    = 50_000,
        replay_factor: int      = 1,
        hidden: int             = 128,
        device: torch.device    = torch.device("cpu"),
    ):
        self.n_actions          = n_actions
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.eps_start          = eps_start
        self.eps_end            = eps_end
        self.eps_decay_steps    = eps_decay_steps
        self.replay_factor      = replay_factor
        self.device             = device
        self.total_steps        = 0

        self.q_net      = QNetwork(obs_dim, n_actions, hidden).to(device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)
        self.loss_fn   = nn.MSELoss()

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.q_net(s).argmax(1).item())

    def _gradient_step(self):
        if len(self.buffer) < self.batch_size:
            return
        s, a, r, s2, d = self.buffer.sample(self.batch_size)

        S  = torch.tensor(s,  device=self.device)
        A  = torch.tensor(a,  device=self.device).unsqueeze(1)
        R  = torch.tensor(r,  device=self.device)
        S2 = torch.tensor(s2, device=self.device)
        D  = torch.tensor(d,  device=self.device)

        q_pred = self.q_net(S).gather(1, A).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(S2).max(1).values
            q_tgt  = R + self.gamma * q_next * (1.0 - D)

        loss = self.loss_fn(q_pred, q_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def observe(self, s, a, r, s2, terminated):
        """Store transition, do ρ gradient steps, maybe update target."""
        self.buffer.push(s, a, r, s2, float(terminated))
        self.total_steps += 1
        for _ in range(self.replay_factor):
            self._gradient_step()
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


# ═══════════════════════════════════════════════════════════════
# Worker function  (one per seed, runs in its own process)
# ═══════════════════════════════════════════════════════════════
def _train_worker(cfg: dict, seed: int, device_str: str, result_queue):
    """Train one DQN run. Puts a summary dict into result_queue when done."""

    # ── Reproducibility ───────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in device_str:
        torch.cuda.manual_seed(seed)

    device = torch.device(device_str)

    # ── Environment ───────────────────────────────────────────
    env = gym.make("MountainCar-v0", max_episode_steps=cfg["truncation"])
    obs, _ = env.reset(seed=seed)
    obs_dim   = env.observation_space.shape[0]   # 2
    n_actions = env.action_space.n               # 3

    # ── Agent ─────────────────────────────────────────────────
    agent = DQNAgent(
        obs_dim=obs_dim, n_actions=n_actions,
        lr=cfg["lr"], gamma=cfg["gamma"],
        buffer_size=cfg["buffer_size"], batch_size=cfg["batch_size"],
        target_update_freq=cfg["target_update"],
        eps_start=cfg["eps_start"], eps_end=cfg["eps_end"],
        eps_decay_steps=cfg["eps_decay"],
        replay_factor=cfg["replay_factor"],
        hidden=cfg["hidden"], device=device,
    )

    ep_returns, ep_timesteps, ep_lengths, ep_epsilons = [], [], [], []
    ep_ret    = 0.0
    ep_len    = 0
    t0 = time.time()

    # ── Weight save paths ─────────────────────────────────────
    os.makedirs(cfg["log_dir"], exist_ok=True)
    tag       = (cfg.get("run_tag") or
                 f"trunc{cfg['truncation']}_rho{cfg['replay_factor']}_seed{seed}")
    ckpt_dir  = os.path.join(cfg["log_dir"], "weights")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, f"{tag}_best.pt")
    final_ckpt= os.path.join(ckpt_dir, f"{tag}_final.pt")

    best_avg20 = -float("inf")

    with tqdm(
        range(1, cfg["total_timesteps"] + 1),
        desc=f"seed {seed:>2} [{device_str}]",
        unit="ts",
        dynamic_ncols=True,
        leave=True,
    ) as pbar:
        for t in pbar:
            action                                     = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Only pass `terminated` (goal reached) as done-flag,
            # NOT `truncated` — so bootstrapping is not cut off on timeout.
            agent.observe(obs, action, reward, next_obs, terminated)

            ep_ret += reward
            ep_len += 1
            obs     = next_obs

            if done:
                ep_returns.append(ep_ret)
                ep_timesteps.append(t)
                ep_lengths.append(ep_len)
                ep_epsilons.append(agent.epsilon)
                obs, _ = env.reset()
                ep_ret = 0.0
                ep_len = 0

                # ── Save best weights (based on rolling avg20) ─
                if len(ep_returns) >= 20:
                    avg20 = float(np.mean(ep_returns[-20:]))
                    if avg20 > best_avg20:
                        best_avg20 = avg20
                        torch.save({
                            "q_net"       : agent.q_net.state_dict(),
                            "target_net"  : agent.target_net.state_dict(),
                            "optimizer"   : agent.optimizer.state_dict(),
                            "total_steps" : agent.total_steps,
                            "episode"     : len(ep_returns),
                            "avg20"       : avg20,
                            "seed"        : seed,
                        }, best_ckpt)

                pbar.set_postfix(
                    ep    = len(ep_returns),
                    ret   = f"{ep_returns[-1]:.0f}",
                    avg20 = f"{np.mean(ep_returns[-20:]):.0f}",
                    eps   = f"{agent.epsilon:.2f}",
                )

    env.close()

    # ── Save final weights ────────────────────────────────────
    torch.save({
        "q_net"       : agent.q_net.state_dict(),
        "target_net"  : agent.target_net.state_dict(),
        "optimizer"   : agent.optimizer.state_dict(),
        "total_steps" : agent.total_steps,
        "episode"     : len(ep_returns),
        "seed"        : seed,
    }, final_ckpt)

    # ── Save CSV ──────────────────────────────────────────────
    csv_path = os.path.join(cfg["log_dir"], f"{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "timestep", "return", "ep_length", "epsilon"])
        for i, (ret, ts, eplen, eps) in enumerate(
            zip(ep_returns, ep_timesteps, ep_lengths, ep_epsilons)
        ):
            w.writerow([i + 1, ts, ret, eplen, eps])

    summary = dict(
        seed       = seed,
        episodes   = len(ep_returns),
        best       = max(ep_returns) if ep_returns else float("nan"),
        last20     = float(np.mean(ep_returns[-20:])) if len(ep_returns) >= 20 else float("nan"),
        elapsed    = time.time() - t0,
        best_ckpt  = best_ckpt,
        final_ckpt = final_ckpt,
    )
    result_queue.put(summary)


# ═══════════════════════════════════════════════════════════════
# Parallel launcher
# ═══════════════════════════════════════════════════════════════
def run_parallel(cfg: dict, seeds: List[int], max_workers: int):
    """
    Spawn up to `max_workers` subprocesses simultaneously.
    GPU assignment: worker i → cuda:(i % num_gpus)  or cpu.
    """
    num_gpus = torch.cuda.device_count()

    def _device_for(idx: int) -> str:
        # If user forced a device, use it for every worker
        forced = cfg.get("forced_device")
        if forced:
            return forced
        if num_gpus > 0:
            return f"cuda:{idx % num_gpus}"
        return "cpu"

    mp.set_start_method("spawn", force=True)
    result_queue: mp.Queue = mp.Queue()

    pending    = list(seeds)
    active: List[mp.Process] = []
    launched   = 0
    results    = []

    print(f"\n{'═'*62}")
    print(f"  Parallel DQN  |  seeds={len(seeds)}  "
          f"workers={max_workers}  GPUs={num_gpus}")
    print(f"  truncation={cfg['truncation']}  "
          f"ρ={cfg['replay_factor']}  "
          f"timesteps={cfg['total_timesteps']}")
    print(f"{'═'*62}\n")

    while pending or active:
        # Fill worker slots
        while pending and len(active) < max_workers:
            seed       = pending.pop(0)
            device_str = _device_for(launched)
            p = mp.Process(
                target=_train_worker,
                args=(cfg, seed, device_str, result_queue),
                daemon=True,
            )
            p.start()
            active.append(p)
            print(f"  ▶ launched  seed={seed:>3}  device={device_str}")
            launched += 1

        # Non-blocking result collection
        try:
            s = result_queue.get(timeout=2.0)
            results.append(s)
            print(f"  ✔ done      seed={s['seed']:>3}  "
                  f"eps={s['episodes']:>4}  "
                  f"best={s['best']:>8.1f}  "
                  f"last20={s['last20']:>8.1f}  "
                  f"time={s['elapsed']:>6.0f}s")
        except Exception:
            pass

        active = [p for p in active if p.is_alive()]

    # Drain any remaining results
    while len(results) < len(seeds):
        try:
            s = result_queue.get(timeout=5.0)
            results.append(s)
            print(f"  ✔ done      seed={s['seed']:>3}  "
                  f"eps={s['episodes']:>4}  "
                  f"best={s['best']:>8.1f}  "
                  f"last20={s['last20']:>8.1f}  "
                  f"time={s['elapsed']:>6.0f}s")
        except Exception:
            break

    print(f"\n{'═'*62}")
    print(f"  All {len(results)}/{len(seeds)} seeds complete.")
    print(f"  Logs → {os.path.abspath(cfg['log_dir'])}/")
    print(f"{'═'*62}\n")
    return results


# ═══════════════════════════════════════════════════════════════
# Single-process fallback  (--workers 1)
# ═══════════════════════════════════════════════════════════════
def run_serial(cfg: dict, seeds: List[int], device_str: str):
    q: queue_module.Queue = queue_module.Queue()
    for seed in seeds:
        _train_worker(cfg, seed, device_str, q)
        s = q.get()
        print(f"  ✔ seed={s['seed']:>3}  eps={s['episodes']:>4}  "
              f"best={s['best']:>8.1f}  last20={s['last20']:>8.1f}  "
              f"time={s['elapsed']:>6.0f}s")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel Vanilla DQN — MountainCar-v0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seeds",           type=int, nargs="+", default=list(range(15)))
    p.add_argument("--workers",         type=int, default=4,
                   help="Max parallel subprocesses. Use 1 for serial (debug).")
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--truncation",      type=int, default=2000)
    p.add_argument("--replay_factor",   type=int, default=1,
                   help="ρ: gradient updates per env timestep")
    p.add_argument("--log_dir",         type=str, default="logs")
    # Hyperparameters
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--buffer_size",     type=int,   default=50_000)
    p.add_argument("--batch_size",      type=int,   default=64)
    p.add_argument("--target_update",   type=int,   default=500)
    p.add_argument("--eps_start",       type=float, default=1.0)
    p.add_argument("--eps_end",         type=float, default=0.05)
    p.add_argument("--eps_decay",       type=int,   default=50_000)
    p.add_argument("--hidden",          type=int,   default=128)
    p.add_argument("--device",          type=str,   default="auto",
                   help="'auto' (round-robin GPUs), 'cpu', or 'cuda:0' etc.")
    return p.parse_args()


if __name__ == "__main__":
    args  = parse_args()

    cfg = dict(
        total_timesteps = args.total_timesteps,
        truncation      = args.truncation,
        replay_factor   = args.replay_factor,
        lr              = args.lr,
        gamma           = args.gamma,
        buffer_size     = args.buffer_size,
        batch_size      = args.batch_size,
        target_update   = args.target_update,
        eps_start       = args.eps_start,
        eps_end         = args.eps_end,
        eps_decay       = args.eps_decay,
        hidden          = args.hidden,
        log_dir         = args.log_dir,
        run_tag         = "",
    )

    if args.device != "auto":
        cfg["forced_device"] = args.device

    n_workers = min(args.workers, len(args.seeds))

    if n_workers == 1:
        device_str = (args.device if args.device != "auto"
                      else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        print(f"Serial mode  device={device_str}  seeds={args.seeds}")
        run_serial(cfg, args.seeds, device_str)
    else:
        run_parallel(cfg, seeds=args.seeds, max_workers=n_workers)