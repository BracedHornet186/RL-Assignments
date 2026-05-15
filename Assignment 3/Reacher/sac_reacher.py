"""
DA6400 RL PA3 - Section 2.3: Reacher
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random, os, json
from dm_control import suite
from tqdm import tqdm


# ─────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────
TOTAL_TIMESTEPS    = 500_000
EVAL_INTERVAL      = 10_000
EVAL_EPISODES      = 20
RANDOM_EXPLORE     = 10_000
BATCH_SIZE         = 256
REPLAY_BUFFER_SIZE = 1_000_000
GAMMA              = 0.99
TAU                = 0.005
LR_ACTOR = LR_CRITIC = LR_ALPHA = 3e-4
HIDDEN_DIM         = 256
EPISODE_LENGTH     = 1000
FINAL_EVAL_EPISODES  = 500
FINAL_EPISODE_LENGTH = 5000
LOG_STD_MAX, LOG_STD_MIN, EPSILON = 2, -20, 1e-6


# ─────────────────────────────────────────────────────────
# REWARD FUNCTIONS
# ─────────────────────────────────────────────────────────

def reward_ra(to_target, action):
    dist = float(np.linalg.norm(to_target))
    if dist < 0.05:
        return 1.0, True
    return -dist - float(np.dot(action, action)), False

def reward_rb(to_target):
    dist = float(np.linalg.norm(to_target))
    return (1.0, True) if dist < 0.05 else (0.0, False)

# ─────────────────────────────────────────────────────────
# ENVIRONMENT WRAPPER
# ─────────────────────────────────────────────────────────

class ReacherEnv:
    def __init__(self, reward_type="rb", seed=0):
        assert reward_type in ("ra", "rb", "rc")
        self.rtype = reward_type
        self.env   = suite.load("reacher", "easy", task_kwargs={"random": seed})
        self._step = 0
        ts = self.env.reset()
        self._obs_dim  = self._flat(ts.observation).shape[0]
        self._act_dim  = self.env.action_spec().shape[0]
        self._act_lim  = float(self.env.action_spec().maximum[0])

    @property
    def obs_dim(self): return self._obs_dim
    @property
    def act_dim(self): return self._act_dim
    @property
    def act_limit(self): return self._act_lim

    def _flat(self, d):
        return np.concatenate([np.array(d[k]).flatten()
                                for k in sorted(d)]).astype(np.float32)

    def reset(self):
        ts = self.env.reset()
        self._step = 0
        self._od   = ts.observation
        return self._flat(ts.observation)

    def step(self, action):
        action = np.clip(action, -self._act_lim, self._act_lim)
        ts     = self.env.step(action)
        od     = ts.observation
        self._step += 1
        tt  = np.array(od["to_target"]).flatten()
        vel = np.array(od.get("velocity", np.zeros(self._act_dim))).flatten()

        if self.rtype == "ra":
            r, in_t = reward_ra(tt, action)
            done, trunc = False, self._step >= EPISODE_LENGTH
        elif self.rtype == "rb":
            r, in_t = reward_rb(tt)
            done, trunc = False, self._step >= EPISODE_LENGTH
        else:  # rc
            dist = float(np.linalg.norm(tt))
            v    = float(np.linalg.norm(vel))
            in_t = dist < 0.05
            terminal = bool(in_t and v < 0.05)
            r = -1.0
            done = terminal
            trunc = False                         # Rc episodes never truncate
            if not terminal and self._step >= EPISODE_LENGTH:
                # Timeout: -20 penalty, env resets, but episode CONTINUES
                r += -20.0
                ts_reset = self.env.reset()
                od = ts_reset.observation
                self._step = 0
                tt   = np.array(od["to_target"]).flatten()
                in_t = float(np.linalg.norm(tt)) < 0.05

        self._od = od
        return self._flat(od), r, done, trunc, {"in_target": in_t, "to_target": tt}


# ─────────────────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size=REPLAY_BUFFER_SIZE):
        self.max_size = max_size
        self.ptr = self.size = 0
        self.obs      = np.zeros((max_size, obs_dim),  dtype=np.float32)
        self.action   = np.zeros((max_size, act_dim),  dtype=np.float32)
        self.reward   = np.zeros((max_size, 1),         dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim),  dtype=np.float32)
        self.done     = np.zeros((max_size, 1),         dtype=np.float32)

        # Pre-allocated pinned tensors for async H→D copy
        self._use_pin = torch.cuda.is_available()
        if self._use_pin:
            shapes = {"obs": (BATCH_SIZE, obs_dim), "action": (BATCH_SIZE, act_dim),
                      "reward": (BATCH_SIZE, 1), "next_obs": (BATCH_SIZE, obs_dim),
                      "done": (BATCH_SIZE, 1)}
            self._pin = {k: torch.zeros(*s).pin_memory() for k, s in shapes.items()}

    def add(self, obs, act, rew, nobs, done):
        p = self.ptr
        self.obs[p]=obs; self.action[p]=act; self.reward[p]=rew
        self.next_obs[p]=nobs; self.done[p]=done
        self.ptr  = (p+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        if self._use_pin:
            def _t(arr, key):
                self._pin[key].copy_(torch.from_numpy(arr[idx]))
                return self._pin[key].to(device, non_blocking=True)
            return (_t(self.obs,"obs"), _t(self.action,"action"),
                    _t(self.reward,"reward"), _t(self.next_obs,"next_obs"),
                    _t(self.done,"done"))
        def _t(arr): return torch.as_tensor(arr[idx]).to(device)
        return _t(self.obs), _t(self.action), _t(self.reward), \
               _t(self.next_obs), _t(self.done)


# ─────────────────────────────────────────────────────────
# NETWORKS
# ─────────────────────────────────────────────────────────

def mlp(in_d, out_d, h=HIDDEN_DIM):
    return nn.Sequential(nn.Linear(in_d,h), nn.ReLU(),
                         nn.Linear(h,h),    nn.ReLU(),
                         nn.Linear(h,out_d))

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.lim  = act_limit
        self.body = nn.Sequential(nn.Linear(obs_dim,HIDDEN_DIM), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM,HIDDEN_DIM), nn.ReLU())
        self.mu   = nn.Linear(HIDDEN_DIM, act_dim)
        self.ls   = nn.Linear(HIDDEN_DIM, act_dim)

    def sample(self, obs):
        x  = self.body(obs)
        mu = self.mu(x)
        ls = self.ls(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std= ls.exp()
        d  = Normal(mu, std)
        xt = d.rsample()
        yt = torch.tanh(xt)
        a  = yt * self.lim
        lp = d.log_prob(xt) - torch.log(self.lim*(1-yt.pow(2))+EPSILON)
        return a, lp.sum(-1, keepdim=True), torch.tanh(mu)*self.lim

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q1 = mlp(obs_dim+act_dim, 1)
        self.q2 = mlp(obs_dim+act_dim, 1)

    def forward(self, o, a):
        sa = torch.cat([o,a],-1)
        return self.q1(sa), self.q2(sa)

    def q_min(self, o, a):
        q1,q2 = self.forward(o,a)
        return torch.min(q1,q2)


# ─────────────────────────────────────────────────────────
# SAC AGENT  ← FIX 3 + FIX 4
# ─────────────────────────────────────────────────────────

class SAC:
    def __init__(self, obs_dim, act_dim, act_limit, device,
                 fixed_alpha=None, init_log_alpha=0.0,
                 target_entropy_override=None):
        self.device = device
        self.actor  = Actor(obs_dim, act_dim, act_limit).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.ctgt   = Critic(obs_dim, act_dim).to(device)
        self.ctgt.load_state_dict(self.critic.state_dict())
        for p in self.ctgt.parameters(): p.requires_grad_(False)

        self.a_opt  = optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.c_opt  = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.fixed_alpha    = fixed_alpha
        self.target_entropy = (target_entropy_override
                               if target_entropy_override is not None
                               else -float(act_dim))
        if fixed_alpha is None:
            self.log_alpha = torch.tensor([init_log_alpha], requires_grad=True,
                                          device=device)
            self.al_opt    = optim.Adam([self.log_alpha], lr=LR_ALPHA)
        else:
            self.log_alpha = torch.tensor([float(np.log(fixed_alpha))],
                                          device=device)
            self.al_opt    = None

        # FIX 3: pre-allocated obs tensor — avoids malloc every step
        self._obst = torch.zeros(1, obs_dim, device=device)

    @property
    def alpha(self): return self.log_alpha.exp()

    def select_action(self, obs, det=False):
        self._obst.copy_(torch.as_tensor(obs))   # in-place, no alloc
        with torch.no_grad():
            if det:
                _, _, ma = self.actor.sample(self._obst)
                return ma[0].cpu().numpy()
            a, _, _ = self.actor.sample(self._obst)
            return a[0].cpu().numpy()

    def update(self, buf):
        obs, act, rew, nobs, done = buf.sample(BATCH_SIZE, self.device)
        with torch.no_grad():
            na, nlp, _ = self.actor.sample(nobs)
            q1n, q2n   = self.ctgt(nobs, na)
            qt = rew + GAMMA*(1-done)*(torch.min(q1n,q2n) - self.alpha*nlp)

        q1,q2 = self.critic(obs,act)
        cl = F.mse_loss(q1,qt) + F.mse_loss(q2,qt)
        self.c_opt.zero_grad(set_to_none=True)
        cl.backward(); self.c_opt.step()

        pi, lp, _ = self.actor.sample(obs)
        al = (self.alpha.detach()*lp - self.critic.q_min(obs,pi)).mean()
        self.a_opt.zero_grad(set_to_none=True)
        al.backward(); self.a_opt.step()

        if self.fixed_alpha is None:
            ent_l = -(self.log_alpha*(lp+self.target_entropy).detach()).mean()
            self.al_opt.zero_grad(set_to_none=True)
            ent_l.backward(); self.al_opt.step()

        # FIX 4: vectorised soft update (single CUDA kernel, no Python loop)
        with torch.no_grad():
            torch._foreach_mul_(list(self.ctgt.parameters()), 1.0-TAU)
            torch._foreach_add_(list(self.ctgt.parameters()),
                                  list(self.critic.parameters()), alpha=TAU)


# ─────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────

def evaluate_policy(agent, eval_envs, n_eps=EVAL_EPISODES, ep_len=EPISODE_LENGTH):
    """eval_envs: {rtype: ReacherEnv} — created once per training run."""
    out = {}
    for rtype, env in eval_envs.items():
        rets = []
        for _ in range(n_eps):
            obs = env.reset(); ep_r = 0.0
            for _ in range(ep_len):
                obs, r, done, trunc, _ = env.step(agent.select_action(obs, det=True))
                ep_r += r
                if done or trunc: break
            rets.append(ep_r)
        out[rtype] = float(np.mean(rets))
    return out

def final_evaluate(agent, reward_type, seed):
    env = ReacherEnv(reward_type=reward_type, seed=seed+2000)
    stg_l, sit_l = [], []
    for _ in tqdm(range(FINAL_EVAL_EPISODES), desc="  final eval", leave=False):
        obs = env.reset(); reached=False; stg=FINAL_EPISODE_LENGTH; sit=0
        for t in range(FINAL_EPISODE_LENGTH):
            obs,_,done,trunc,info = env.step(agent.select_action(obs, det=True))
            if info["in_target"]:
                if not reached: reached=True; stg=t+1
                sit += 1
            if done or trunc: break
        stg_l.append(stg); sit_l.append(sit)
    return np.array(stg_l), np.array(sit_l)


# ─────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────

def train_sac(reward_type, seed, total_timesteps=TOTAL_TIMESTEPS,
              save_dir="results", utd_ratio=None, sac_kwargs=None,
              run_tag=None, progress_file=None):
    if utd_ratio is None:
        utd_ratio = 4 if reward_type == "rc" else 1
    tag   = run_tag if run_tag else f"sac_r{reward_type}"
    fname = os.path.join(save_dir, f"{tag}_seed{seed}.json")
    if os.path.exists(fname):
        # signal "100% done" to any external monitor so its bar can settle.
        if progress_file is not None:
            try:
                with open(progress_file, "w") as f:
                    f.write(f"{total_timesteps} {total_timesteps}\n")
            except OSError: pass
        print(f"  [SKIP] {fname} exists."); return

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_env = ReacherEnv(reward_type=reward_type, seed=seed)
    obs_dim, act_dim, act_lim = (train_env.obs_dim,
                                  train_env.act_dim, train_env.act_limit)

    # FIX 1: build eval envs once, reuse every 10K steps
    eval_envs = {rt: ReacherEnv(reward_type=rt, seed=seed+1000)
                 for rt in ("ra","rb","rc")}

    agent  = SAC(obs_dim, act_dim, act_lim, device, **(sac_kwargs or {}))
    buffer = ReplayBuffer(obs_dim, act_dim)

    log = {"timesteps":[], "eval_ra":[], "eval_rb":[], "eval_rc":[]}
    obs = train_env.reset(); rng = np.random.default_rng(seed)

    # When the parent monitors via progress_file, suppress this worker's bar.
    pbar = tqdm(total=total_timesteps,
                desc=f"{tag} s{seed}", unit="step",
                dynamic_ncols=True,
                disable=(progress_file is not None))

    for n in range(1, total_timesteps+1):
        if n <= RANDOM_EXPLORE:
            action = rng.uniform(-act_lim, act_lim, size=act_dim).astype(np.float32)
        else:
            action = agent.select_action(obs)

        nobs, r, done, trunc, _ = train_env.step(action)
        buffer.add(obs, action, r, nobs, float(done and not trunc))
        obs = nobs
        if done or trunc: obs = train_env.reset()

        if n >= RANDOM_EXPLORE and buffer.size >= BATCH_SIZE:
            for _ in range(utd_ratio):
                agent.update(buffer)

        if n % EVAL_INTERVAL == 0:
            rets = evaluate_policy(agent, eval_envs)
            log["timesteps"].append(n)
            for rt in ("ra","rb","rc"): log[f"eval_{rt}"].append(rets[rt])
            pbar.set_postfix({f"R{reward_type.upper()}": f"{rets[reward_type]:+.1f}",
                              "α": f"{agent.alpha.item():.3f}"})

        if progress_file is not None and n % 1000 == 0:
            try:
                with open(progress_file, "w") as f:
                    f.write(f"{n} {total_timesteps}\n")
            except OSError: pass

        pbar.update(1)
    pbar.close()

    stg, sit = final_evaluate(agent, reward_type, seed)
    log["final_steps_to_goal"]   = stg.tolist()
    log["final_steps_in_target"] = sit.tolist()

    os.makedirs(save_dir, exist_ok=True)
    with open(fname,"w") as f: json.dump(log, f)
    print(f"  Saved → {fname}")


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--reward",   default="rb", choices=["ra","rb","rc"])
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--steps",    type=int, default=TOTAL_TIMESTEPS)
    p.add_argument("--save_dir", default="results")
    p.add_argument("--gamma",          type=float, default=None,
                   help="Override discount factor (module-level GAMMA)")
    p.add_argument("--target_entropy", type=float, default=None,
                   help="Override target entropy (default = -act_dim)")
    p.add_argument("--init_log_alpha", type=float, default=0.0,
                   help="Initial log α for auto-tuned SAC (default 0 → α=1)")
    p.add_argument("--fixed_alpha",    type=float, default=None,
                   help="If set, disables auto-tune and uses this α")
    p.add_argument("--utd_ratio",      type=int,   default=None,
                   help="Gradient updates per env step "
                        "(default: 4 for rc, 1 for ra/rb)")
    p.add_argument("--run_tag",        type=str,   default=None,
                   help="Filename tag (default 'sac_r<reward>')")
    p.add_argument("--progress_file",  type=str,   default=None,
                   help="Path to write 'step total' every 1K env steps "
                        "(used by run_experiments.py to drive a parent tqdm bar)")
    a = p.parse_args()

    if a.gamma is not None:
        GAMMA = a.gamma                     # updates module global
        print(f"  [hp] GAMMA overridden to {GAMMA}")

    sac_kwargs = {}
    if a.target_entropy is not None: sac_kwargs["target_entropy_override"] = a.target_entropy
    if a.init_log_alpha != 0.0:      sac_kwargs["init_log_alpha"]          = a.init_log_alpha
    if a.fixed_alpha is not None:    sac_kwargs["fixed_alpha"]             = a.fixed_alpha

    train_sac(a.reward, a.seed, a.steps, a.save_dir,
              utd_ratio=a.utd_ratio, sac_kwargs=sac_kwargs,
              run_tag=a.run_tag, progress_file=a.progress_file)
