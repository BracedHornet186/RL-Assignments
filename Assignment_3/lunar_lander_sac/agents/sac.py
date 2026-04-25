"""
Soft Actor-Critic (SAC) with automated temperature tuning.
Supports both continuous and discrete action spaces.
Uses clipped double Q-learning and reparameterization trick.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np


# ─────────────────────────────────────────────
#  Shared MLP utility
# ─────────────────────────────────────────────
def mlp(input_dim, hidden_dims, output_dim, activation=nn.ReLU, output_activation=None):
    layers = []
    dims = [input_dim] + list(hidden_dims)
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), activation()]
    layers.append(nn.Linear(dims[-1], output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────
#  Replay Buffer
# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs      = np.zeros((max_size, obs_dim),    dtype=np.float32)
        self.action   = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward   = np.zeros((max_size, 1),          dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim),    dtype=np.float32)
        self.done     = np.zeros((max_size, 1),          dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.action[self.ptr]   = action
        self.reward[self.ptr]   = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr]     = done
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[idx]).to(device),
            torch.FloatTensor(self.action[idx]).to(device),
            torch.FloatTensor(self.reward[idx]).to(device),
            torch.FloatTensor(self.next_obs[idx]).to(device),
            torch.FloatTensor(self.done[idx]).to(device),
        )


# ─────────────────────────────────────────────
#  CONTINUOUS SAC
# ─────────────────────────────────────────────
LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6


class GaussianActor(nn.Module):
    """Squashed Gaussian policy (tanh)."""
    def __init__(self, obs_dim, action_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim, hidden[:-1], hidden[-1])
        self.mu_head  = nn.Linear(hidden[-1], action_dim)
        self.log_std_head = nn.Linear(hidden[-1], action_dim)

    def forward(self, obs):
        h = F.relu(self.net(obs))
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        x = dist.rsample()                         # reparameterization
        y = torch.tanh(x)
        log_prob = dist.log_prob(x) - torch.log(1 - y.pow(2) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return y, log_prob

    def get_action(self, obs):
        with torch.no_grad():
            action, _ = self.sample(obs)
        return action.cpu().numpy()


class Critic(nn.Module):
    """Twin Q-networks (clipped double Q-learning)."""
    def __init__(self, obs_dim, action_dim, hidden=(256, 256)):
        super().__init__()
        in_dim = obs_dim + action_dim
        self.q1 = mlp(in_dim, hidden, 1)
        self.q2 = mlp(in_dim, hidden, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, obs, action):
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


class SAC:
    """
    Continuous SAC with automated temperature tuning.
    """
    def __init__(
        self,
        obs_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        buffer_size=int(1e6),
        hidden=(256, 256),
        auto_alpha=True,
        alpha=0.2,
        device="cpu",
    ):
        self.device      = device
        self.gamma       = gamma
        self.tau         = tau
        self.batch_size  = batch_size
        self.auto_alpha  = auto_alpha

        # Networks
        self.actor  = GaussianActor(obs_dim, action_dim, hidden).to(device)
        self.critic = Critic(obs_dim, action_dim, hidden).to(device)
        self.critic_target = Critic(obs_dim, action_dim, hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        # Temperature
        if auto_alpha:
            self.target_entropy = -action_dim          # heuristic
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        self.replay = ReplayBuffer(obs_dim, action_dim, buffer_size)

    # ── soft update ──────────────────────────────────────────────
    def _soft_update(self):
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    # ── single gradient step ─────────────────────────────────────
    def update(self):
        if self.replay.size < self.batch_size:
            return {}

        obs, action, reward, next_obs, done = self.replay.sample(self.batch_size, self.device)

        # ── Critic loss ──────────────────────────────────────────
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_action)
            q_target = reward + self.gamma * (1 - done) * (
                torch.min(q1_t, q2_t) - self.alpha * next_log_pi
            )

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Actor loss ───────────────────────────────────────────
        pi, log_pi = self.actor.sample(obs)
        q_pi = self.critic.q_min(obs, pi)
        actor_loss = (self.alpha * log_pi - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── Alpha loss ───────────────────────────────────────────
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item()

        self._soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha_loss":  alpha_loss,
            "alpha":       self.alpha,
        }

    def select_action(self, obs, evaluate=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        if evaluate:
            with torch.no_grad():
                mu, _ = self.actor.forward(obs_t)
                return torch.tanh(mu).cpu().numpy()[0]
        return self.actor.get_action(obs_t)[0]

    def store(self, obs, action, reward, next_obs, done):
        self.replay.add(obs, action, reward, next_obs, done)


# ─────────────────────────────────────────────
#  DISCRETE SAC
# ─────────────────────────────────────────────
class DiscreteActor(nn.Module):
    """Softmax policy over discrete actions."""
    def __init__(self, obs_dim, n_actions, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim, hidden, n_actions)

    def forward(self, obs):
        return F.softmax(self.net(obs), dim=-1)

    def sample(self, obs):
        probs = self.forward(obs)
        dist  = Categorical(probs)
        action = dist.sample()
        log_prob = torch.log(probs + EPSILON)        # log prob for ALL actions
        return action, probs, log_prob

    def get_action(self, obs):
        with torch.no_grad():
            probs = self.forward(obs)
            return Categorical(probs).sample().item()


class DiscreteCritic(nn.Module):
    """Twin Q-networks mapping obs → Q(s,·) for all actions."""
    def __init__(self, obs_dim, n_actions, hidden=(256, 256)):
        super().__init__()
        self.q1 = mlp(obs_dim, hidden, n_actions)
        self.q2 = mlp(obs_dim, hidden, n_actions)

    def forward(self, obs):
        return self.q1(obs), self.q2(obs)


class DiscreteSAC:
    """
    Discrete SAC (Christodoulou 2019).
    Uses E_a[Q(s,a)] weighted by policy probabilities.
    """
    def __init__(
        self,
        obs_dim,
        n_actions,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        buffer_size=int(1e6),
        hidden=(256, 256),
        auto_alpha=True,
        alpha=0.2,
        device="cpu",
    ):
        self.device     = device
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.auto_alpha = auto_alpha
        self.n_actions  = n_actions

        self.actor  = DiscreteActor(obs_dim, n_actions, hidden).to(device)
        self.critic = DiscreteCritic(obs_dim, n_actions, hidden).to(device)
        self.critic_target = DiscreteCritic(obs_dim, n_actions, hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        if auto_alpha:
            self.target_entropy = -np.log(1.0 / n_actions) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        # Use scalar action_dim=1 for buffer (store action index)
        self.replay = ReplayBuffer(obs_dim, 1, buffer_size)

    def _soft_update(self):
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def update(self):
        if self.replay.size < self.batch_size:
            return {}

        obs, action, reward, next_obs, done = self.replay.sample(self.batch_size, self.device)
        action = action.long()   # indices

        # ── Critic loss ──────────────────────────────────────────
        with torch.no_grad():
            _, next_probs, next_log_probs = self.actor.sample(next_obs)
            q1_t, q2_t = self.critic_target(next_obs)
            q_min_t = torch.min(q1_t, q2_t)
            # Soft Bellman backup: E_a[Q - alpha*logpi]
            v_next = (next_probs * (q_min_t - self.alpha * next_log_probs)).sum(dim=-1, keepdim=True)
            q_target = reward + self.gamma * (1 - done) * v_next

        q1_all, q2_all = self.critic(obs)
        q1 = q1_all.gather(1, action)
        q2 = q2_all.gather(1, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Actor loss ───────────────────────────────────────────
        _, probs, log_probs = self.actor.sample(obs)
        q1_all, q2_all = self.critic(obs)
        q_min = torch.min(q1_all, q2_all).detach()
        actor_loss = (probs * (self.alpha * log_probs - q_min)).sum(dim=-1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── Alpha loss ───────────────────────────────────────────
        alpha_loss = 0.0
        if self.auto_alpha:
            entropy = -(probs * log_probs).sum(dim=-1)
            alpha_loss = (self.log_alpha * (entropy - self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item()

        self._soft_update()
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha_loss":  alpha_loss,
            "alpha":       self.alpha,
        }

    def select_action(self, obs, evaluate=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        if evaluate:
            with torch.no_grad():
                probs = self.actor.forward(obs_t)
                return probs.argmax(dim=-1).item()
        return self.actor.get_action(obs_t)

    def store(self, obs, action, reward, next_obs, done):
        self.replay.add(obs, np.array([action]), reward, next_obs, done)
