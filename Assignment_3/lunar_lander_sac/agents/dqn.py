"""
Deep Q-Network (DQN) with experience replay and target network.
Used to compare with Discrete-SAC on LunarLander-v3 (discrete).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.sac import ReplayBuffer, mlp


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim, hidden, n_actions)

    def forward(self, obs):
        return self.net(obs)


class DQN:
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
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=50000,
        device="cpu",
    ):
        self.device     = device
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.n_actions  = n_actions
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.steps      = 0

        self.qnet        = QNetwork(obs_dim, n_actions, hidden).to(device)
        self.qnet_target = QNetwork(obs_dim, n_actions, hidden).to(device)
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.replay    = ReplayBuffer(obs_dim, 1, buffer_size)

    @property
    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -self.steps / self.eps_decay
        )

    def select_action(self, obs, evaluate=False):
        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.qnet(obs_t).argmax(dim=-1).item()

    def store(self, obs, action, reward, next_obs, done):
        self.replay.add(obs, np.array([action]), reward, next_obs, done)

    def _soft_update(self):
        for p, tp in zip(self.qnet.parameters(), self.qnet_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def update(self):
        self.steps += 1
        if self.replay.size < self.batch_size:
            return {}

        obs, action, reward, next_obs, done = self.replay.sample(self.batch_size, self.device)
        action = action.long()

        with torch.no_grad():
            q_next = self.qnet_target(next_obs).max(dim=-1, keepdim=True)[0]
            q_target = reward + self.gamma * (1 - done) * q_next

        q_val = self.qnet(obs).gather(1, action)
        loss = nn.functional.mse_loss(q_val, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._soft_update()

        return {"loss": loss.item(), "epsilon": self.epsilon}
