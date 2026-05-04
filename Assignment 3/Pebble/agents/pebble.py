"""
PEBBLE: Feedback Efficient Interactive Reinforcement Learning via
Relabelling Experience and Unsupervised Pre-training (Lee et al. 2021)
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque


# ─────────────────────────────────────────────
#  Reward Model  r_ψ(s, a)
# ─────────────────────────────────────────────
class RewardModel(nn.Module):
    """
    Learned reward function r_ψ : (obs, action) → R.
    Trained via binary cross-entropy on human preference labels.
    """
    def __init__(self, obs_dim, action_dim, hidden=(256, 256), lr=3e-4, device="cpu"):
        super().__init__()
        dims = [obs_dim + action_dim] + list(hidden) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1])]
            if i < len(dims) - 2:
                layers += [nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)   # (B, 1)

    def predict_reward(self, obs, action):
        """Scalar reward for a single (obs, action) pair — used during SAC training."""
        with torch.no_grad():
            obs_t    = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            return self.forward(obs_t, action_t).item()

    def segment_return(self, obs_seg, act_seg):
        """
        Sum r_ψ over a trajectory segment.
        obs_seg : (T, obs_dim)  tensors
        act_seg : (T, action_dim)
        Returns scalar tensor.
        """
        r = self.forward(obs_seg, act_seg)   # (T, 1)
        return r.sum()

    def update(self, seg1_obs, seg1_act, seg2_obs, seg2_act, labels):
        """
        Preference loss (Bradley-Terry model):
          P(σ¹ ≻ σ²) = exp(Σr(s,a) over σ¹) / [exp(Σr σ¹) + exp(Σr σ²)]
          L = -E[y log P(σ¹≻σ²) + (1-y) log P(σ²≻σ¹)]
        where y=1 means segment 1 is preferred.

        All inputs are tensors on self.device.
        labels : (B,) float, 1 if seg1 preferred, 0 if seg2 preferred, 0.5 if equal.
        """
        B = seg1_obs.shape[0]

        r1_list, r2_list = [], []
        for i in range(B):
            r1_list.append(self.segment_return(seg1_obs[i], seg1_act[i]))
            r2_list.append(self.segment_return(seg2_obs[i], seg2_act[i]))

        r1 = torch.stack(r1_list)   # (B,)
        r2 = torch.stack(r2_list)   # (B,)

        # Bradley-Terry preference probability
        logits = torch.stack([r1, r2], dim=1)          # (B, 2)
        # label=1 → prefer seg1 → class 0 in cross_entropy (r1 should be higher)
        # label=0 → prefer seg2 → class 1
        # label=0.5 → equal → soft target
        soft_labels = torch.stack([labels, 1.0 - labels], dim=1)   # (B, 2)
        log_probs   = F.log_softmax(logits, dim=1)
        loss = -(soft_labels * log_probs).sum(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"model": self.state_dict(),
                    "opt":   self.optimizer.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["opt"])


# ─────────────────────────────────────────────
#  Preference Buffer
# ─────────────────────────────────────────────
class PreferenceBuffer:
    """
    Stores (segment1, segment2, label) triples.
    Each segment is a fixed-length sequence of (obs, action) pairs.
    """
    def __init__(self, max_size=3000, segment_len=50):
        self.max_size    = max_size
        self.segment_len = segment_len
        self.buffer      = deque(maxlen=max_size)

    def add(self, seg1_obs, seg1_act, seg2_obs, seg2_act, label):
        """label: 1.0 = seg1 preferred, 0.0 = seg2 preferred, 0.5 = equal."""
        self.buffer.append((seg1_obs, seg1_act, seg2_obs, seg2_act, label))

    def sample(self, batch_size, device):
        idx = np.random.randint(0, len(self.buffer), size=batch_size)
        s1o, s1a, s2o, s2a, lbl = [], [], [], [], []
        for i in idx:
            item = self.buffer[i]
            s1o.append(item[0]); s1a.append(item[1])
            s2o.append(item[2]); s2a.append(item[3])
            lbl.append(item[4])

        return (
            torch.FloatTensor(np.array(s1o)).to(device),
            torch.FloatTensor(np.array(s1a)).to(device),
            torch.FloatTensor(np.array(s2o)).to(device),
            torch.FloatTensor(np.array(s2a)).to(device),
            torch.FloatTensor(np.array(lbl)).to(device),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
#  Simulated Teacher
# ─────────────────────────────────────────────
class SimulatedTeacher:
    """
    Oracle teacher that labels preference queries using ground-truth reward.
    Given two segments, labels 1 if seg1 has higher total GT reward, 0 otherwise.
    Adds noise via error_prob to simulate imperfect human feedback.
    """
    def __init__(self, gt_reward_fn, error_prob=0.0):
        """
        gt_reward_fn : fn(obs_seq, act_seq) -> float
            Computes ground-truth return for a segment.
        error_prob : probability of flipping the label (teacher noise).
        """
        self.gt_reward_fn = gt_reward_fn
        self.error_prob   = error_prob

    def label(self, seg1_obs, seg1_act, seg2_obs, seg2_act):
        """Returns label: 1.0 if seg1 preferred, 0.0 if seg2, 0.5 if equal."""
        r1 = self.gt_reward_fn(seg1_obs, seg1_act)
        r2 = self.gt_reward_fn(seg2_obs, seg2_act)

        if np.isclose(r1, r2, atol=1e-4):
            return 0.5

        label = 1.0 if r1 > r2 else 0.0

        # Flip with error_prob
        if self.error_prob > 0 and np.random.rand() < self.error_prob:
            label = 1.0 - label

        return label