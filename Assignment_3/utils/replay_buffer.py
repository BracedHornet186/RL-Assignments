import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, size, device):
        # np.empty is much faster than np.zeros for large buffer allocation
        self.obs = np.empty((size, obs_dim), dtype=np.float32)
        self.next_obs = np.empty((size, obs_dim), dtype=np.float32)
        self.actions = np.empty((size, action_dim), dtype=np.float32)
        self.rewards = np.empty((size, 1), dtype=np.float32)
        self.not_done = np.empty((size, 1), dtype=np.float32)
        self.not_done_no_max = np.empty((size, 1), dtype=np.float32) 

        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        # Using np.copyto is slightly faster than direct assignment for arrays
        np.copyto(self.obs[self.ptr], obs)
        np.copyto(self.next_obs[self.ptr], next_obs)
        np.copyto(self.actions[self.ptr], action)
        
        self.rewards[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.not_done_no_max[self.ptr] = 1. - done_no_max

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        # torch.as_tensor avoids unnecessary data copies
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.actions[idx], device=self.device),
            torch.as_tensor(self.rewards[idx], device=self.device),
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.not_done[idx], device=self.device),
            torch.as_tensor(self.not_done_no_max[idx], device=self.device),
        )