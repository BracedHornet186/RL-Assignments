import torch
import torch.nn as nn


# 🔹 MLP builder
def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    layers = []
    dim = input_dim

    for _ in range(hidden_depth):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.ReLU())
        dim = hidden_dim

    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


# 🔹 Weight initialization (Kaiming)
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)


# 🔹 Soft update (for target networks)
def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


# 🔹 Tensor → numpy
def to_np(t):
    return t.detach().cpu().numpy()