from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear

from hl_gauss_pytorch import HLGaussLoss

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# reward functions - following table 6

# task rewards

# style rewards

# regularization rewards

# post task reward

# actor critic networks

class MLP(Module):
    def __init__(
        self,
        *dims
    ):
        super().__init__()
        assert len(dims) >= 2, 'must have at least two dimensions'

        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        layers = ModuleList([Linear(dim_in, dim_out) for dim_in, dim_out in dim_pairs])

        self.layers = layers

    def forward(
        self,
        x
    ):

        for ind, layer in enumerate(self.layers, start = 1):
            is_last = ind == len(self.layers)

            x = layer(x)

            if not is_last:
                x = F.silu(x)

        return x

class Actor(Module):
    def __init__(
        self,
        dims = (512, 256, 128)
    ):
        super().__init__()

        self.mlp = MLP(*dims)

    def forward(
        self,
        x
    ):
        return self.mlp(x)

class Critic(Module):
    def __init__(
        self,
        dims = (512, 256)
    ):
        super().__init__()

        self.mlp = MLP(*dims)

    def forward(
        self,
        x
    ):
        return self.mlp(x)

class Critics(Module):
    def __init__(
        self,
        critics: list[Critic]
    ):
        super().__init__()

        self.critics = critics
