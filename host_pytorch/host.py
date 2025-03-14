from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear

from hl_gauss_pytorch import HLGaussLoss

from einops import repeat
from einops.layers.torch import EinMix as Mix

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

class GroupedMLP(Module):
    def __init__(
        self,
        *dims,
        num_mlps = 1,
    ):
        super().__init__()

        assert len(dims) >= 2, 'must have at least two dimensions'

        dim_pairs = list(zip(dims[:-1], dims[1:]))

        # handle first layer as no grouped dimension yet

        first_dim_in, first_dim_out = dim_pairs.pop(0)

        first_layer = Mix('b ... i -> b ... g o', weight_shape = 'g i o', bias_shape = 'g o', g = num_mlps, i = first_dim_in, o = first_dim_out)

        # rest of the layers

        layers = [Mix('b ... g i -> b ... g o', weight_shape = 'g i o', bias_shape = 'g o', g = num_mlps, i = dim_in, o = dim_out) for dim_in, dim_out in dim_pairs]

        self.layers = ModuleList([first_layer, *layers])
        self.num_mlps = num_mlps

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
