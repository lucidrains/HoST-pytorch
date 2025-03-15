from __future__ import annotations

import torch
from torch import tensor
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn import Module, ModuleList, Linear

from hl_gauss_pytorch import HLGaussLoss

import einx
from einops import repeat, rearrange, reduce
from einops.layers.torch import EinMix as Mix

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# sampling related

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def calc_entropy(prob, eps = 1e-20, dim = -1):
    return -(prob * log(prob, eps)).sum(dim = dim)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

# reward functions - following table 6

# task rewards

# style rewards

# regularization rewards

# post task reward

# networks

# simple mlp for actor

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

# actor

class Actor(Module):
    def __init__(
        self,
        num_actions,
        dims: tuple[int, ...] = (512, 256, 128),
        eps_clip = 0.2,
        beta_s = .01,
    ):
        super().__init__()

        dims = (*dims, num_actions)

        self.net = MLP(*dims)

        # ppo loss related

        self.eps_clip = eps_clip
        self.beta_s = beta_s

    def forward_for_loss(
        self,
        state,
        actions,
        old_log_probs,
        advantages
    ):
        clip = self.eps_clip
        logits = self.net(state)

        prob = logits.softmax(dim = -1)

        distribution = Categorical(prob)

        log_probs = distribution.log_prob(actions)

        ratios = (log_probs - old_log_probs).exp()

        # classic clipped surrogate objective from ppo

        surr1 = ratios * advantages
        surr2 = ratios.clamp(1. - clip, 1. + clip) * advantages
        loss = -torch.min(surr1, surr2) - self.beta_s * calc_entropy(prob)

        return loss.sum()

    def forward(
        self,
        state,
        sample = False,
        sample_return_log_prob = True
    ):

        logits = self.net(state)

        prob = logits.softmax(dim = -1)

        if not sample:
            return prob

        distribution = Categorical(prob)

        actions = distribution.sample()

        log_prob = distribution.log_prob(actions)

        if not sample_return_log_prob:
            return sampled_actions

        return actions, log_prob

# grouped mlp
# for multiple critics in one forward pass
# all critics must share the same MLP network structure

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

# critics

class Critics(Module):
    def __init__(
        self,
        weights: tuple[float, ...],
        dims: tuple[int, ...] = (512, 256),
        num_critics = 4,
    ):
        super().__init__()
        dims = (*dims, 1)

        self.mlps = GroupedMLP(*dims, num_mlps = num_critics)

        assert len(weights) == num_critics
        self.register_buffer('weights', tensor(weights))

    @torch.no_grad()
    def calc_advantages(
        self,
        values,
        rewards
    ):
        batch = values.shape[0]

        advantages = rewards - values

        advantages = rearrange(advantages, 'b g -> g b')
        norm_advantages = F.layer_norm(advantages, (batch,))

        weighted_norm_advantages = einx.multiply('g b, g', norm_advantages, self.weights)
        return reduce(weighted_norm_advantages, 'g b -> b', 'sum')

    def forward(
        self,
        state,
        rewards = None # Float['b g']
    ):
        values = self.mlps(state)
        values = rearrange(values, '... 1 -> ...')

        if not exists(rewards):
            return values

        return F.mse_loss(rewards, values)
