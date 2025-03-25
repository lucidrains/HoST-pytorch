from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import tensor, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear

from hl_gauss_pytorch import HLGaussLoss

import einx
from einops import repeat, rearrange, reduce
from einops.layers.torch import EinMix as Mix

# constants

INF = float('inf')

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

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def get_log_prob(t, indices, is_prob = False):
    log_probs = t.log_softmax(dim = -1) if not is_prob else log(t)
    indices = rearrange(indices, '... -> ... 1')
    sel_log_probs = log_probs.gather(-1, indices)
    return rearrange(sel_log_probs, '... 1 -> ...')

# generalized advantage estimate

def calc_gae(
    rewards,
    values,
    masks,
    gamma = 0.99,
    lam = 0.95
):

    device = rewards.device

    gae = 0.
    returns = torch.empty_like(rewards)

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lam * masks[i] * gae
        returns[i] = gae + values[i]

    return returns

# === reward functions === table 6 - they have a mistake where they redefine ankle parallel reward twice

@dataclass
class State:
    head_height: Float['']
    projected_gravity_vector: Float[''] # orientation of robot base
    joint_velocity: Float['d']
    joint_acceleration: Float['d']

# the f_tol function in the paper

# which is a reward shaping function that defines a reward of 1. for anything between the bounds, 0. anything outside the margins, and then some margin value with a gaussian shape otherwise
# this came from "deepmind control" paper - https://github.com/google-deepmind/dm_control/blob/46390cfc356dfcb4235a2417efb2c3ab260194b8/dm_control/utils/rewards.py#L93
# we will just stick with the default gaussian, and what was used in this paper, for simplicity

def ftol(
    value: Tensor,
    bounds: tuple[float, float],
    margin = 0.,
    value_at_margin = 0.1
):
    low, high = bounds

    assert low < high, 'invalid bounds'
    assert margin >= 0., 'margin must be greater equal to 0.'

    in_bounds = low <= value <= high

    if margin == 0.:
        return in_bounds.float()

    # gaussian sigmoid

    distance_margin = torch.where(x < low, low - x, x - high) / margin

    scale = torch.sqrt(-2 * value_at_margin.log())
    return (-0.5 * (distance_margin * scale).pow(2)).exp()

# task rewards - It specifies the high-level task objectives.

def reward_head_height(state: State):
    """ The head of robot head in the world frame """

    return ftol(state.head_height, (1., INF), 1., 0.1)

def reward_base_orientation(state: State):
    """ The orientation of the robot base represented by projected gravity vector. """

    θz_base = state.projected_gravity_vector
    return ftol(-θz_base, (0.99, INF), 1., 0.05)

# style rewards - It specifies the style of standing-up motion.

def reward_waist_yaw_deviation(state: State):
    """ It penalizes the large joint angle of the waist yaw. """
    raise NotImplementedError

def reward_hip_roll_yaw_deviation(state: State):
    """ It penalizes the large joint angle of hip roll/yaw joints. """
    raise NotImplementedError

def reward_shoulder_roll_deviation(state: State):
    """ It penalizes the large joint angle of shoulder roll joint. """
    raise NotImplementedError

def reward_foot_displacement(state: State):
    """ It encourages robot CoM locates in support polygon, inspired by https://ieeexplore.ieee.org/document/1308858 """
    raise NotImplementedError

def reward_ankle_parallel(state: State):
    """ It encourages the ankles to be parallel to the ground via ankle keypoints. """
    raise NotImplementedError

def reward_foot_distance(state: State):
    """ It penalizes a far distance between feet. """
    raise NotImplementedError

def reward_foot_stumble(state: State):
    """ It penalizes a horizontal contact force with the environment. """
    raise NotImplementedError

def reward_shank_orientation(state: State):
    """ It encourages the left/right shank to be perpendicular to the ground. """
    raise NotImplementedError

def reward_waist_yaw_deviation(state: State):
    """ It penalizes the large joint angle of the waist yaw. """
    raise NotImplementedError

def reward_base_angular_velocity(state: State):
    """ It encourages low angular velocity of the during rising up. """
    raise NotImplementedError

# regularization rewards - It specifies the regulariztaion on standing-up motion.

def reward_joint_acceleration(state: State):
    """ It penalizes the high joint accelrations. """

    return state.joint_acceleration.norm(dim = -1) ** 2

def reward_action_rate(state: State):
    """ It penalizes the high changing speed of action. """
    raise NotImplementedError

def reward_smoothness(state: State):
    """ It penalizes the discrepancy between consecutive actions. """
    raise NotImplementedError

def reward_torques(state: State):
    """ It penalizes the high joint torques. """
    raise NotImplementedError

def reward_joint_power(state: State):
    """ It penalizes the high joint power """
    raise NotImplementedError

def reward_joint_velocity(state: State):
    """ It penalizes the high joint velocity. """

    return state.joint_velocity.norm(dim = -1) ** 2

def reward_joint_tracking_error(state: State):
    """ It penalizes the error between PD target (Eq. (1)) and actual joint position. """
    raise NotImplementedError

def reward_joint_pos_limits(state: State):
    """ It penalizes the joint position that beyond limits. """
    raise NotImplementedError

def reward_joint_vel_limits(state: State):
    """ It penalizes the joint velocity that beyond limits. """
    raise NotImplementedError

# post task reward - It specifies the desired behaviors after a successful standing up.

def reward_base_angular_velocity(state: State):
    """ It encourages low angular velocity of robot base after standing up. """
    raise NotImplementedError

def reward_base_linear_velocity(state: State):
    """ It encourages low linear velocity of robot base after standing up. """
    raise NotImplementedError

def reward_base_orientation(state: State):
    """ It encourages the robot base to be perpendicular to the ground. """
    raise NotImplementedError

def reward_base_height(state: State):
    """ It encourages the robot base to reach a target height. """
    raise NotImplementedError

def reward_upper_body_posture(state: State):
    """ It encourages the robot to track a target upper body postures. """
    raise NotImplementedError

def reward_feet_parallel(state: State):
    """ In encourages the feet to be parallel to each other. """
    raise NotImplementedError

# reward config with all the weights

REWARD_CONFIG = [
    ('task', 2.5, [
        (reward_head_height, 1),
        (reward_base_orientation, 1),
    ]),
    ('style', 1., [
        (reward_waist_yaw_deviation, -10),
        (reward_hip_roll_yaw_deviation, -10 / 10),
        (reward_shoulder_roll_deviation, -0.25 / -10),
        (reward_foot_displacement, -2.5),
        (reward_ankle_parallel, 2.5 / 2.5),
        (reward_foot_distance, 20),
        (reward_foot_stumble, -10),
        (reward_shank_orientation, 0 / -25), # do not understand this 0(G) / -25(PSW)
        (reward_waist_yaw_deviation, 10),
        (reward_base_angular_velocity, 1),
    ]),
    ('regularization', 0.1, [
        (reward_joint_acceleration, -2.5e-7),
        (reward_action_rate, -1e-2),
        (reward_smoothness, -1e-2),
        (reward_torques, -2.5e-6),
        (reward_joint_power, -2.5e-5),
        (reward_joint_velocity, -1e-4),
        (reward_joint_tracking_error, -2.5e-1),
        (reward_joint_pos_limits, -1e2),
        (reward_joint_vel_limits, -1.),
    ]),
    ('post_task', 1, [
        (reward_base_angular_velocity, 10),
        (reward_base_linear_velocity, 10),
        (reward_base_orientation, 10),
        (reward_base_height, 10),
        (reward_upper_body_posture, 10),
        (reward_feet_parallel, 2.5),
    ])
]

class RewardShapingWrapper(Module):
    def __init__(
        self,
        config = REWARD_CONFIG,
        critics_kwargs: dict = dict()
    ):
        super().__init__()

        self.config = config

        # based on the reward group config
        # can instantiate the critics automatically

        num_reward_groups = len(config)
        critics_weights = [reward_group_weight for _, reward_group_weight, _ in config]

        self.critics = Critics(
            critics_weights,
            num_critics = num_reward_groups,
            **critics_kwargs
        )

        # then store the weights of the individual reward shaping functions, per group

        num_reward_fns = [len(reward_group_fns) for _, _, reward_group_fns in config]
        self.split_dims = num_reward_fns

        reward_fn_weights = tensor([weight for _, _, reward_group_fns in config for _, weight in reward_group_fns])
        self.register_buffer('reward_weights', reward_fn_weights)

        self.reward_fns = [reward_fn for _, _, reward_group_fns in config for reward_fn, _ in reward_group_fns]

    def forward(
        self,
        state: State
    ):
        assert isinstance(state, State)

        rewards = tensor([reward_fn(state) for reward_fn in self.reward_fns])

        weighted_rewards = rewards * self.reward_weights

        weighted_rewards_by_groups = weighted_rewards.split(self.split_dims)

        rewards = stack([reward_group.sum() for reward_group in weighted_rewards_by_groups])

        return rewards

# === networks ===

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

        actions = gumbel_sample(logits, dim = -1)

        log_probs = get_log_prob(prob, actions, is_prob = True)

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

        if not sample:
            prob = logits.softmax(dim = -1)
            return prob

        actions = gumbel_sample(logits, dim = -1)

        log_prob = get_log_prob(logits, actions)

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
