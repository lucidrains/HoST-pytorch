from __future__ import annotations

from functools import partial
from typing import Iterable, NamedTuple, Callable
from pydantic import BaseModel, TypeAdapter

from pathlib import Path
from dataclasses import dataclass, asdict
from collections import namedtuple

from tqdm import tqdm

import torch
from torch import nn, Tensor, tensor, cat, stack
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Module, ModuleList, Linear
from torch.utils.data import DataLoader, TensorDataset

from hl_gauss_pytorch import HLGaussLoss

import einx
from einops import repeat, rearrange, reduce, pack
from einops.layers.torch import Rearrange, Reduce, EinMix as Mix

from assoc_scan import AssocScan

from host_pytorch.tensor_typing import Float, Int, Bool

# constants

INF = float('inf')

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def safe_cat(acc, el, dim = -1):
    if not exists(acc):
        return el

    return cat((acc, el), dim = dim)

# sampling related

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def is_empty(t):
    return t.numel() == 0

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

def calc_target_and_gae(
    rewards: Float['g n'],
    values: Float['g n+1'],
    masks: Bool['n'],
    gamma = 0.99,
    lam = 0.95,
    use_accelerated = None

):
    assert values.shape[-1] == (rewards.shape[-1] + 1)

    use_accelerated = default(use_accelerated, rewards.is_cuda)
    device = rewards.device

    masks = repeat(masks, 'n -> g n', g = rewards.shape[0])

    values, values_next = values[:, :-1], values[:, 1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    gates, delta = gates[..., :, None], delta[..., :, None]

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    gae = scan(gates, delta)

    gae = gae[..., :, 0]

    returns = gae + values

    return returns, gae

# === reward functions === table 6 - they have a mistake where they redefine ankle parallel reward twice

@dataclass
class State:
    head_height: Float['']
    angular_velocity: Float['xyz']
    linear_velocity: Float['xyz']
    orientation: Float['xyz']
    projected_gravity_vector: Float[''] # orientation of robot base
    joint_velocity: Float['d']
    joint_acceleration: Float['d']
    joint_torque: Float['d']
    joint_position: Float['d']
    left_ankle_keypoint_z: Float['d']
    right_ankle_keypoint_z: Float['d']
    left_feet_height: Float['']
    right_feet_height: Float['']
    left_shank_angle: Float['']
    right_shank_angle: Float['']
    left_feet_pos: Float['xyz']
    right_feet_pos: Float['xyz']
    upper_body_posture: Float['d']
    height_base: Float['']
    contact_force: Float['xyz']
    hip_joint_angle_lr: Float['']
    robot_base_angle_q: Float['xyz']
    feet_angle_q: Float['xyz']
    knee_joint_angle_lr: Float['lr']
    shoulder_joint_angle_l: Float['']
    shoulder_joint_angle_r: Float['']
    waist_yaw_joint_angle: Float['']

@dataclass
class HyperParams:
    height_stage1_thres: Float['']  # they divide standing up into 2 phases, by whether the height_base reaches thresholds of stage 1 and stage2
    height_stage2_thres: Float['']
    joint_velocity_abs_limit: Float['d']
    joint_position_PD_target: Float['d']
    joint_position_lower_limit: Float['d']
    joint_position_higher_limit: Float['d']
    upper_body_posture_target: Float['d']
    height_base_target: Float['']
    ankle_parallel_thres: float = 0.05
    joint_power_T: float = 1.
    feet_parallel_min_height_diff: float = 0.02
    feet_distance_thres: float = 0.9
    waist_yaw_joint_angle_thres: float = 1.4
    contact_force_ratio_is_foot_stumble: float = 3.
    max_hip_joint_angle_lr: float = 1.4
    min_hip_joint_angle_lr: float = 0.9
    knee_joint_angle_max_min: tuple[float, float] = (2.85, -0.06)
    shoulder_joint_angle_max_min: tuple[float, float] = (-0.02, 0.02)

# state to one tensor + past actions
# not sure how to treat past actions, so will just keep it separate and pass it into actor and critic as another kwarg

def state_to_tensor(
    state: State
) -> Float['state_dim']:

    state_dict = asdict(state)

    state_features, _ = pack([*state_dict.values()], '*')

    return state_features

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

    distance_margin = torch.where(value < low, low - value, value - high) / margin

    scale = torch.sqrt(-2 * tensor(value_at_margin).log())
    return (-0.5 * (distance_margin * scale).pow(2)).exp()

# task rewards - It specifies the high-level task objectives.

def reward_head_height(state: State, hparam: HyperParams, past_actions = None):
    """ The head of robot head in the world frame """

    return ftol(state.head_height, (1., INF), 1., 0.1)

def reward_base_orientation(state: State, hparam: HyperParams, past_actions = None):
    """ The orientation of the robot base represented by projected gravity vector. """

    θz_base = state.projected_gravity_vector
    return ftol(-θz_base, (0.99, INF), 1., 0.05)

# style rewards - It specifies the style of standing-up motion.

def reward_waist_yaw_deviation(state: State, hparam: HyperParams, past_actions = None):
    """ It penalizes the large joint angle of the waist yaw. """

    return state.waist_yaw_joint_angle > hparam.waist_yaw_joint_angle_thres

def reward_hip_roll_yaw_deviation(state: State, hparam: HyperParams, past_actions = None):
    """ It penalizes the large joint angle of hip roll/yaw joints. """

    hip_joint_angle_lr = state.hip_joint_angle_lr.abs() # may not be absolute operator.. todo: figure out what | means in the paper

    return (
        (hip_joint_angle_lr.amax() > hparam.max_hip_joint_angle_lr) |
        (hip_joint_angle_lr.amin() < hparam.min_hip_joint_angle_lr)
    )

def reward_knee_deviation(state: State, hparam: HyperParams, past_actions = None):
    """ It penalizes the large joint angle of the knee joints """

    max_thres, min_thres = hparam.knee_joint_angle_max_min
    return ((state.knee_joint_angle_lr.amax() > max_thres) | (state.knee_joint_angle_lr.amin() < min_thres)).any()

def reward_shoulder_roll_deviation(state: State, hparam: HyperParams, past_actions = None):
    """ It penalizes the large joint angle of shoulder roll joint. """

    max_thres, min_thres = hparam.shoulder_joint_angle_max_min
    max_value, min_value = state.shoulder_joint_angle_l.amax(), state.shoulder_joint_angle_r.amin()
    return ((max_value < max_thres) | (min_value > min_thres)).any()

def reward_foot_displacement(state: State, hparam: HyperParams, past_actions = None):
    """ It encourages robot CoM locates in support polygon, inspired by https://ieeexplore.ieee.org/document/1308858 """

    is_past_stage2 = state.height_base > hparam.height_stage2_thres

    robot_base_angle_qxy = state.robot_base_angle_q[:2]
    feet_angle_qxy = state.feet_angle_q[:2]

    return is_past_stage2 * torch.exp((robot_base_angle_qxy - feet_angle_qxy).norm().pow(2).clamp(min = 0.3).mul(-2))

def reward_ankle_parallel(state: State, hparam: HyperParams, past_actions = None):
    """ It encourages the ankles to be parallel to the ground via ankle keypoints. """

    left_qz = state.left_ankle_keypoint_z
    right_qz = state.right_ankle_keypoint_z

    var = lambda t: t.var(dim = -1, unbiased = True)

    ankle_is_parallel = ((var(left_qz) + var(right_qz)) * 0.5) < hparam.ankle_parallel_thres

    return ankle_is_parallel.float()

def reward_foot_distance(state: State, hparam: HyperParams, past_actions = None):
    """ It penalizes a far distance between feet. """

    return (state.left_feet_pos - state.right_feet_pos).norm().pow(2) > hparam.feet_distance_thres

def reward_foot_stumble(state: State, hparam: HyperParams, past_actions = None):
    """ It penalizes a horizontal contact force with the environment. """

    Fxy, Fz = state.contact_force[:-2], state.contact_force[-1]

    return (Fxy > hparam.contact_force_ratio_is_foot_stumble * Fz).any().float()

def reward_shank_orientation(state: State, hparam: HyperParams, past_actions = None):
    """ It encourages the left/right shank to be perpendicular to the ground. """

    is_past_stage1 = (state.height_base > hparam.height_stage1_thres).float()
    θlr = (state.left_shank_angle + state.right_shank_angle) * 0.5

    return ftol(θlr, (0.8, INF), 1., 0.1) * is_past_stage1

def reward_base_angular_velocity(state: State, hparam: HyperParams, past_actions = None):
    """ It encourages low angular velocity of the during rising up. """

    is_past_stage1 = (state.height_base > hparam.height_stage1_thres).float()

    angular_velocity_base = state.angular_velocity[:2]

    return is_past_stage1 * angular_velocity_base.norm().pow(2).mul(-2).exp()

# regularization rewards - It specifies the regulariztaion on standing-up motion.

def reward_joint_acceleration(state: State, hparam: HyperParams, past_actions = None):
    """ It penalizes the high joint accelrations. """

    return state.joint_acceleration.norm().pow(2)

def reward_action_rate(state: State, hparam: HyperParams, past_actions = None):
    """ It penalizes the high changing speed of action. """

    if not exists(past_actions) or len(past_actions) == 1:
        return tensor(0.)

    prev_action, curr_action = past_actions[-2:]
    return (prev_action - curr_action).norm().pow(2)

def reward_smoothness(state: State, hparam: HyperParams, past_actions = None):
    """ It penalizes the discrepancy between consecutive actions. """

    if not exists(past_actions) or len(past_actions) <= 2:
        return tensor(0.)

    prev_prev_action, prev_action, curr_action = past_actions[-3:]
    return (curr_action - 2 * prev_action + prev_prev_action).norm().pow(2)

def reward_torques(state: State, hparam: HyperParams, past_actions = None):
    """ It penalizes the high joint torques. """

    return state.joint_torque.norm().pow(2)

def reward_joint_power(state: State, hparam: HyperParams, past_actions = None): # not sure what T is
    """ It penalizes the high joint power """

    power = state.joint_torque * state.joint_velocity
    return power.abs().pow(hparam.joint_power_T).sum() # just use a sum for now, unsure how to make a scalar from equation

def reward_joint_velocity(state: State, hparam: HyperParams, past_actions = None):
    """ It penalizes the high joint velocity. """

    return state.joint_velocity.norm().pow(2)

def reward_joint_tracking_error(state: State, hparam: HyperParams, past_actions = None):
    """ It penalizes the error between PD target (Eq. (1)) and actual joint position. """

    return (state.joint_position - hparam.joint_position_PD_target).norm(dim = -1) ** 2

def reward_joint_pos_limits(state: State, hparam: HyperParams, past_actions = None):
    """ It penalizes the joint position that beyond limits. """

    pos = state.joint_position
    low_limit, high_limit = hparam.joint_position_lower_limit, hparam.joint_position_higher_limit

    return ((pos - low_limit).clip(-INF, 0) + (pos - high_limit).clip(0, INF)).sum()

def reward_joint_vel_limits(state: State, hparam: HyperParams, past_actions = None):
    """ It penalizes the joint velocity that beyond limits. """

    return (state.joint_velocity.abs() - hparam.joint_velocity_abs_limit).clip(0., INF).sum()

# post task reward - It specifies the desired behaviors after a successful standing up.

def reward_base_angular_velocity(state: State, hparam: HyperParams, past_actions = None):
    """ It encourages low angular velocity of robot base after standing up. """

    is_past_stage2 = state.height_base > hparam.height_stage2_thres

    angular_velocity_base = state.angular_velocity[:2]

    return is_past_stage2 * angular_velocity_base.norm().mul(-2).exp()

def reward_base_linear_velocity(state: State, hparam: HyperParams, past_actions = None):
    """ It encourages low linear velocity of robot base after standing up. """

    is_past_stage2 = state.height_base > hparam.height_stage2_thres

    linear_velocity_base = state.linear_velocity[:2]
    return is_past_stage2 * linear_velocity_base.norm().mul(-2).exp()

def reward_base_orientation(state: State, hparam: HyperParams, past_actions = None):
    """ It encourages the robot base to be perpendicular to the ground. """

    is_past_stage2 = state.height_base > hparam.height_stage2_thres

    orientation_base = state.orientation[:2]
    return is_past_stage2 * orientation_base.norm().mul(-2).exp()

def reward_base_height(state: State, hparam: HyperParams, past_actions = None):
    """ It encourages the robot base to reach a target height. """

    is_past_stage2 = state.height_base > hparam.height_stage2_thres

    return is_past_stage2 * (state.height_base - hparam.height_base_target).norm().pow(2).mul(-20).exp()

def reward_upper_body_posture(state: State, hparam: HyperParams, past_actions = None):
    """ It encourages the robot to track a target upper body postures. """

    is_past_stage2 = state.height_base > hparam.height_stage2_thres

    return is_past_stage2 * (state.upper_body_posture - hparam.upper_body_posture_target).norm().mul(-1.).pow(2)

def reward_feet_parallel(state: State, hparam: HyperParams, past_actions = None):
    """ In encourages the feet to be parallel to each other. """

    is_past_stage2 = state.height_base > hparam.height_stage2_thres

    return is_past_stage2 * (state.left_feet_height - state.right_feet_height).abs().clamp(min = hparam.feet_parallel_min_height_diff).mul(-20.)

# reward config with all the weights

class RewardFunctionAndWeight(NamedTuple):
    function: Callable
    weight: float

class RewardGroup(NamedTuple):
    name: str
    weight: float
    reward: list[RewardFunctionAndWeight]

def validate_reward_config_(config):
    adapter = TypeAdapter(list[RewardGroup])
    adapter.validate_python(config)

DEFAULT_REWARD_CONFIG = [
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
        config: list[RewardGroup] = DEFAULT_REWARD_CONFIG,
        critics_kwargs: dict = dict(),
        reward_hparams: HyperParams | None = None
    ):
        super().__init__()

        # use pydantic for validating the config and then store

        validate_reward_config_(config)
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

        # default reward hyperparams, but can be overridden

        if isinstance(reward_hparams, dict):
            reward_hparams = HyperParams(**reward_hparams)

        self.reward_hparams = reward_hparams

    def forward(
        self,
        state: State,
        reward_hparams: HyperParams | None = None
    ):
        assert isinstance(state, State)

        reward_hparams = default(reward_hparams, self.reward_hparams)

        assert exists(reward_hparams)

        rewards = tensor([reward_fn(state, reward_hparams) for reward_fn in self.reward_fns])

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

# module for handling variable length past actions

class PastActionsNet(Module):
    def __init__(
        self,
        num_actions,
        dim_action_embed,
        past_action_conv_kernel
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Embedding(num_actions, dim_action_embed),
            Rearrange('b na da -> b da na'),
            nn.Conv1d(dim_action_embed, dim_action_embed, past_action_conv_kernel, padding = past_action_conv_kernel // 2),
            nn.ReLU(),
        )

        self.null_action_embed = nn.Parameter(torch.zeros(dim_action_embed))

    def forward(
        self,
        past_actions
    ):
        is_padding = past_actions < 0
        past_actions = past_actions.masked_fill(is_padding, 0)

        action_embed = self.net(past_actions)

        action_embed = action_embed.masked_fill(is_padding[..., None, :], 0.)
        return reduce(action_embed, '... na -> ...', 'sum')

# actor

class Actor(Module):
    def __init__(
        self,
        num_actions,
        dims: tuple[int, ...] = (512, 256, 128),
        eps_clip = 0.2,
        beta_s = .01,
        dim_action_embed = 4,
        past_action_conv_kernel = 3,
    ):
        super().__init__()
        assert not divisible_by(past_action_conv_kernel, 2)

        first_state_dim, *dims = dims

        # embedding past actions + a simple depthwise conv, to account for action rate / action smoothness rewards

        self.num_actions = num_actions

        self.past_actions_net = PastActionsNet(
            num_actions,
            dim_action_embed,
            past_action_conv_kernel
        )

        # backbone mlp

        first_state_dim += dim_action_embed

        dims = (first_state_dim, *dims, num_actions)

        self.net = MLP(*dims)

        # ppo loss related

        self.eps_clip = eps_clip
        self.beta_s = beta_s

    def forward_net(
        self,
        state,
        past_actions: Int['b na'] | None = None
    ):

        no_batch = state.ndim == 1

        if not exists(past_actions):
            batch, device = state.shape[0], state.device
            past_actions = torch.full((batch, 1), -1, device = device, dtype = torch.long)

        action_embed = self.past_actions_net(past_actions)
        state = cat((state, action_embed), dim = -1)

        logits = self.net(state)

        if no_batch:
            logits = rearrange(logits, '1 ... -> ...')

        return logits

    def forward_for_loss(
        self,
        state,
        actions,
        old_log_probs,
        advantages,
        past_actions: Int['b na'] | None = None
    ):
        clip = self.eps_clip

        logits = self.forward_net(state, past_actions)

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
        state: Float['d'] | Float['b d'],
        past_actions: Int['b na'] | Int['na'] | None = None,
        sample = False,
        sample_return_log_prob = True
    ):
        if state.ndim == 1:
            state = rearrange(state, 'd -> 1 d')

            if exists(past_actions):
                past_actions = rearrange(past_actions, '... -> 1 ...')

        logits = self.forward_net(state, past_actions)

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
        *,
        num_actions: int,
        dims: tuple[int, ...] = (512, 256),
        num_critics = 4,
        dim_action_embed = 16,
        past_action_conv_kernel = 3
    ):
        super().__init__()

        first_dim, *dims = dims

        # critics can see the past actions as well

        self.past_actions_net = PastActionsNet(
            num_actions,
            dim_action_embed,
            past_action_conv_kernel
        )

        first_dim += dim_action_embed

        # grouped mlp for the grouped critic scheme from boston university paper

        dims = (first_dim, *dims, 1)

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
        state: Float['d'] | Float['b d'],
        rewards: Float['b g'] | None = None,
        past_actions: Int['b na'] | None = None,
    ):
        no_batch = state.ndim == 1

        if no_batch:
            state = rearrange(state, 'd -> 1 d')
            if exists(past_actions):
                past_actions = rearrange(past_actions, '... -> 1 ...')

        if not exists(past_actions):
            batch, device = state.shape[0], state.device
            past_actions = torch.full((batch, 1), -1, device = device, dtype = torch.long)

        past_actions_embed = self.past_actions_net(past_actions)

        state = cat((state, past_actions_embed), dim = -1)

        values = self.mlps(state)
        values = rearrange(values, '... 1 -> ...')

        if no_batch:
            values = rearrange(values, '1 ... -> ...')

        if not exists(rewards):
            return values

        return F.mse_loss(rewards, values)

# hierarchical controller

class HierarchicalController(Module):
    def __init__(
        self,
        dim_state,
        actors: list[Actor]
    ):
        super().__init__()

        self.actors = ModuleList(actors)

        self.gates = nn.Sequential(
            nn.Linear(dim_state, len(actors)),
            nn.Softmax(dim = -1)
        )

    def forward(
        self,
        state
    ):
        gates = self.to_gates(state)
        raise NotImplementedError

# memories

Memory = namedtuple('Memory', [
    'state',
    'actions',
    'old_log_probs',
    'rewards',
    'values',
    'done'
])

Memories = namedtuple('Memories', [
    'memories',
    'next_value'
])

# agent - consisting of actor and critic

class Agent(Module):
    def __init__(
        self,
        *,
        actor: dict,
        critics: dict,
        num_past_actions = 3,
        actor_lr = 1e-4,
        critics_lr = 1e-4,
        actor_optim_kwargs: dict = dict(),
        critics_optim_kwargs: dict = dict(),
        optim_klass = Adam,
        reward_config: list[RewardGroup] = DEFAULT_REWARD_CONFIG,
        reward_hparams: HyperParams | dict | None = None,
        num_episodes = 5,
        max_episode_timesteps = 100,
        gae_gamma = 0.99,
        gae_lam = 0.95,
        calc_gae_use_accelerated = None,
        epochs = 2,
        batch_size = 16
    ):
        super().__init__()

        actor = Actor(**actor)

        reward_shaper = RewardShapingWrapper(
            reward_config,
            critics_kwargs = {**critics, 'num_actions': actor.num_actions},
            reward_hparams = reward_hparams
        )

        self.actor = actor
        self.critics = reward_shaper.critics

        self.state_to_reward = reward_shaper

        self.actor_optim = optim_klass(self.actor.parameters(), lr = actor_lr, **actor_optim_kwargs)
        self.critics_optim = optim_klass(self.critics.parameters(), lr = critics_lr, **critics_optim_kwargs)

        self.num_past_actions = num_past_actions

        self.epochs = epochs
        self.batch_size = batch_size

        # episodes with environment

        self.num_episodes = num_episodes
        self.max_episode_timesteps = max_episode_timesteps

        # calculating gae

        self.calc_target_and_gae = partial(
            calc_target_and_gae,
            gamma = gae_gamma,
            lam = gae_lam,
            use_accelerated = calc_gae_use_accelerated
        )

    def save(
        self,
        path: str | Path,
        overwrite = False
    ):
        if isinstance(path, str):
            path = Path(path)

        assert overwrite or not path.exists(), f'{str(path)} already exists'

        pkg = dict(
            actor = self.actor.state_dict(),
            critics = self.critics.state_dict(),
            actor_optim = self.actor_optim.state_dict(),
            critics_optim = self.critics_optim.state_dict(),
        )

        torch.save(pkg, str(path))

    def load(
        self,
        path: str | Path,
        load_optim = True
    ):
        if isinstance(path, str):
            path = Path(path)

        assert path.exists()

        pkg = torch.load(str(path), weights_only = True)

        self.actor.load_state_dict(pkg['actor'])
        self.critics.load_state_dict(pkg['critics'])

        if not load_optim:
            return

        if 'actor_optim' in pkg:
            self.actor_optim.load_state_dict(pkg['actor_optim'])

        if 'critics_optim' in pkg:
            self.critics_optim.load_state_dict(pkg['critics_optim'])

    def learn(
        self,
        memories_and_next_value: Memories,
    ):
        memories, next_value = memories_and_next_value

        # stack all the memories across time

        (
            states,
            actions,
            log_probs,
            rewards,
            values,
            dones
        ) = map(stack, zip(*memories))

        # calculating returns and generalized advantage estimate

        rewards = rearrange(rewards, '... g -> g ...')

        values_with_next, _ = pack((values, next_value), '* g')
        values_with_next = rearrange(values_with_next, '... g -> g ...')

        returns, gae = self.calc_target_and_gae(rewards, values_with_next, dones)

        returns = rearrange(returns, 'g ... -> ... g')
        gae = rearrange(gae, 'g ... -> ... g')

        # prepare past actions

        episode_num = dones.cumsum(dim = -1)
        episode_num = F.pad(episode_num, (1, -1), value = 0)

        actions = F.pad(actions, (0, 0, self.num_past_actions, 0), value = -1)

        actions_with_past = actions.unfold(0, self.num_past_actions + 1, 1)

        episode_num_with_past = F.pad(episode_num, (self.num_past_actions, 0), value = 0)
        episode_num_with_past = episode_num_with_past.unfold(0, self.num_past_actions + 1, 1)

        actions_with_past = rearrange(actions_with_past, 't na past -> t past na')

        is_episode_overlap = episode_num[..., None] != episode_num_with_past

        actions_with_past.masked_fill_(is_episode_overlap[..., None], -1)

        # ready experience data

        dataset = TensorDataset(states, actions_with_past, log_probs, returns, values)

        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        # update actor and critic

        self.actor.train()
        self.critics.train()

        for _ in tqdm(range(self.epochs), 'training epoch'):
            for (
                states,
                actions_with_past,
                log_probs,
                returns,
                values
            ) in dataloader:

                past_actions, actions = actions_with_past[:, :-1], actions_with_past[:, -1]

                past_actions = rearrange(past_actions, '... 1 -> ...')

                critics_loss = self.critics(states, past_actions = past_actions, rewards = returns)
                critics_loss.backward()

                self.critics_optim.step()
                self.critics_optim.zero_grad()

                advantages = self.critics.calc_advantages(values, rewards = returns)

                actor_loss = self.actor.forward_for_loss(states, actions, log_probs, advantages, past_actions = past_actions)
                actor_loss.backward()

                self.actor_optim.step()
                self.actor_optim.zero_grad()

    def forward(
        self,
        env,
        reward_hparams: HyperParams | None = None

    ) -> Memories:

        memories = []

        for _ in tqdm(range(self.num_episodes), desc = 'episodes'):

            timestep = 0
            state = env.reset()

            past_actions = None

            while timestep < self.max_episode_timesteps:

                # State -> Float['state_dim']

                state_tensor = state_to_tensor(state)

                # done would be whether it is the last time step of the episode
                # but can also have some termination condition based on the robot's internal state

                done = timestep == (self.max_episode_timesteps - 1)
                done = tensor(done, device = state_tensor.device)

                # rewards are derived from the state

                rewards = self.state_to_reward(state, reward_hparams)

                # sample the action from the actor
                # and get the values from the critics

                actor_critic_past_actions = past_actions[-self.num_past_actions:] if exists(past_actions) else None

                with torch.no_grad():
                    self.actor.eval()
                    self.critics.eval()

                    actions, log_probs = self.actor(state_tensor, sample = True, past_actions = actor_critic_past_actions)

                    values = self.critics(state_tensor, past_actions = actor_critic_past_actions)

                # store past actions

                past_actions = safe_cat(past_actions, actions)

                # store memories

                memory = Memory(
                    state = state_tensor,
                    actions = actions,
                    old_log_probs = log_probs,
                    rewards = rewards,
                    values = values,
                    done = done
                )

                memories.append(memory)

                # next state

                state = env(actions)

                # increment timestep

                timestep += 1

        with torch.no_grad():
            self.critics.eval()

            state_tensor = state_to_tensor(state)
            next_value = self.critics(state_tensor)

        return Memories(memories, next_value)
