from __future__ import annotations

from random import choice
from functools import partial
from itertools import zip_longest

from types import SimpleNamespace
from typing import Iterable, NamedTuple, Callable
from pydantic import BaseModel, TypeAdapter

from pathlib import Path
from dataclasses import dataclass, asdict
from collections import namedtuple, OrderedDict

from tqdm import tqdm

import torch
from torch.nested import nested_tensor
from torch import nn, Tensor, tensor, is_tensor, cat, stack
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Module, ModuleList, Linear
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from hl_gauss_pytorch import HLGaussLoss

from assoc_scan import AssocScan

from host_pytorch.tensor_typing import Float, Int, Bool

# https://arxiv.org/html/2503.19037v1

from evolutionary_policy_optimization import LatentGenePool

# einstein notation related

import einx
from einops import repeat, rearrange, reduce, pack
from einops.layers.torch import Rearrange, Reduce, EinMix as Mix

# b - batch
# t - time
# a - actions
# d - feature dimension
# past - past actions

# constants

INF = float('inf')

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(arr):
    return arr[0]

def xnor(x, y):
    return not (x ^ y)

def divisible_by(num, den):
    return (num % den) == 0

def cast_tuple(t):
    return (t,) if not isinstance(t, tuple) else t

def is_unique_els(arr):
    assert len(arr) == len(set(arr))

def safe_cat(acc, el, dim = -1):
    if not exists(acc):
        return el

    return cat((acc, el), dim = dim)

# sampling related

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def is_empty(t):
    return t.numel() == 0

def exclusive_cumsum(t):
    t = t.cumsum(dim = -1)
    return F.pad(t, (1, -1), value = 0)

def safe_div(num, den, eps = 1e-8):
    den = den.clamp(min = eps) if is_tensor(den) else max(den, eps)
    return num / den

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def calc_entropy(prob, eps = 1e-20, dim = -1):
    return -(prob * log(prob, eps)).sum(dim = dim)

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

# nested tensor related

def to_nested_tensor_and_inverse(tensors):
    jagged_dims = tuple(t.shape[-1] for t in tensors)

    tensor = cat(tensors, dim = -1)
    batch = tensor.shape[0]

    tensor = rearrange(tensor, 'b n -> (b n)')
    nt = nested_tensor(tensor.split(jagged_dims * batch), layout = torch.jagged, device = tensor.device, requires_grad = True)

    def inverse(t):
        t = cat(t.unbind())
        return rearrange(t, '(b n) -> b n', b = batch)

    return nt, inverse

def nested_sum(t, lens: tuple[int, ...]):
    # needed as backwards not supported for sum with nested tensor

    batch, device = t.shape[0], t.device

    lens = tensor(lens, device = device)
    indices = lens.cumsum(dim = -1) - 1
    indices = repeat(indices, 'n -> b n', b = batch)

    cumsum = t.cumsum(dim = -1)
    sum_at_indices = cumsum.gather(-1, indices)
    sum_at_indices = F.pad(sum_at_indices, (1, 0), value = 0.)

    return sum_at_indices[..., 1:] - sum_at_indices[..., :-1]

def nested_argmax(t, lens: tuple[int, ...]):
    batch, device = t.shape[0], t.device

    t = rearrange(t, 'b nl -> nl b')
    split_tensors = t.split(lens, dim = 0)

    padded_tensors = pad_sequence(split_tensors, batch_first = True, padding_value = max_neg_value(t))
    padded_tensors = rearrange(padded_tensors, 'n l b -> b n l')

    return padded_tensors.argmax(dim = -1)

# generalized advantage estimate

def calc_target_and_gae(
    rewards: Float['g n'],
    values: Float['g n'],
    masks: Bool['n'],
    gamma = 0.99,
    lam = 0.95,
    use_accelerated = None

):
    assert values.shape[-1] == rewards.shape[-1]

    use_accelerated = default(use_accelerated, rewards.is_cuda)
    device = rewards.device

    masks = repeat(masks, 'n -> g n', g = rewards.shape[0])

    values = F.pad(values, (0, 1), value = 0.)
    values, values_next = values[:, :-1], values[:, 1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    gae = scan(gates, delta)

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

# derived states
# they use the robot's height as a way of activating certain rewards
# i think this could eventually be extended to be some learned classifier

def state_is_past_stage_1(state: State, hparam: HyperParams):
    return state.height_base > hparam.height_stage1_thres

def state_is_past_stage_2(state: State, hparam: HyperParams):
    return state.height_base > hparam.height_stage2_thres

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

def reward_head_height(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ The head of robot head in the world frame """

    return ftol(state.head_height, (1., INF), 1., 0.1)

def reward_base_orientation(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ The orientation of the robot base represented by projected gravity vector. """

    θz_base = state.projected_gravity_vector
    return ftol(-θz_base, (0.99, INF), 1., 0.05)

# style rewards - It specifies the style of standing-up motion.

def reward_waist_yaw_deviation(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It penalizes the large joint angle of the waist yaw. """

    return state.waist_yaw_joint_angle > hparam.waist_yaw_joint_angle_thres

def reward_hip_roll_yaw_deviation(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It penalizes the large joint angle of hip roll/yaw joints. """

    hip_joint_angle_lr = state.hip_joint_angle_lr.abs() # may not be absolute operator.. todo: figure out what | means in the paper

    return (
        (hip_joint_angle_lr.amax() > hparam.max_hip_joint_angle_lr) |
        (hip_joint_angle_lr.amin() < hparam.min_hip_joint_angle_lr)
    )

def reward_knee_deviation(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It penalizes the large joint angle of the knee joints """

    max_thres, min_thres = hparam.knee_joint_angle_max_min
    return ((state.knee_joint_angle_lr.amax() > max_thres) | (state.knee_joint_angle_lr.amin() < min_thres)).any()

def reward_shoulder_roll_deviation(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It penalizes the large joint angle of shoulder roll joint. """

    max_thres, min_thres = hparam.shoulder_joint_angle_max_min
    max_value, min_value = state.shoulder_joint_angle_l.amax(), state.shoulder_joint_angle_r.amin()
    return ((max_value < max_thres) | (min_value > min_thres)).any()

def reward_foot_displacement(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It encourages robot CoM locates in support polygon, inspired by https://ieeexplore.ieee.org/document/1308858 """

    robot_base_angle_qxy = state.robot_base_angle_q[:2]
    feet_angle_qxy = state.feet_angle_q[:2]

    return derived_state.past_stage2 * torch.exp((robot_base_angle_qxy - feet_angle_qxy).norm().pow(2).clamp(min = 0.3).mul(-2))

def reward_ankle_parallel(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It encourages the ankles to be parallel to the ground via ankle keypoints. """

    left_qz = state.left_ankle_keypoint_z
    right_qz = state.right_ankle_keypoint_z

    var = lambda t: t.var(dim = -1, unbiased = True)

    ankle_is_parallel = ((var(left_qz) + var(right_qz)) * 0.5) < hparam.ankle_parallel_thres

    return ankle_is_parallel.float()

def reward_foot_distance(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It penalizes a far distance between feet. """

    return (state.left_feet_pos - state.right_feet_pos).norm().pow(2) > hparam.feet_distance_thres

def reward_foot_stumble(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It penalizes a horizontal contact force with the environment. """

    Fxy, Fz = state.contact_force[:-2], state.contact_force[-1]

    return (Fxy > hparam.contact_force_ratio_is_foot_stumble * Fz).any().float()

def reward_shank_orientation(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It encourages the left/right shank to be perpendicular to the ground. """

    θlr = (state.left_shank_angle + state.right_shank_angle) * 0.5

    return ftol(θlr, (0.8, INF), 1., 0.1) * derived_state.past_stage1

def reward_base_angular_velocity(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It encourages low angular velocity of the during rising up. """

    angular_velocity_base = state.angular_velocity[:2]

    return derived_state.past_stage1 * angular_velocity_base.norm().pow(2).mul(-2).exp()

# regularization rewards - It specifies the regulariztaion on standing-up motion.

def reward_joint_acceleration(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It penalizes the high joint accelrations. """

    return state.joint_acceleration.norm().pow(2)

def reward_action_rate(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It penalizes the high changing speed of action. """

    if not exists(past_actions) or len(past_actions) == 1:
        return tensor(0.)

    prev_action, curr_action = past_actions[-2:]
    return (prev_action - curr_action).norm().pow(2)

def reward_smoothness(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It penalizes the discrepancy between consecutive actions. """

    if not exists(past_actions) or len(past_actions) <= 2:
        return tensor(0.)

    prev_prev_action, prev_action, curr_action = past_actions[-3:]
    return (curr_action - 2 * prev_action + prev_prev_action).norm().pow(2)

def reward_torques(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It penalizes the high joint torques. """

    return state.joint_torque.norm().pow(2)

def reward_joint_power(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None): # not sure what T is
    """ It penalizes the high joint power """

    power = state.joint_torque * state.joint_velocity
    return power.abs().pow(hparam.joint_power_T).sum() # just use a sum for now, unsure how to make a scalar from equation

def reward_joint_velocity(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It penalizes the high joint velocity. """

    return state.joint_velocity.norm().pow(2)

def reward_joint_tracking_error(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It penalizes the error between PD target (Eq. (1)) and actual joint position. """

    return (state.joint_position - hparam.joint_position_PD_target).norm(dim = -1) ** 2

def reward_joint_pos_limits(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It penalizes the joint position that beyond limits. """

    pos = state.joint_position
    low_limit, high_limit = hparam.joint_position_lower_limit, hparam.joint_position_higher_limit

    return ((pos - low_limit).clip(-INF, 0) + (pos - high_limit).clip(0, INF)).sum()

def reward_joint_vel_limits(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It penalizes the joint velocity that beyond limits. """

    return (state.joint_velocity.abs() - hparam.joint_velocity_abs_limit).clip(0., INF).sum()

# post task reward - It specifies the desired behaviors after a successful standing up.

def reward_base_angular_velocity(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It encourages low angular velocity of robot base after standing up. """

    angular_velocity_base = state.angular_velocity[:2]

    return derived_state.past_stage2 * angular_velocity_base.norm().mul(-2).exp()

def reward_base_linear_velocity(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It encourages low linear velocity of robot base after standing up. """

    linear_velocity_base = state.linear_velocity[:2]
    return derived_state.past_stage2 * linear_velocity_base.norm().mul(-2).exp()

def reward_base_orientation(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It encourages the robot base to be perpendicular to the ground. """

    orientation_base = state.orientation[:2]
    return derived_state.past_stage2 * orientation_base.norm().mul(-2).exp()

def reward_base_height(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It encourages the robot base to reach a target height. """

    return derived_state.past_stage2 * (state.height_base - hparam.height_base_target).norm().pow(2).mul(-20).exp()

def reward_upper_body_posture(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ It encourages the robot to track a target upper body postures. """

    return derived_state.past_stage2 * (state.upper_body_posture - hparam.upper_body_posture_target).norm().mul(-1.).pow(2)

def reward_feet_parallel(state: State, hparam: HyperParams, derived_state: SimpleNamespace, past_actions = None):
    """ In encourages the feet to be parallel to each other. """

    return derived_state.past_stage2 * (state.left_feet_height - state.right_feet_height).abs().clamp(min = hparam.feet_parallel_min_height_diff).mul(-20.)

# reward config with all the weights

class RewardFunctionAndWeight(NamedTuple):
    function: Callable
    weight: float

class RewardGroup(NamedTuple):
    name: str
    weight: float
    reward: list[RewardFunctionAndWeight]

class DerivedStateConfig(NamedTuple):
    name: str
    function: Callable

class RewardShapingConfig(BaseModel):
    derived_states: list[DerivedStateConfig]
    reward_groups: list[RewardGroup]

def validate_reward_config_and_index_by_name(reward_group_config):
    indexed_config = OrderedDict()

    for group_name, group_weight, reward_functions in reward_group_config:
        if group_name not in indexed_config:
            indexed_config[group_name] = (OrderedDict(), group_weight)

        indexed_group_rewards, _ = indexed_config[group_name]

        for reward_function, weight in reward_functions:
            reward_function_name = reward_function.__name__
            assert reward_function_name not in indexed_group_rewards
            indexed_group_rewards[reward_function_name] = (reward_function, weight)

    return indexed_config

DEFAULT_REWARD_SHAPING_CONFIG = dict(
    derived_states = [
        ('past_stage1', state_is_past_stage_1),
        ('past_stage2', state_is_past_stage_2)
    ],
    reward_groups = [
        ('task', 2.5, [
            (reward_head_height, 1),
            (reward_base_orientation, 1),
        ]),
        ('style', 1., [
            (reward_waist_yaw_deviation, -10),
            (reward_hip_roll_yaw_deviation, -10 / 10),
            (reward_knee_deviation, -0.25 / -10),
            (reward_shoulder_roll_deviation, -2.5),
            (reward_foot_displacement, 2.5 / 2.5),
            (reward_ankle_parallel, 20),
            (reward_foot_distance, -10),
            (reward_foot_stumble, 0 / -25),
            (reward_shank_orientation, 10), # do not understand this 0(G) / -25(PSW)
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
    ],
)

# the reward shaping module

class RewardShapingWrapper(Module):
    def __init__(
        self,
        reward_shape_config: dict = DEFAULT_REWARD_SHAPING_CONFIG,
        critics_kwargs: dict = dict(),
        reward_hparams: HyperParams | None = None
    ):
        super().__init__()

        # use pydantic for validating the config and then store

        reward_shape_config = RewardShapingConfig(**reward_shape_config)

        self.derived_state_config = reward_shape_config.derived_states

        # process reward group config and index by name

        config = validate_reward_config_and_index_by_name(reward_shape_config.reward_groups)
        self.config = config

        # based on the reward group config
        # can instantiate the critics automatically

        num_reward_groups = len(config)
        critics_weights = [reward_group_weight for _, reward_group_weight in config.values()]

        self.critics = Critics(
            critics_weights,
            num_critics = num_reward_groups,
            **critics_kwargs
        )

        # default reward hyperparams, but can be overridden

        if isinstance(reward_hparams, dict):
            reward_hparams = HyperParams(**reward_hparams)

        self.reward_hparams = reward_hparams

        # initialize reward shaper based on config

        self.init_reward_shaper(config)

    def init_reward_shaper(
        self,
        config
    ):
        # store the weights of the individual reward shaping functions, per group

        num_reward_fns = [len(reward_fns) for reward_fns, weight in config.values()]

        self.split_dims = num_reward_fns

        reward_fn_weights = tensor([weight for reward_group_fns, _ in config.values() for _, weight in reward_group_fns.values()])
        self.register_buffer('reward_weights', reward_fn_weights)

        self.reward_fns = [reward_fn for reward_group_fns, _ in config.values() for reward_fn, _ in reward_group_fns.values()]

    def add_reward_function_(
        self,
        reward_fn: Callable,
        group_name: str,
        weight: float
    ):
        assert group_name in self.config

        indexed_group_rewards, _ = self.config[group_name]

        reward_fn_name = reward_fn.__name__
        assert reward_fn_name not in indexed_group_rewards, f'reward function names must be unique'

        indexed_group_rewards[reward_fn_name] = (reward_fn, weight)

        self.init_reward_shaper(self.config)

    def delete_reward_function_(
        self,
        group_name: str,
        reward_fn_name: str
    ):
        assert group_name in self.config

        indexed_group_rewards, _ = self.config[group_name]

        assert reward_fn_name in indexed_group_rewards

        del indexed_group_rewards[reward_fn_name]

        self.init_reward_shaper(self.config)

    def forward(
        self,
        state: State,
        reward_hparams: HyperParams | None = None
    ):
        assert isinstance(state, State)

        reward_hparams = default(reward_hparams, self.reward_hparams)

        assert exists(reward_hparams)

        # compute all derived states only once, may eventualy involve small trained classifiers

        derived_state = SimpleNamespace(**{name: fn(state, reward_hparams) for name, fn in self.derived_state_config})

        # compute all rewards from all functions

        rewards = tensor([reward_fn(state, reward_hparams, derived_state) for reward_fn in self.reward_fns])

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
        x,
        cond = None
    ):

        cond = default(cond, (None,))

        for ind, (layer, layer_cond) in enumerate(zip_longest(self.layers, cond), start = 1):
            is_last = ind == len(self.layers)

            x = layer(x)

            if exists(layer_cond):
                x = x * layer_cond

            if not is_last:
                x = F.silu(x)

        return x

# module for handling variable length past actions

class PastActionsNet(Module):
    def __init__(
        self,
        num_actions: int | tuple[int, ...],
        dim_action_embed,
        past_action_conv_kernel
    ):
        super().__init__()

        num_actions = cast_tuple(num_actions)

        total_actions = sum(num_actions)

        self.num_actions = num_actions
        dim_all_actions = dim_action_embed * len(num_actions)

        self.dim_all_actions = dim_all_actions

        # network that does a convolution across previous actions before pooling

        self.net = nn.Sequential(
            nn.Embedding(total_actions, dim_action_embed),
            Rearrange('b past a d -> b (a d) past'),
            nn.Conv1d(dim_all_actions, dim_all_actions, past_action_conv_kernel, padding = past_action_conv_kernel // 2),
            nn.ReLU(),
        )

        self.null_action_embed = nn.Parameter(torch.zeros(dim_all_actions))

        offset = exclusive_cumsum(tensor(num_actions))
        self.register_buffer('action_offset', offset)

    def forward(
        self,
        past_actions # Int[b past a]
    ):

        is_padding = past_actions < 0

        past_actions = past_actions + self.action_offset
        past_actions = past_actions.masked_fill(is_padding, 0)

        action_embed = self.net(past_actions)
        all_actions_are_padding = is_padding.all(dim = -1)

        action_embed = action_embed.masked_fill(all_actions_are_padding[..., None, :], 0.)
        return reduce(action_embed, '... past -> ...', 'sum')

# actor

class Actor(Module):
    def __init__(
        self,
        num_actions: int | tuple[int, ...],
        dims: tuple[int, ...] = (512, 256, 128),
        eps_clip = 0.2,
        entropy_weight = .01,
        dim_action_embed = 4,
        dim_latent = None,
        past_action_conv_kernel = 3,
        norm_advantages = True
    ):
        super().__init__()
        assert not divisible_by(past_action_conv_kernel, 2)

        first_state_dim, *dims = dims

        # embedding past actions + a simple depthwise conv, to account for action rate / action smoothness rewards

        num_actions = cast_tuple(num_actions)

        self.num_actions = num_actions
        self.num_action_sets = len(num_actions)

        self.past_actions_net = PastActionsNet(
            num_actions,
            dim_action_embed,
            past_action_conv_kernel
        )

        offset = exclusive_cumsum(tensor(num_actions))
        self.register_buffer('action_offset', offset)

        # latents

        has_latent = exists(dim_latent)

        self.has_latent = has_latent
        self.latent_output_split_dims = dims

        if has_latent:
            self.to_conditioning = nn.Sequential(
                nn.Linear(dim_latent, sum(dims)),
                nn.SiLU()
            )

        # backbone mlp

        first_state_dim += self.past_actions_net.dim_all_actions

        dims = (first_state_dim, *dims, sum(num_actions))

        self.net = MLP(*dims)

        # ppo loss related

        self.eps_clip = eps_clip
        self.entropy_weight = entropy_weight
        self.norm_advantages = norm_advantages

    def forward_net(
        self,
        state,
        past_actions: Int['b past a'] | None = None,
        latents: Float['d'] | Float['b d'] | None = None,
    ):

        assert xnor(self.has_latent, exists(latents))

        mlp_conds = None

        if self.has_latent:
            if latents.ndim == 1:
                latents = repeat(latents, 'd -> b d', b = state.shape[0])

            mlp_conds = self.to_conditioning(latents).split(self.latent_output_split_dims, dim = -1)

        if not exists(past_actions):
            batch, device = state.shape[0], state.device
            past_actions = torch.full((batch, 1, self.num_action_sets), -1, device = device, dtype = torch.long)

        action_embed = self.past_actions_net(past_actions)

        state = cat((state, action_embed), dim = -1)

        logits = self.net(state, cond = mlp_conds)

        return logits

    def forward_for_loss(
        self,
        state,
        actions,
        old_log_probs,
        advantages,
        latents = None,
        past_actions: Int['b past a'] | None = None,
        norm_eps = 1e-5
    ):
        batch = advantages.shape[0]

        clip = self.eps_clip

        if self.norm_advantages:
            advantages = F.layer_norm(advantages, (batch,), eps = norm_eps)

        advantages = rearrange(advantages, 'b -> b 1')

        logits = self.forward_net(state, past_actions, latents = latents)

        logits_per_action_set = logits.split(self.num_actions, dim = -1)

        nested_logits, inverse_nested = to_nested_tensor_and_inverse(logits_per_action_set)

        nested_probs = nested_logits.softmax(dim = -1)

        probs = inverse_nested(nested_probs)

        # entropy

        entropies = nested_sum(-probs * log(probs), self.num_actions)

        indices = actions + self.action_offset

        log_probs = log(probs).gather(-1, indices)

        ratios = (log_probs - old_log_probs).exp()

        # classic clipped surrogate objective from ppo

        surr1 = ratios * advantages
        surr2 = ratios.clamp(1. - clip, 1. + clip) * advantages
        loss = -torch.min(surr1, surr2) - self.entropy_weight * entropies

        loss = reduce(loss, 'b a -> b', 'sum')
        return loss.mean()

    def forward(
        self,
        state: Float['d'] | Float['b d'],
        latents = None,
        past_actions: Int['b past a'] | Int['past a'] | None = None,
        sample = False,
        sample_return_log_prob = True,
        sample_temperature = 1.,
        use_nested_argmax = True
    ):
        no_batch = state.ndim == 1

        if no_batch:
            state = rearrange(state, 'd -> 1 d')

        if exists(past_actions) and past_actions.ndim == 2:
            past_actions = rearrange(past_actions, '... -> 1 ...')

        logits = self.forward_net(state, past_actions, latents = latents)

        # split logits for multiple actions

        logits_per_action_set = logits.split(self.num_actions, dim = -1)

        nested_logits, inverse_nested = to_nested_tensor_and_inverse(logits_per_action_set)

        nested_probs = nested_logits.softmax(dim = -1)

        probs = inverse_nested(nested_probs)

        if not sample:
            return probs

        noised_logits = safe_div(logits, sample_temperature) + gumbel_noise(logits)

        if use_nested_argmax:
            actions = nested_argmax(noised_logits, self.num_actions)
        else:
            actions = tuple(t.argmax(dim = -1) for t in noised_logits.split(self.num_actions, dim = -1))
            actions = stack(actions, dim = -1)

        indices = actions + self.action_offset

        log_probs = log(probs).gather(-1, indices)

        if no_batch:
            log_probs = rearrange(log_probs, '1 ... -> ...')
            actions = rearrange(actions, '1 ... -> ...')

        if not sample_return_log_prob:
            return actions

        return actions, log_probs

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
        x,
        cond = None
    ):

        cond = default(cond, (None,))

        for ind, (layer, layer_cond) in enumerate(zip_longest(self.layers, cond), start = 1):
            is_last = ind == len(self.layers)

            x = layer(x)

            if exists(layer_cond):
                x = x * layer_cond

            if not is_last:
                x = F.silu(x)

        return x

# critics

class Critics(Module):
    def __init__(
        self,
        weights: tuple[float, ...],
        *,
        num_actions: int | tuple[int, ...],
        dims: tuple[int, ...] = (512, 256),
        num_critics = 4,
        dim_action_embed = 16,
        dim_latent = None,
        past_action_conv_kernel = 3,
        value_eps_clip = 0.4
    ):
        super().__init__()

        first_dim, *dims = dims

        num_actions = cast_tuple(num_actions)

        self.num_action_sets = len(num_actions)

        # latents

        has_latent = exists(dim_latent)

        self.has_latent = has_latent
        self.latent_output_split_dims = dims

        if has_latent:
            self.to_conditioning = nn.Sequential(
                nn.Linear(dim_latent, num_critics * sum(dims)),
                nn.SiLU(),
                Rearrange('... (num_critics d) -> ... num_critics d', num_critics = num_critics)
            )

        # critics can see the past actions as well

        self.past_actions_net = PastActionsNet(
            num_actions,
            dim_action_embed,
            past_action_conv_kernel
        )

        first_dim += self.past_actions_net.dim_all_actions

        # grouped mlp for the grouped critic scheme from boston university paper

        dims = (first_dim, *dims, 1)

        self.mlps = GroupedMLP(*dims, num_mlps = num_critics)

        assert len(weights) == num_critics
        self.register_buffer('weights', tensor(weights))

        # value clip eps

        self.value_eps_clip = value_eps_clip

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

    def forward_for_loss(
        self,
        state: Float['d'] | Float['b d'],
        old_values: Float['b g'],
        rewards: Float['b g'],
        past_actions: Int['b past a'] | Int['past a'] | None = None,
        latents: Float['d'] | Float['b d'] | None = None,
    ):
        eps = self.value_eps_clip
        values = self.forward(state = state, past_actions = past_actions, latents = latents)

        clipped_value = old_values + (values - old_values).clamp(-eps, eps)

        loss = F.mse_loss(values, rewards, reduction = 'none')
        clipped_loss = F.mse_loss(clipped_value, rewards, reduction = 'none')

        return torch.max(loss, clipped_loss).mean()

    def forward(
        self,
        state: Float['d'] | Float['b d'],
        past_actions: Int['b past a'] | Int['past a'] | None = None,
        latents: Float['d'] | Float['b d'] | None = None
    ):

        # handle no batch

        no_batch = state.ndim == 1

        if no_batch:
            state = rearrange(state, 'd -> 1 d')
            if exists(past_actions):
                past_actions = rearrange(past_actions, '... -> 1 ...')

        # latents

        assert xnor(self.has_latent, exists(latents))

        mlp_conds = None

        if self.has_latent:
            if latents.ndim == 1:
                latents = repeat(latents, 'd -> b d', b = state.shape[0])

            mlp_conds = self.to_conditioning(latents).split(self.latent_output_split_dims, dim = -1)

        # handle past actions

        if not exists(past_actions):
            batch, device = state.shape[0], state.device
            past_actions = torch.full((batch, 1, self.num_action_sets), -1, device = device, dtype = torch.long)

        past_actions_embed = self.past_actions_net(past_actions)

        state = cat((state, past_actions_embed), dim = -1)

        values = self.mlps(state, cond = mlp_conds)
        values = rearrange(values, '... 1 -> ...')

        if no_batch:
            values = rearrange(values, '1 ... -> ...')

        return values

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
    'learnable',
    'state',
    'actions',
    'old_log_probs',
    'rewards',
    'values',
    'done'
])

# agent - consisting of actor and critic

class Agent(Module):
    def __init__(
        self,
        *,
        num_actions: int | tuple[int, ...],
        actor: dict,
        critics: dict,
        num_past_actions = 3,
        actor_lr = 1e-4,
        critics_lr = 1e-4,
        actor_weight_decay = 1e-3,
        critics_weight_decay = 1e-3,
        actor_optim_kwargs: dict = dict(),
        critics_optim_kwargs: dict = dict(),
        optim_klass = Adam,
        reward_config: dict = DEFAULT_REWARD_SHAPING_CONFIG,
        reward_hparams: HyperParams | dict | None = None,
        num_episodes = 5,
        max_episode_timesteps = 100,
        gae_gamma = 0.99,
        gae_lam = 0.95,
        calc_gae_use_accelerated = None,
        epochs = 2,
        batch_size = 16,
        env_hparams: list | None = None
    ):
        super().__init__()

        assert 'num_actions' not in actor
        assert 'num_actions' not in critics

        num_actions_kwarg = dict(num_actions = num_actions)

        actor = Actor(**actor, **num_actions_kwarg)

        reward_shaper = RewardShapingWrapper(
            reward_config,
            critics_kwargs = {**critics, **num_actions_kwarg},
            reward_hparams = reward_hparams
        )

        self.actor = actor
        self.critics = reward_shaper.critics

        self.state_to_reward = reward_shaper

        self.actor_optim = optim_klass(self.actor.parameters(), lr = actor_lr, weight_decay = actor_weight_decay, **actor_optim_kwargs)
        self.critics_optim = optim_klass(self.critics.parameters(), lr = critics_lr, weight_decay = critics_weight_decay, **critics_optim_kwargs)

        self.num_past_actions = num_past_actions

        self.epochs = epochs
        self.batch_size = batch_size

        # episodes with environment

        self.num_episodes = num_episodes
        self.max_episode_timesteps = max_episode_timesteps

        # domain randomization related

        self.env_hparams = env_hparams

        # calculating gae

        self.calc_target_and_gae = partial(
            calc_target_and_gae,
            gamma = gae_gamma,
            lam = gae_lam,
            use_accelerated = calc_gae_use_accelerated
        )

    def add_reward_function_(
        self,
        reward_fn: Callable,
        group_name: str,
        weight: float
    ):
        self.state_to_reward.add_reward_function_(
            reward_fn,
            group_name,
            weight
        )

    def delete_reward_function_(
        self,
        group_name: str,
        reward_fn_name: str
    ):
        self.state_to_reward.delete_reward_function_(
            group_name,
            reward_fn_name
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
        memories: list[Memory]
    ):
        # stack all the memories across time

        (
            learnable,
            states,
            actions,
            log_probs,
            rewards,
            values,
            dones
        ) = map(stack, zip(*memories))

        # calculating returns and generalized advantage estimate

        rewards = rearrange(rewards, '... g -> g ...')
        values_for_gae = rearrange(values, 'n g -> g n')

        boundaries = 1. - dones.float()

        returns, gae = self.calc_target_and_gae(rewards, values_for_gae, boundaries)

        returns = rearrange(returns, 'g ... -> ... g').detach()
        gae = rearrange(gae, 'g ... -> ... g').detach()

        # prepare past actions

        episode_num = exclusive_cumsum(dones)

        actions = F.pad(actions, (0, 0, self.num_past_actions, 0), value = -1)

        actions_with_past = actions.unfold(0, self.num_past_actions + 1, 1)

        episode_num_with_past = F.pad(episode_num, (self.num_past_actions, 0), value = 0)
        episode_num_with_past = episode_num_with_past.unfold(0, self.num_past_actions + 1, 1)

        actions_with_past = rearrange(actions_with_past, 't a past -> t past a')

        is_episode_overlap = episode_num[..., None] != episode_num_with_past

        actions_with_past = actions_with_past.masked_fill(is_episode_overlap[..., None], -1)

        # ready experience data

        data = (states, actions_with_past, log_probs, returns, values)

        data = tuple(t[learnable] for t in data)

        dataset = TensorDataset(*data)

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

                past_actions, actions = actions_with_past[:, :-1, :], actions_with_past[:, -1, :]

                critics_loss = self.critics.forward_for_loss(states, past_actions = past_actions, old_values = values, rewards = returns)
                critics_loss.backward()

                self.critics_optim.step()
                self.critics_optim.zero_grad()

                advantages = self.critics.calc_advantages(values, rewards = returns)

                actor_loss = self.actor.forward_for_loss(states, actions, log_probs, advantages, past_actions = past_actions)
                actor_loss.backward()

                self.actor_optim.step()
                self.actor_optim.zero_grad()

    @torch.no_grad()
    def forward(
        self,
        env,
        reward_hparams: HyperParams | None = None
    ) -> list[Memory]:

        self.actor.eval()
        self.critics.eval()

        memories = []
        memory = None

        for _ in tqdm(range(self.num_episodes), desc = 'episodes'):

            timestep = 0

            env_reset_kwargs = dict()
            if exists(self.env_hparams):
                env_hparam = choice(self.env_params)
                env_reset_kwargs.update(env_hparam = env_hparam)

            state = env.reset(**env_reset_kwargs)

            past_actions = None

            while timestep < self.max_episode_timesteps:

                # State -> Float['state_dim']

                state_tensor = state_to_tensor(state)

                # rewards are derived from the state

                rewards = self.state_to_reward(state, reward_hparams)

                # sample the action from the actor
                # and get the values from the critics

                actor_critic_past_actions = past_actions[-self.num_past_actions:] if exists(past_actions) else None

                actions, log_probs = self.actor(state_tensor, sample = True, past_actions = actor_critic_past_actions)

                values = self.critics(state_tensor, past_actions = actor_critic_past_actions)

                # store past actions

                past_action = rearrange(actions, '... -> 1 ...')
                past_actions = safe_cat(past_actions, past_action, dim = 0)

                # store memories

                memory = Memory(
                    learnable = tensor(True),
                    state = state_tensor,
                    actions = actions,
                    old_log_probs = log_probs,
                    rewards = rewards,
                    values = values,
                    done = tensor(False)
                )

                memories.append(memory)

                # next state

                state = env(actions)

                # increment timestep

                timestep += 1

        # add a non-learnable memory

        state_tensor = state_to_tensor(state)
        next_value = self.critics(state_tensor)

        memory = memory._replace(
            learnable = tensor(False),
            values = next_value,
            done = tensor(True)
        )

        memories.append(memory)

        # return memories

        return memories
