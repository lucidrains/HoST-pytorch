from __future__ import annotations
from dataclasses import asdict

import torch
from torch import tensor, randn, randint
from torch.nn import Module

from host_pytorch.host import State, HyperParams
from host_pytorch.tensor_typing import Int

from einops import pack

# functions

def mock_hparams():
    return HyperParams(
        height_stage1_thres = randn(()),
        height_stage2_thres = randn(()),
        joint_velocity_abs_limit = randn((3,)),
        joint_position_PD_target = randn((3,)),
        joint_position_lower_limit = randn((3,)),
        joint_position_higher_limit = randn((3,)),
        upper_body_posture_target = randn((3,)),
        height_base_target = randn(()),
        ankle_parallel_thres = 0.05,
        joint_power_T = 1.,
        feet_parallel_min_height_diff = 0.02,
        feet_distance_thres = 0.9,
        waist_yaw_joint_angle_thres = 1.4,
        contact_force_ratio_is_foot_stumble = 3.,
        max_hip_joint_angle_lr = 1.4,
        min_hip_joint_angle_lr = 0.9,
        knee_joint_angle_max_min = (2.85, -0.06),
        shoulder_joint_angle_max_min = (-0.02, 0.02)
    )

def random_state():
    return State(
        head_height = randn(()),
        angular_velocity = randn((3,)),
        linear_velocity = randn((3,)),
        orientation = randn((3,)),
        projected_gravity_vector = randn(()),
        joint_velocity = randn((3,)),
        joint_acceleration = randn((3,)),
        joint_torque = randn((3,)),
        joint_position = randn((3,)),
        left_ankle_keypoint_z = randn((3,)),
        right_ankle_keypoint_z = randn((3,)),
        left_feet_height = randn(()),
        right_feet_height = randn(()),
        left_feet_pos = randn((3,)),
        right_feet_pos = randn((3,)),
        left_shank_angle = randn(()),
        right_shank_angle = randn(()),
        upper_body_posture = randn((3,)),
        height_base = randn(()),
        contact_force = randn((3,)),
        hip_joint_angle_lr = randn(()),
        robot_base_angle_q = randn((3,)),
        feet_angle_q = randn((3,)),
        knee_joint_angle_lr = randn((2,)),
        shoulder_joint_angle_l = randn(()),
        shoulder_joint_angle_r = randn(()),
        waist_yaw_joint_angle = randn(())
    )

# mock env

class Env(Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('dummy', tensor(0))

    @property
    def dim_state(self):
        state_dict = asdict(random_state())

        state_features, _ = pack([*state_dict.values()], '*')

        return state_features.shape[-1]

    @property
    def device(self):
        return self.dummy.device

    def reset(
        self
    ) -> State:
        return random_state()

    def forward(
        self,
        actions: Int['a'],
    ) -> State:

        return random_state()
