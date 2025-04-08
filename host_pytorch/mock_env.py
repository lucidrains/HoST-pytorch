from __future__ import annotations

import torch
from torch import tensor, randn, randint
from torch.nn import Module

from host_pytorch import State

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
        past_actor_actions = randint(0, 1, (3, 2)),
    )

# mock env

class Env(Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('dummy', tensor(0))

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
