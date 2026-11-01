from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, stack, tensor

from einx import get_at
from einops import repeat

from host_pytorch.host import State, state_to_tensor, exists

# humanoid standup-v5 adapter: 17 torque channels -> 17 discrete action groups

ACTION_LOW = -0.4
ACTION_HIGH = 0.4
FRAME_SKIP = 5
DEFAULT_NUM_BINS = 11
NUM_JOINTS = 17

def to_tensor(x):
    return tensor(np.asarray(x, dtype = np.float32))

def mat_to_euler(mat):
    r = np.asarray(mat).reshape(3, 3)

    roll = np.arctan2(r[2, 1], r[2, 2])
    pitch = np.arcsin(np.clip(-r[2, 0], -1., 1.))
    yaw = np.arctan2(r[1, 0], r[0, 0])
    return np.array([roll, pitch, yaw])

def quat_to_euler(quat, mujoco):
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype = np.float64))
    return mat_to_euler(mat)

def body_pitch(xmat):
    r = np.asarray(xmat).reshape(3, 3)
    return np.arcsin(np.clip(-r[2, 0], -1., 1.))

@torch.no_grad()
def select_action(actor, observation):
    logits = actor.forward_net(observation[None])

    actions = [
        group_logits.argmax(dim = -1)
        for group_logits in logits.split(actor.num_actions, dim = -1)
    ]

    return stack(actions, dim = -1)[0]

def to_indices(actions):
    if isinstance(actions, Tensor):
        actions = actions.detach().cpu().numpy()

    return np.asarray(actions, dtype = np.int64)

SUPINE = [
    -0.023934540540761003, -0.0043274978853108225, 0.08566313459454498,
    0.9998517796416483, -0.002084440868074425, 0.01708551972572183,
    -0.000398584242906153, 0.00558391897316152, -0.1140933707754823,
    0.004005565347680796, -0.023492358164565602, -0.0002780495705784499,
    0.09475272860981634, -0.03303412846260827, -0.023796510645414565,
    -0.005942066489197335, 0.0948781377038809, -0.033043773964836404,
    -0.21445459129998518, 0.14753069336672872, -1.1552050737435733,
    0.2440176355657165, -0.11514624134992313, -1.147981857581031,
]

class HumanoidStandupEnv:
    """ Gymnasium humanoid stand-up with a HoST `State` interface. """

    def __init__(
        self,
        num_bins = DEFAULT_NUM_BINS,
        render_mode = None,
        seed = 0,
        assist_force = 0.,
        initial_qpos = None
    ):
        import gymnasium as gym

        self.num_bins = num_bins
        self.action_bins = np.linspace(ACTION_LOW, ACTION_HIGH, num_bins, dtype = np.float32)
        self.assist_force = float(assist_force)
        self.initial_qpos = None if not exists(initial_qpos) else np.asarray(initial_qpos, dtype = np.float64)
        self.render_mode = render_mode

        # v5 adds inertia / contact terms; strip so obs matches v4's 45-d state

        self.env = gym.make(
            'HumanoidStandup-v5',
            render_mode = render_mode,
            include_cinert_in_observation = False,
            include_cvel_in_observation = False,
            include_qfrc_actuator_in_observation = False,
            include_cfrc_ext_in_observation = False,
        )

        self.unwrapped = self.env.unwrapped
        self.model = self.unwrapped.model
        self.data = self.unwrapped.data
        self.dt = self.model.opt.timestep * FRAME_SKIP

        self.foot_ids = [self.model.body(name).id for name in ('right_foot', 'left_foot')]
        self.shin_ids = [self.model.body(name).id for name in ('right_shin', 'left_shin')]
        self.torso_id = self.model.body('torso').id

        self.done = False
        self._prev_joint_velocity = np.zeros(NUM_JOINTS, dtype = np.float32)

        self.reset(seed = seed)

    @property
    def num_actions(self):
        return (self.num_bins,) * NUM_JOINTS

    @property
    def dim_state(self):
        return state_to_tensor(self.state()).shape[-1]

    @property
    def success(self):
        return bool(self.data.qpos[2] > 1.)

    def set_assist_force(self, force):
        self.assist_force = float(force)

    def state_tensor(self, state = None):
        # compact policy observation (the 45-d proprioceptive state)

        return to_tensor(self._obs)

    def reset(
        self,
        seed = None,
        env_hparams = None,
        **kwargs
    ):
        obs, _ = self.env.reset(seed = seed, **kwargs)

        if exists(self.initial_qpos):
            import mujoco

            self.data.qpos[:] = self.initial_qpos
            self.data.qvel[:] = 0.
            mujoco.mj_forward(self.model, self.data)
            obs = self.unwrapped._get_obs()

        self._obs = obs
        self._prev_joint_velocity = np.zeros(NUM_JOINTS, dtype = np.float32)
        self.done = False
        self.data.xfrc_applied[:] = 0.
        return self.state()

    def state(self):
        import mujoco

        d = self.data
        qpos, qvel = d.qpos, d.qvel

        root_pos = qpos[0:3]
        root_quat = qpos[3:7]
        lin_vel = qvel[0:3]
        ang_vel = qvel[3:6]
        joint_pos = qpos[7:]
        joint_vel = qvel[6:]

        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.asarray(root_quat, dtype = np.float64))

        orientation = mat_to_euler(rot)
        projected_gravity_z = -rot.reshape(3, 3)[2, 2]

        feet_pos = d.xpos[self.foot_ids]
        feet_height = feet_pos[:, 2]
        feet_euler = np.stack([quat_to_euler(quat, mujoco) for quat in d.xquat[self.foot_ids]])
        shank_angles = np.array([body_pitch(xmat) for xmat in d.xmat[self.shin_ids]])

        contact_force = d.cfrc_ext[self.foot_ids][:, :3].sum(0)

        joint_acc = (joint_vel - self._prev_joint_velocity) / self.dt
        self._prev_joint_velocity = joint_vel.copy()

        return State(
            head_height = to_tensor(root_pos[2] + 0.19),
            angular_velocity = to_tensor(ang_vel),
            linear_velocity = to_tensor(lin_vel),
            orientation = to_tensor(orientation),
            projected_gravity_vector = to_tensor(projected_gravity_z),
            joint_velocity = to_tensor(joint_vel),
            joint_acceleration = to_tensor(joint_acc),
            joint_torque = to_tensor(d.qfrc_actuator[6:]),
            joint_position = to_tensor(joint_pos),
            left_ankle_keypoint_z = repeat(to_tensor(feet_height[1]), ' -> d', d = 3),
            right_ankle_keypoint_z = repeat(to_tensor(feet_height[0]), ' -> d', d = 3),
            left_feet_height = to_tensor(feet_height[1]),
            right_feet_height = to_tensor(feet_height[0]),
            left_shank_angle = to_tensor(shank_angles[1]),
            right_shank_angle = to_tensor(shank_angles[0]),
            left_feet_pos = to_tensor(feet_pos[1]),
            right_feet_pos = to_tensor(feet_pos[0]),
            upper_body_posture = to_tensor(joint_pos[11:17]),
            height_base = to_tensor(root_pos[2]),
            contact_force = to_tensor(contact_force),
            hip_joint_angle_lr = to_tensor([joint_pos[3], joint_pos[7]]),
            robot_base_angle_q = to_tensor(orientation),
            feet_angle_q = to_tensor(feet_euler[1]),
            knee_joint_angle_lr = to_tensor([joint_pos[6], joint_pos[10]]),
            shoulder_joint_angle_l = to_tensor(joint_pos[14]),
            shoulder_joint_angle_r = to_tensor(joint_pos[11]),
            waist_yaw_joint_angle = to_tensor(joint_pos[0]),
        )

    def actions_to_continuous(self, actions):
        return get_at('[bins], a -> a', self.action_bins, to_indices(actions), backend = 'numpy')

    def step_continuous(self, ctrl):
        self.data.xfrc_applied[self.torso_id, 2] = self.assist_force
        obs, reward, terminated, truncated, info = self.env.step(np.asarray(ctrl, dtype = np.float32))
        self._obs = obs
        self.done = bool(terminated or truncated)
        return obs, reward, self.done, info

    def step(self, actions):
        return self.step_continuous(self.actions_to_continuous(actions))

    def __call__(self, actions):
        self.step(actions)
        return self.state()

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

# 2D biped: torso, head, two 3-link legs, position-controlled joints

PLANAR_ACTION_RANGES = (
    (-1.2, 1.2), (0.0, 1.8), (-0.8, 0.8),
    (-1.2, 1.2), (0.0, 1.8), (-0.8, 0.8),
)

PLANAR_START_QPOS = (
    0., 0.35, 0.,
    -1.0, 2.0, -1.0,
    -1.0, 2.0, -1.0,
)

class PlanarHumanoidEnv:
    """ Small MuJoCo biped that starts in a squat and stands up. """

    def __init__(
        self,
        num_bins = 5,
        seed = 0,
        render_size = 480
    ):
        import mujoco

        self.mujoco = mujoco
        self.num_bins = num_bins
        self.action_bins = np.stack([
            np.linspace(low, high, num_bins, dtype = np.float32)
            for low, high in PLANAR_ACTION_RANGES
        ])
        self.action_ranges = PLANAR_ACTION_RANGES

        xml_path = Path(__file__).resolve().parent / 'planar_humanoid.xml'
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * FRAME_SKIP
        self.frame_skip = FRAME_SKIP
        self.render_size = render_size
        self.renderer = None

        self.foot_ids = [self.model.body(name).id for name in ('foot_l', 'foot_r')]
        self.shin_ids = [self.model.body(name).id for name in ('shin_l', 'shin_r')]

        self.done = False
        self._prev_joint_velocity = np.zeros(6, dtype = np.float32)
        self.reset(seed = seed)

    @property
    def num_actions(self):
        return (self.num_bins,) * 6

    @property
    def dim_state(self):
        return 28

    @property
    def success(self):
        return bool(self.data.qpos[1] > 0.60 and np.cos(self.data.qpos[2]) > 0.8)

    def reset(self, seed = None, **kwargs):
        if exists(seed):
            np.random.seed(seed)

        qpos = np.array(PLANAR_START_QPOS, dtype = np.float64)
        qpos[0] += np.random.uniform(-0.02, 0.02)
        qpos[1] += np.random.uniform(-0.01, 0.01)
        qpos[2] += np.random.uniform(-0.03, 0.03)
        qpos[3:] += np.random.uniform(-0.08, 0.08, 6)

        self.data.qpos[:] = qpos
        self.data.qvel[:] = np.random.uniform(-0.03, 0.03, self.model.nv)
        self.data.xfrc_applied[:] = 0.
        self.mujoco.mj_forward(self.model, self.data)

        self._prev_joint_velocity = np.zeros(6, dtype = np.float32)
        self.done = False
        return self.state()

    def state(self):
        d = self.data
        qpos, qvel = d.qpos, d.qvel

        root_pos = qpos[0:3]
        root_pitch = qpos[2]
        joint_pos = qpos[3:]
        joint_vel = qvel[3:]

        feet_pos = d.xpos[self.foot_ids]
        feet_height = feet_pos[:, 2]
        feet_pitch = np.array([body_pitch(xmat) for xmat in d.xmat[self.foot_ids]])
        shank_pitch = np.array([body_pitch(xmat) for xmat in d.xmat[self.shin_ids]])

        contact_force = d.cfrc_ext[self.foot_ids][:, :3].sum(0)
        joint_acc = (joint_vel - self._prev_joint_velocity) / self.dt
        self._prev_joint_velocity = joint_vel.copy()

        orientation = np.array([0., root_pitch, 0.])
        gravity_z = -np.cos(root_pitch)

        return State(
            head_height = to_tensor(root_pos[1] + 0.62),
            angular_velocity = to_tensor([0., qvel[2], 0.]),
            linear_velocity = to_tensor([qvel[0], 0., qvel[1]]),
            orientation = to_tensor(orientation),
            projected_gravity_vector = to_tensor(gravity_z),
            joint_velocity = to_tensor(joint_vel),
            joint_acceleration = to_tensor(joint_acc),
            joint_torque = to_tensor(d.qfrc_actuator),
            joint_position = to_tensor(joint_pos),
            left_ankle_keypoint_z = repeat(to_tensor(feet_height[0]), ' -> d', d = 3),
            right_ankle_keypoint_z = repeat(to_tensor(feet_height[1]), ' -> d', d = 3),
            left_feet_height = to_tensor(feet_height[0]),
            right_feet_height = to_tensor(feet_height[1]),
            left_shank_angle = to_tensor(shank_pitch[0]),
            right_shank_angle = to_tensor(shank_pitch[1]),
            left_feet_pos = to_tensor(feet_pos[0]),
            right_feet_pos = to_tensor(feet_pos[1]),
            upper_body_posture = to_tensor([0., 0., 0.]),
            height_base = to_tensor(root_pos[1]),
            contact_force = to_tensor(contact_force),
            hip_joint_angle_lr = to_tensor([joint_pos[0], joint_pos[3]]),
            robot_base_angle_q = to_tensor(orientation),
            feet_angle_q = to_tensor([0., feet_pitch[0], 0.]),
            knee_joint_angle_lr = to_tensor([joint_pos[1], joint_pos[4]]),
            shoulder_joint_angle_l = to_tensor(0.),
            shoulder_joint_angle_r = to_tensor(0.),
            waist_yaw_joint_angle = to_tensor(0.),
        )

    def state_tensor(self, state = None):
        d = self.data
        qpos, qvel = d.qpos, d.qvel

        feet_z = d.xpos[self.foot_ids][:, 2]
        contacts = (feet_z < 0.06).astype(np.float32)

        observation = np.concatenate((
            np.array([
                qpos[1],
                np.sin(qpos[2]),
                np.cos(qpos[2]),
                qvel[0],
                qvel[1],
                qvel[2],
                feet_z[0],
                feet_z[1],
                contacts[0],
                contacts[1],
            ], dtype = np.float32),
            qpos[3:].astype(np.float32),
            qvel[3:].astype(np.float32),
            d.qfrc_actuator[3:].astype(np.float32),
        ))

        return to_tensor(observation)

    def actions_to_continuous(self, actions):
        return get_at('a [i], a -> a', self.action_bins, to_indices(actions), backend = 'numpy')

    def step(self, actions):
        ctrl = self.actions_to_continuous(actions)
        self.data.ctrl[:] = ctrl

        for _ in range(self.frame_skip):
            self.mujoco.mj_step(self.model, self.data)

        self.done = bool(self.data.qpos[1] < 0.25 or abs(self.data.qpos[2]) > 1.3)
        return self.state()

    def __call__(self, actions):
        return self.step(actions)

    def render(self):
        if not exists(self.renderer):
            self.renderer = self.mujoco.Renderer(
                self.model,
                height = self.render_size,
                width = self.render_size,
            )

        camera = self.mujoco.MjvCamera()
        camera.lookat[:] = [0., 0., 0.6]
        camera.distance = 2.2
        camera.azimuth = 90.
        camera.elevation = -10.

        self.renderer.update_scene(self.data, camera = camera)
        return self.renderer.render()

    def close(self):
        if exists(self.renderer):
            self.renderer.close()
            self.renderer = None
