import torch
from torch import randn

from host_pytorch.host import (
    MLP,
    Actor,
    Critics,
    GroupedMLP,
    RewardShapingWrapper,
    HyperParams,
    Agent,
)

from host_pytorch.mock_env import Env, mock_hparams

def test_actor_critic_reward_shaper():

    actor = Actor(
        4,
        dim_action_embed = 4,
        past_action_conv_kernel = 3,
    )

    state = torch.randn(4, 512)
    actions, log_prob = actor(state, past_actions = torch.randint(0, 4, (4, 2)), sample = True)

    critics = Critics([1., 2.], num_critics = 2, num_actions = 4)

    loss = critics(state, rewards = torch.randn(4, 2))
    loss.backward()

    values = critics(state)
    advantages = critics.calc_advantages(values, rewards = torch.randn(4, 2))

    policy_loss = actor.forward_for_loss(state, actions, log_prob, advantages)
    policy_loss.backward()

    reward_shaping = RewardShapingWrapper(
        critics_kwargs = dict(
            num_actions = 4
        )
    )

    env = Env()

    hparams = mock_hparams()

    rewards = reward_shaping(env.reset(), hparams)

def test_e2e():

    env = Env()

    agent = Agent(
        actor = dict(
            dims = (env.dim_state, 256, 128),
            num_actions = 5,
        ),
        critics = dict(
            dims = (env.dim_state, 256),
            num_actions = 5
        ),
        reward_hparams = dict(
            height_stage1_thres = randn(()),
            height_stage2_thres = randn(()),
            joint_velocity_abs_limit = randn((3,)),
            joint_position_PD_target = randn((3,)),
            joint_position_lower_limit = randn((3,)),
            joint_position_higher_limit = randn((3,)),
            upper_body_posture_target = randn((3,)),
            height_base_target = randn((3,)),
        )
    )

    memories = agent(env)

    agent.learn(memories)

    agent.save('./standing-up-policy.pt', overwrite = True)
