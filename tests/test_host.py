import pytest

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
    LatentGenePool
)

from host_pytorch.mock_env import Env, mock_hparams

def test_actor_critic_reward_shaper():

    actor = Actor(
        4,
        dim_action_embed = 4,
        past_action_conv_kernel = 3,
    )

    state = torch.randn(4, 512)
    actions, log_prob = actor(state, past_actions = torch.randint(0, 4, (4, 2, 1)), sample = True)

    critics = Critics([1., 2.], num_critics = 2, num_actions = 4)

    loss = critics.forward_for_loss(state, rewards = torch.randn(4, 2), old_values = torch.randn(4, 2))
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

@pytest.mark.parametrize('num_actions', (5, (3, 5, 2)))
def test_e2e(
    num_actions
):

    env = Env()

    agent = Agent(
        num_actions = num_actions,
        actor = dict(
            dims = (env.dim_state, 256, 128),
        ),
        critics = dict(
            dims = (env.dim_state, 256),
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

    # able to add some custom reward in addition to defaults

    def custom_reward(state, hparam, past_actions = None):
        return 5.

    agent.add_reward_function_(
        custom_reward,
        group_name = 'regularization',
        weight = 1e-5
    )

    # able to delete a reward

    agent.delete_reward_function_(
        group_name = 'regularization',
        reward_fn_name = 'custom_reward'
    )

    # learning

    memories = agent(env)

    agent.learn(memories)

    agent.save('./standing-up-policy.pt', overwrite = True)

def test_actor_critic_with_latents():
    latent_gene_pool = LatentGenePool(
        num_latents = 32,
        dim_latent = 64
    )

    actor = Actor(
        4,
        dim_action_embed = 4,
        past_action_conv_kernel = 3,
        dim_latent = 64
    )

    state = torch.randn(4, 512)
    latent = latent_gene_pool(latent_id = 4)

    actions, log_prob = actor(
        state,
        latents = latent,
        past_actions = torch.randint(0, 4, (4, 2, 1)),
        sample = True
    )

    critics = Critics(
        [1., 2.],
        num_critics = 2,
        num_actions = 4,
        dim_latent = 64
    )

    loss = critics.forward_for_loss(
        state,
        latents = latent,
        rewards = torch.randn(4, 2),
        old_values = torch.randn(4, 2)
    )

    loss.backward()
