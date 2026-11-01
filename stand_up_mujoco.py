"""Roll out a trained planar humanoid stand-up policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import torch

from host_pytorch import Agent
from mujoco_env import PlanarHumanoidEnv, select_action

# constants

HEIGHT_SUCCESS_THRES = 0.58
PITCH_SUCCESS_THRES = 0.45

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default = 'standing-up-policy.pt')
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--steps', type = int, default = 250)
    parser.add_argument('--video', default = 'standing_up.mp4')
    return parser.parse_args()

def load_actor(env, path):
    assert Path(path).exists(), f'{path} not found. Run `python train_standup_rl.py` first.'

    agent = Agent(
        num_actions = env.num_actions,
        actor = dict(dims = (28, 256, 128)),
        critics = dict(dims = (28, 256)),
        num_past_actions = 0,
        reward_hparams = None,
    )

    package = torch.load(path, map_location = 'cpu', weights_only = True)
    agent.actor.load_state_dict(package['actor'])

    agent.actor.eval()
    return agent.actor

@torch.no_grad()
def run_episode(env, actor, seed, steps, render = False):
    env.reset(seed = seed)
    heights, pitches, frames = [], [], []

    for _ in range(steps):
        action = select_action(actor, env.state_tensor())
        env.step(action)

        heights.append(float(env.data.qpos[1]))
        pitches.append(float(env.data.qpos[2]))

        if render:
            frames.append(env.render())

        if env.done:
            break

    return heights, pitches, frames

def main():
    args = parse_args()

    env = PlanarHumanoidEnv(seed = args.seed)
    actor = load_actor(env, args.checkpoint)

    heights, pitches, frames = run_episode(env, actor, args.seed, args.steps, render = True)
    success = heights[-1] > HEIGHT_SUCCESS_THRES and abs(pitches[-1]) < PITCH_SUCCESS_THRES

    print(f'seed {args.seed}: standing={success}  height={heights[-1]:.3f} m  pitch={pitches[-1]:.2f} rad')

    if args.video:
        imageio.mimsave(args.video, frames, fps = 40)
        print(f'wrote {args.video}')

    env.close()

if __name__ == '__main__':
    main()
