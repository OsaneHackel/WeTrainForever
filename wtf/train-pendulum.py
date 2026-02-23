"""
Train TD3 on Pendulum-v1 (Checkpoint 1).
Saves reward/length statistics as a .pkl file (same format as DDPG)
so results can be directly compared in the report notebook.

Usage examples:
    # run from Hockey-TD3
    python checkpoint1/train_pendulum.py

    # vary exploration noise (analogous to DDPG eps sweep)
    python checkpoint1/train_pendulum.py --exploration_noise 0.1 --seed 0
    python checkpoint1/train_pendulum.py --exploration_noise 0.2 --seed 0

    # use pink noise instead of Gaussian
    python checkpoint1/train_pendulum.py --noise_type Pink --seed 0
"""

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pickle
import sys
import os

from wtf.agents.SAC import SAC
from wtf.agents.DDPG import DDPGAgent
from wtf.utils import generate_id
from wtf.utilssac import create_state_dict


def train_pendulum():

    env_name        = "Pendulum-v1"
    dir = 'checkpoints/Pendulum_DDPG/'
    os.makedirs(dir, exist_ok=True)
    run_name = f"{generate_id()}_DDPG"

    # *** Environment ***
    env = gym.make(env_name)


    #obs_dim      = env.observation_space.shape[0]
    max_timesteps = 200   # Pendulum episodes are 200 steps
    max_episodes = 5_000
    warmup_steps = 10000
    train_iter=32

    # TD3 needs a per-player action space (Box)
    action_space = env.action_space  # already a Box for Pendulum

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # *** Agent ***

    '''agent = SAC(
        env.observation_space,
        action_space,
        gamma=0.99,
        tau=5e-3,
        alpha=0.2,
        batch_size=128,
        buffer_size=1_000_000,
        lr=3e-4,
        critic_optimizer="SLS",
        policy_optimizer = "SLS"
    )'''
    agent = DDPGAgent(
        env.observation_space,
        action_space,
        device = device,
        eps = 0.01,
        discount=0.99,
        tau=5e-3,
        alpha=0.2,
        batch_size=128,
        ema_update = 2e-3,
        max_episodes = 50_000,
        optimizer = "ADAM",
        lr_scheduler = False    
    )

    # *** Logging ***
    rewards  = []
    lengths  = []
    total_steps = 0

    

    def save_statistics():
        stat_dir= dir
        with open(f"{dir}/{run_name}-stat.pkl", 'wb') as f:
            pickle.dump({
                "rewards":          rewards,
                "lengths":          lengths
            }, f)

    # *** Training loop ***
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0

        for t in range(max_timesteps):
            total_steps += 1

            if total_steps < warmup_steps:
                # random actions during warm-up
                action = env.action_space.sample()
            else:
                action = agent.act(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition((state, action, reward, next_state, done))

            state = next_state
            episode_reward += reward

            if done:
                break

        # gradient updates after each episode (like DDPG's train_iter)
        if total_steps >= warmup_steps:
            for _ in range(train_iter):
                agent.train(1)

        rewards.append(episode_reward)
        lengths.append(t + 1)

        # checkpoint every 500 episodes
        if episode % 500 == 0:
            print("########## Saving checkpoint... ##########")
            state_dict = create_state_dict("DDPG", agent)
            torch.save(state_dict, f"{dir}/{run_name}-dict.pth")
            save_statistics()

        # logging
        if episode % 20 == 0:
            avg_reward = np.mean(rewards[-20:])
            avg_length = int(np.mean(lengths[-20:]))
            print(f'Episode {episode} \t avg length: {avg_length} \t reward: {avg_reward:.1f}')

    save_statistics()
    print(f"Done. Stats saved to {dir}/{run_name}")
    env.close()


if __name__ == '__main__':
    train_pendulum()
