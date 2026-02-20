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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TD3_agent import TD3_Agent

# helper files
from pendulum_commandline_config import build_parser

# AGENT LADEN @OSANE
from TD3_agent import TD3_Agent

def train_pendulum():
    parser = build_parser()
    opts, _ = parser.parse_args()

    env_name        = opts.env_name
    noise_type      = opts.noise_type
    noise_beta      = opts.noise_beta
    expl_noise      = opts.exploration_noise
    max_episodes    = opts.max_episodes
    warmup_steps    = opts.warmup_steps
    train_iter      = opts.train_iter
    log_interval    = opts.log_interval
    seed            = opts.seed

    os.makedirs('checkpoint1/results/TD3_results/', exist_ok=True)

    # *** Environment ***
    env = gym.make(env_name)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.reset(seed=seed)

    obs_dim      = env.observation_space.shape[0]
    max_timesteps = 200   # Pendulum episodes are 200 steps

    # TD3 needs a per-player action space (Box)
    action_space = env.action_space  # already a Box for Pendulum

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Env: {env_name}, noise: {noise_type}({expl_noise}), seed: {seed}, device: {device}")

    # *** Agent ***
    TD3 = TD3_Agent(
        obs_dim            = obs_dim,
        act_dim            = action_space.shape[0],
        observation_space  = env.observation_space,
        action_space       = action_space,
        device             = device,
        gamma              = opts.gamma,
        tau                = opts.tau,
        batch_size         = opts.batch_size,
        actor_lr           = opts.actor_lr,
        critic_lr          = opts.critic_lr,
        policy_delay       = opts.policy_delay,
        noise_type         = noise_type,
        noise_beta         = noise_beta,
        episode_length     = max_timesteps,
        exploration_noise  = expl_noise,
        noise_target_policy= 0.2,
        clip_noise         = 0.5,
        use_PrioritizedExpReplay = False,
        PER_alpha          = 0.6,
        PER_beta_init      = 0.4,
        PER_beta_n_steps   = 100_000,
        seed               = seed,
    )

    # *** Logging ***
    rewards  = []
    lengths  = []
    total_steps = 0

    tag = f"TD3_{env_name}-noise{noise_type}-eps{expl_noise}-s{seed}"

    def save_statistics():
        with open(f"./checkpoint1/results/TD3_results/{tag}-stat.pkl", 'wb') as f:
            pickle.dump({
                "rewards":          rewards,
                "lengths":          lengths,
                "noise_type":       noise_type,
                "exploration_noise": expl_noise,
                "seed":             seed,
            }, f)

    # *** Training loop ***
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        TD3.reset_noise()
        episode_reward = 0.0

        for t in range(max_timesteps):
            total_steps += 1

            if total_steps < warmup_steps:
                # random actions during warm-up
                action = env.action_space.sample()
            else:
                action = TD3.select_action(state, explore=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            TD3.buffer.add_experience(
                state      = state,
                action     = action,
                reward     = reward,
                next_state = next_state,
                ended      = done,
            )

            state = next_state
            episode_reward += reward

            if done:
                break

        # gradient updates after each episode (like DDPG's train_iter)
        if total_steps >= warmup_steps:
            for _ in range(train_iter):
                TD3.train_step()

        rewards.append(episode_reward)
        lengths.append(t + 1)

        # checkpoint every 500 episodes
        if episode % 500 == 0:
            print("########## Saving checkpoint... ##########")
            torch.save(TD3.actor.state_dict(),
                       f'./checkpoint1/results/TD3_results/{tag}_ep{episode}.pth')
            save_statistics()

        # logging
        if episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))
            print(f'Episode {episode} \t avg length: {avg_length} \t reward: {avg_reward:.1f}')

    save_statistics()
    print(f"Done. Stats saved to ./results/TD3_results/{tag}-stat.pkl")
    env.close()


if __name__ == '__main__':
    train_pendulum()
