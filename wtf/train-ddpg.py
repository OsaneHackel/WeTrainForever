import random
import numpy as np
import torch
import pickle
import hockey.hockey_env as h_env

from datetime import datetime
from pathlib import Path
from gymnasium import spaces
import imageio

from wtf.utilssac import create_state_dict
from wtf.eval import evaluate_ddpg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(8)

#Reward function for 2026-02-19_16-23-03-Hockey-SAC
def get_reward(info):
    reward = 25.0 * info["winner"]
    reward += 2.0 * info["reward_closeness_to_puck"]
    reward += 3.0 * info["reward_touch_puck"]
    return reward
'''
#reward function for 2026-02-19_18-49-15-Hockey-SAC
def get_reward(info):
    reward = 20.0 * info["winner"]
    reward += 1.0 * info["reward_closeness_to_puck"]
    reward += 0.2 * info["reward_touch_puck"]
    return reward

# next?
def get_reward(info):
    reward = 30.0 * info["winner"]
    reward += 2.0 * info["reward_closeness_to_puck"]
    reward += 1.5 * info["reward_touch_puck"]
    return reward'''

def train_sac_step_based(agent,
                         max_episodes=50_000,
                         max_timesteps=300,
                         warmup_steps=10_000,
                         self_play=False, 
                         eps = 0.01):
    base_dir = Path('checkpoints')
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f'{ts}-Hockey-DDPG-{eps}'

    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True)

    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)

    # Fixed opponents for stability (can swap later)
    weak_opponent = h_env.BasicOpponent(weak=True)
    strong_opponent = h_env.BasicOpponent(weak=False)
    opponents = [weak_opponent, strong_opponent]

    total_env_steps = 0
    rewards_log = []
    c_loss = []
    p_loss =[]
    lrs = []

    for episode in range(1, max_episodes + 1):
        # Rollout
        agent.reset()
        obs_agent1, _ = env.reset()
        obs_agent2 = env.obs_agent_two()

        opponent = random.choice(opponents)
        episode_reward = 0
        frames = [] if episode % 100 == 0 else None

        #agent.to_device('cpu')
        for t in range(max_timesteps):
            # Always train as player 1 (much more stable early on)
            a1 = agent.act(obs_agent1)  # SAC should be stochastic internally
            a2 = opponent.act(obs_agent2)

            obs_agent1_new, _, done, trunc, info1 = env.step(np.hstack([a1, a2]))
            info2 = env.get_info_agent_two()
            obs_agent2_new = env.obs_agent_two()

            reward = get_reward(info1)
            agent.store_transition((obs_agent1, a1, reward, obs_agent1_new, done))
            episode_reward += reward

            if self_play:
                opp_reward = get_reward(info2)
                agent.store_transition((obs_agent2, a2, opp_reward, obs_agent2_new, done))

            obs_agent1 = obs_agent1_new
            obs_agent2 = obs_agent2_new
            total_env_steps += 1

            if frames is not None:
                frames.append(env.render(mode='rgb_array'))

            if done or trunc:
                break

        # Train
        if total_env_steps > warmup_steps:
            #agent.to_device(device)
            losses_epoch, lr_epoch = agent.train(t+1)
            c_losses = [tpl[0] for tpl in losses_epoch]
            p_losses = [tpl[1] for tpl in losses_epoch]
            c_loss.extend(c_losses)
            p_loss.extend(p_losses)
            lrs.extend(lr_epoch)

        rewards_log.append(episode_reward)

        if frames:
            outpath = run_dir / 'games' / f'{episode:05d}.gif'
            outpath.parent.mkdir(exist_ok=True)
            imageio.mimwrite(outpath, frames, fps=30)


        # ---- LOGGING ----
        if episode % 20 == 0:
            avg_reward = np.mean(rewards_log[-20:])
            print(f"Episode {episode} | Avg Reward: {avg_reward:.2f} | Steps: {total_env_steps}", flush=True)

        def save_statistics(stat_path):
            stats = {
                "rewards" : rewards_log, 
                "c_loss":c_loss,
                "p_loss": p_loss,
                "lrs": lrs
            }
            with open(stat_path, 'wb') as f:
                pickle.dump(stats, f)
        #250
        if episode % 250 ==0:
            checkpoint_path = run_dir / f'checkpoint_{episode}.pth'
            stat_path = run_dir / f'stats_{episode}.pth'
            fig_path = run_dir / f'figures_{episode}'
            fig_path.mkdir()

            state_dict = create_state_dict("DDPG", agent)
            torch.save(state_dict, checkpoint_path)
            save_statistics(stat_path)
            evaluate_ddpg("DDPG", fig_path, stat_path, checkpoint_path)

    return rewards_log


if __name__ == "__main__":
    from wtf.agents.DDPG import DDPGAgent

    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    full_action_space = env.action_space 
    n_actions_per_player = full_action_space.shape[0] // 2
    agent_action_space = spaces.Box(low=full_action_space.low[:n_actions_per_player],
                                    high=full_action_space.high[:n_actions_per_player],
                                    dtype=full_action_space.dtype)
    critic_optimizer="SLS"
    policy_optimizer = "SLS"
    
    agent = DDPGAgent(
        env.observation_space,
        full_action_space,
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
    #agent.to_device(device)

    train_sac_step_based(
        agent,
        #10_000
        warmup_steps=10_000,
        self_play=True,       
        eps=0.01
    )
