import random
import numpy as np
import torch
import pickle
import hockey.hockey_env as h_env

from datetime import datetime
from pathlib import Path
from gymnasium import spaces
import imageio

from wtf.utilssac import create_state_dict, load_agent
from wtf.eval import evaluate
from wtf.agents.Agent_Pool import AgentPool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(8)

#Reward function for 2026-02-19_16-23-03-Hockey-SAC
def get_reward(info):
    reward = 10.0 * info["winner"]
    #reward += 2.0 * info["reward_closeness_to_puck"]
    #reward += 3.0 * info["reward_touch_puck"]
    return reward

def fill_agent_pool():
    #ddpg = load_agent("./BestAgents/DDPG.pth", "DDPG", evaluate=True)
    static_agents = [
        h_env.BasicOpponent(weak=True),
        h_env.BasicOpponent(weak=False),
        load_agent("./BestAgents/SAC_adam.pth", "SAC", evaluate=True),
        load_agent("./BestAgents/SAC_adam_2.pth", "SAC", evaluate=True),
        load_agent("./BestAgents/SAC_best.pth", "SAC", evaluate=True),
    ]
    pool = AgentPool(max_agents=10, static_agents=static_agents)
    return pool

def train_sac_step_based(agent,
                         max_episodes=50_000,
                         max_timesteps=300,
                         warmup_steps=10_000,
                         self_play=False, 
                         critic_optimizer="ADAM",
                         policy_optimizer="ADAM"):

    base_dir = Path('checkpoints')
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f'{ts}-Hockey-SAC-critic-optim-selfplay'

    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True)

    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)

    opponents_pool = fill_agent_pool()

    total_env_steps = 0
    rewards_log = []
    c_loss = []
    p_loss = []
    a_loss = []
    critic_lrs = []
    policy_lrs = []

    for episode in range(1, max_episodes + 1):
        # Rollout
        agent.reset()
        obs_agent1, _ = env.reset()
        obs_agent2 = env.obs_agent_two()

        opponent = opponents_pool.get_agent()
        episode_reward = 0
        frames = [] if episode % 100 == 0 else None

        agent.to_device('cpu')
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
            agent.to_device(device)
            c_loss_epoch, p_loss_epoch, a_loss_epoch, critic_lrs_epoch, policy_lrs_epoch = agent.train(t+1)
            c_loss.extend(c_loss_epoch)
            p_loss.extend(p_loss_epoch)
            a_loss.extend(a_loss_epoch)
            critic_lrs.extend(critic_lrs_epoch)
            policy_lrs.extend(policy_lrs_epoch)

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
                "c_loss": c_loss,
                "p_loss": p_loss,
                "a_loss": a_loss,
                "policy_lrs": policy_lrs,
                "critic_lrs": critic_lrs
            }
            with open(stat_path, 'wb') as f:
                pickle.dump(stats, f)
        #250
        if episode % 250 ==0:
            checkpoint_path = run_dir / f'checkpoint_{episode}.pth'
            stat_path = run_dir / f'stats_{episode}.pth'
            fig_path = run_dir / f'figures_{episode}'
            fig_path.mkdir()

            state_dict = create_state_dict("SAC", agent)
            torch.save(state_dict, checkpoint_path)
            save_statistics(stat_path)
            evaluate("SAC", fig_path, stat_path, checkpoint_path)
        #1000
        if episode % 1000 == 0:
            opponents_pool.add_agent(agent)
            print("Added agent to pool")

    return rewards_log


if __name__ == "__main__":
    from wtf.agents.SAC import SAC

    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    full_action_space = env.action_space 
    n_actions_per_player = full_action_space.shape[0] // 2
    agent_action_space = spaces.Box(low=full_action_space.low[:n_actions_per_player],
                                    high=full_action_space.high[:n_actions_per_player],
                                    dtype=full_action_space.dtype)
    critic_optimizer="ADAM"
    policy_optimizer = "ADAM"
    
    agent = load_agent("./BestAgents/SAC_best.pth","SAC", evaluate=False)
    agent.to_device(device)

    train_sac_step_based(
        agent,
        warmup_steps=200_000,
        self_play=True,       
        critic_optimizer=critic_optimizer,
        policy_optimizer = policy_optimizer
    )
