import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import optparse
import pickle
import hockey.hockey_env as h_env

from wtf.agents.DDPG import DDPGAgent
from wtf.agents.Agent_Pool import AgentPool
from wtf.utils import generate_id
from wtf.eval import evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(16)


def get_reward(info):
    reward = 25.0 * info["winner"]  # -1 or 1
    reward += 2.0*info["reward_closeness_to_puck"]  # between -30 to 0
    reward += 3.0 * info["reward_touch_puck"]  # between 0-1
    return reward

def run():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env',action='store', type='string',
                         dest='env_name',default="HockeyEnv",
                         help='Environment (default %default)')
    optParser.add_option('-n', '--eps',action='store',  type='float',
                         dest='eps',default=0.075,
                         help='Policy noise (default %default)')
    optParser.add_option('-t', '--train',action='store',  type='int',
                         dest='train',default=12,
                         help='number of training batches per episode (default %default)')
    optParser.add_option('-l', '--lr',action='store',  type='float',
                         dest='lr',default=1e-4,
                         help='learning rate for actor/policy (default %default)')
    optParser.add_option('-m', '--maxepisodes',action='store',  type='int',
                         dest='max_episodes',default=50000,
                         help='number of episodes (default %default)')
    optParser.add_option('-u', '--update',action='store',  type='int',
                         dest='update_every',default=100,
                         help='number of episodes between target network updates (default %default)')
    optParser.add_option('-s', '--seed',action='store',  type='int',
                         dest='seed',default=None,
                         help='random seed (default %default)')
    optParser.add_option('--load_checkpoint',action='store',  type='string',
                         dest='checkpoint_path',default=None,
                         help='path to checkpoint to load (default %default)')
    optParser.add_option('--optimizer',action='store',  type='string',
                         dest='optimizer', default="Adam",
                         help='optimizer to use (default %default)')
    optParser.add_option('--lr_scheduler',action='store_true',dest='lr_scheduler',
                         default=False,help='enable learning rate scheduler'
    )
    opts, args = optParser.parse_args()
    ############## Hyperparameters ##############
    env_name = opts.env_name
    # creating environment
    if env_name == "HockeyEnv":
        env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
        #env = gym.envs.make("Hockey-v0")
        #env = gym.envs.make("Hockey-One-v0", mode=0, weak_opponent=True)
    else:
        env = gym.make(env_name)
    render = False
    log_interval = 20           # print avg reward in the interval
    max_episodes = opts.max_episodes # max training episodes
    max_timesteps = 300         # max timesteps in one episode

    train_iter = opts.train      # update networks for given batched after every episode
    eps = opts.eps               # noise of DDPG policy
    lr  = opts.lr                # learning rate of DDPG policy
    random_seed = opts.seed
    optimizer = opts.optimizer
    #############################################
    run_id= generate_id()

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    ddpg = DDPGAgent(
        env.observation_space, 
        env.action_space,
        device=device,
        eps = eps,
        discount=0.99,
        buffer_size=int(1e6),
        batch_size=64,
        learning_rate_actor = 1e-5,
        learning_rate_critic = 1e-4,
        ema_update = 5e-3,
        max_episodes = max_episodes, 
        optimizer = optimizer,
        lr_scheduler = opts.lr_scheduler
    )
    
    if opts.checkpoint_path is not None:
        print(f"Loading checkpoint from {opts.checkpoint_path}...")
        ckpt= torch.load(f"{opts.checkpoint_path}")
        ddpg.policy.load_state_dict(ckpt["policy"])
        ddpg.Q.load_state_dict(ckpt["Q"])
        ddpg.policy_target.load_state_dict(ckpt["policy_target"])
        ddpg.Q_target.load_state_dict(ckpt["Q_target"])
        ddpg.optimizer.load_state_dict(ckpt["policy_opt"])

    base_dir = Path('checkpoints')
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f'{ts}-{env_name}-DDPG-eps{eps}-l{lr}-{run_id}'

    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True)

    rewards = []
    lengths = []
    losses = []
    lrs= []

    def save_statistics(out_file):
        stats = {
            "rewards" : rewards, 
            "lengths": lengths,
            "eps": eps,
            "train": train_iter,
            "lr": lr, 
            "update_every": opts.update_every,
            "losses": losses,
            "lrs": lrs,
        }
        with open(out_file, 'wb') as f:
            pickle.dump(stats, f)

    
    #fill_buffer(env, ddpg) # self-play to fill the buffer with transitions
    # training loop
    weak_agent = h_env.BasicOpponent(weak=True)
    strong_agent = h_env.BasicOpponent(weak=False)
    pool = AgentPool(max_agents=10, static_agents=2*[weak_agent, strong_agent])
    for i_episode in range(1, max_episodes+1):
        ddpg.reset()
        total_reward=0
        opponent = pool.get_agent()
        is_player_one = random.random() <= 0.5
        explore = float(random.random() <= 0.9)

        obs_agent1, _info = env.reset()  #this line is responsible for making it random who starts?
        obs_agent2 = env.obs_agent_two()
        for t in range(max_timesteps):
            if is_player_one:
                a1 = ddpg.act(obs_agent1, eps=explore)
                a2 = opponent.act(obs_agent2)  
            else:
                a1 = opponent.act(obs_agent1)
                a2 = ddpg.act(obs_agent2, eps=explore)
            
            obs_agent1_new, _, done, trunc, info_agent1 = env.step(np.hstack([a1, a2]))
            info_agent2 = env.get_info_agent_two()
            obs_agent2_new = env.obs_agent_two()

            if is_player_one:
                reward = get_reward(info_agent1)
                action = a1
                obs = obs_agent1
                obs_new = obs_agent1_new
            else:
                reward = get_reward(info_agent2)
                action = a2
                obs = obs_agent2
                obs_new = obs_agent2_new

            total_reward+= reward
            ddpg.store_transition((obs, action, reward, obs_new, done))

            if done or trunc:
                break

        if i_episode < 1000:
            if (i_episode + 1) % 200 == 0:
                print(f'Warmup {i_episode+1}/1000')
            continue  # Warmup

        losses_epoch, lrs_epoch = ddpg.train(train_iter)
        if opts.lr_scheduler:
            ddpg.scheduler.step()
        losses.extend(losses_epoch)
        lrs.extend(lrs_epoch)

        rewards.append(total_reward)
        lengths.append(t)

        # update pool
        if i_episode >= 10_000 and i_episode % 1000 == 0:
            print('Adding current agent to pool')
            new_agent = ddpg.clone(ddpg)
            new_agent._eps = random.random() * eps  # between 0 - eps
            pool.add_agent(new_agent)

        # save checkpoint every 500 episodes
        if i_episode % 500 == 0:
            
            print("########## Saving a checkpoint... ##########")
            checkpoint_path = run_dir / f'checkpoint_{i_episode}-t{train_iter}.pth'
            stat_path = run_dir / f'stats_{i_episode}-t{train_iter}.pth'
            fig_path = run_dir / f'figures_{i_episode}-t{train_iter}'
            fig_path.mkdir()

            state_dict = {
                "policy": ddpg.policy.state_dict(),
                "Q": ddpg.Q.state_dict(),
                "policy_target": ddpg.policy_target.state_dict(),
                "Q_target": ddpg.Q_target.state_dict(),
                "policy_opt": ddpg.optimizer.state_dict(),
            }
            torch.save(state_dict, checkpoint_path)
            save_statistics(stat_path)
            evaluate(fig_path, stat_path, checkpoint_path)

        # logging
        if i_episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))
            avg_lr = np.mean(lrs[-log_interval:]) if lrs else 0
            print('Episode {} \t avg length: {} \t reward: {} \t avg lr: {}'.format(i_episode, avg_length, avg_reward, avg_lr), flush=True)
    stat_path = save_statistics()
    evaluate(stat_path, checkpoint_path)

if __name__ == '__main__':
    run()
