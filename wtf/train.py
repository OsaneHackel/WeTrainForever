import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import gymnasium as gym
import optparse
import pickle
import hockey.hockey_env as h_env
import secrets

from wtf.agents.DDPG import DDPGAgent
from wtf.utils import generate_id, fill_buffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)
    

def run():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env',action='store', type='string',
                         dest='env_name',default="HockeyEnv",
                         help='Environment (default %default)')
    optParser.add_option('-n', '--eps',action='store',  type='float',
                         dest='eps',default=0.1,
                         help='Policy noise (default %default)')
    optParser.add_option('-t', '--train',action='store',  type='int',
                         dest='train',default=32,
                         help='number of training batches per episode (default %default)')
    optParser.add_option('-l', '--lr',action='store',  type='float',
                         dest='lr',default=0.0001,
                         help='learning rate for actor/policy (default %default)')
    optParser.add_option('-m', '--maxepisodes',action='store',  type='float',
                         dest='max_episodes',default=2000,
                         help='number of episodes (default %default)')
    optParser.add_option('-u', '--update',action='store',  type='float',
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
    else:
        env = gym.make(env_name)
    render = False
    log_interval = 20           # print avg reward in the interval
    max_episodes = opts.max_episodes # max training episodes
    max_timesteps = 2000         # max timesteps in one episode

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

    ddpg = DDPGAgent(env.observation_space, env.action_space, eps = eps, learning_rate_actor = lr,
                        update_target_every = opts.update_every, max_episodes = max_episodes, 
                        optimizer = optimizer, lr_scheduler = opts.lr_scheduler)
    
    if opts.checkpoint_path is not None:
        print(f"Loading checkpoint from {opts.checkpoint_path}...")
        ckpt= torch.load(f"results/{opts.checkpoint_path}")
        ddpg.policy.load_state_dict(ckpt["policy"])
        ddpg.Q.load_state_dict(ckpt["Q"])
        ddpg.policy_target.load_state_dict(ckpt["policy_target"])
        ddpg.Q_target.load_state_dict(ckpt["Q_target"])
        ddpg.optimizer.load_state_dict(ckpt["policy_opt"])

    rewards = []
    lengths = []
    losses = []
    lrs= []

    timestep = 0

    def save_statistics():
        with open(f"./results/{run_id}-DDPG_{env_name}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}-{optimizer}-scheduler-{opts.lr_scheduler}.pkl", 'wb') as f:
            pickle.dump({"rewards" : rewards, "lengths": lengths, "eps": eps, "train": train_iter,
                         "lr": lr, "update_every": opts.update_every, "losses": losses, "lrs": lrs}, f)

    
    #fill_buffer(env, ddpg) # self-play to fill the buffer with transitions
    # training loop
    for i_episode in range(1, max_episodes+1):
        ob, _info = env.reset()
        ddpg.reset()
        total_reward=0
        for t in range(max_timesteps):
            timestep += 1
            done = False
            a = ddpg.act(ob)
            (ob_new, reward, done, trunc, _info) = env.step(a)
            total_reward+= reward
            ddpg.store_transition((ob, a, reward, ob_new, done))
            ob=ob_new
            if done or trunc: break
        losses,lrs = ddpg.train(train_iter)
        if opts.lr_scheduler:
            ddpg.scheduler.step()
        losses.extend(losses)
        lrs.extend(lrs)

        rewards.append(total_reward)
        lengths.append(t)

        # save every 500 episodes
        if i_episode % 500 == 0:
            print("########## Saving a checkpoint... ##########")
            path = f'./results/{run_id}-DDPG_{env_name}_{i_episode}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}-{optimizer}-scheduler-{opts.lr_scheduler}.pth'
            #torch.save(ddpg.state(), path)
            torch.save({
                "policy": ddpg.policy.state_dict(),
                "Q": ddpg.Q.state_dict(),
                "policy_target": ddpg.policy_target.state_dict(),
                "Q_target": ddpg.Q_target.state_dict(),
                "policy_opt": ddpg.optimizer.state_dict(),
            }, path)
            save_statistics()

        # logging
        if i_episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))
            avg_lr = np.mean(lrs[-log_interval:]) if lrs else 0

            print('Episode {} \t avg length: {} \t reward: {} \t avg lr: {}'.format(i_episode, avg_length, avg_reward, avg_lr))
    save_statistics()

if __name__ == '__main__':
    run()
