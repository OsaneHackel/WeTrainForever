import numpy as np
import pickle
import torch
import imageio

import hockey.hockey_env as h_env

import wtf.plotting as plo
from wtf.agents.DDPG import DDPGAgent

def load_ddpg(checkpoint_path):
    env = h_env.HockeyEnv()
    ddpg = DDPGAgent(env.observation_space, env.action_space)  
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt= torch.load(checkpoint_path)
    ddpg.policy.load_state_dict(ckpt["policy"])
    ddpg.Q.load_state_dict(ckpt["Q"])
    ddpg.policy_target.load_state_dict(ckpt["policy_target"])
    ddpg.Q_target.load_state_dict(ckpt["Q_target"])
    ddpg.optimizer.load_state_dict(ckpt["policy_opt"])
    return ddpg

def simulate(checkpoint_path, save_path):
    env = h_env.HockeyEnv()
    ddpg = load_ddpg(checkpoint_path)
    opponent = h_env.BasicOpponent(weak=False)
    obs, _ = env.reset()
    obs_opponent = env.obs_agent_two()
    frames = []
    for t in range(200):
        a_op = opponent.act(obs_opponent)
        a = ddpg.policy.predict(obs)  # or random action
        joint_action = np.hstack([a, a_op])
        obs_next, r, done, trunc, _ = env.step(joint_action)
        frames.append(env.render(mode='rgb_array'))
        obs = obs_next
        obs_opponent = env.obs_agent_two()
        if done or trunc:
            break
    
    imageio.mimwrite(save_path / "simulation.gif", frames, fps=30)

def evaluate(out_dir, stat_path, checkpoint_path=None):
    with open(stat_path, 'rb') as f:
        stats = pickle.load(f)
    plo.plot_rewards(stats['rewards'], out_dir)
    plo.plot_lrs(stats['lrs'], out_dir)
    simulate(checkpoint_path, out_dir)

if __name__ == '__main__':
    #stat_path = 'results/Z5qI6oc-DDPG_HockeyEnv-eps0.1-t32-l0.0001-sNone-Adam-scheduler-False.pkl'
    #checkpoint_path ="results/Z5qI6oc-DDPG_HockeyEnv_9500-eps0.1-t32-l0.0001-sNone-Adam-scheduler-False.pth"
    stat_path2 ='results/oAMWN4k-DDPG_HockeyEnv-eps0.1-t32-l0.0001-sNone-Adam-scheduler-False.pkl'
    checkpoint_path2 ='results/oAMWN4k-DDPG_HockeyEnv_10000-eps0.1-t32-l0.0001-sNone-Adam-scheduler-False.pth'
    
    stat_path = 'results/fuzikCY-DDPG_HockeyEnv-eps0.3-t32-l0.0001-sNone-Adam-scheduler-False.pkl'
    checkpoint_path ="results/fuzikCY-DDPG_HockeyEnv_1500-eps0.3-t32-l0.0001-sNone-Adam-scheduler-False.pth"
    evaluate(stat_path, checkpoint_path)
    evaluate(stat_path2, checkpoint_path2)
    
    