import secrets
import torch
import numpy as np
import hockey.hockey_env as h_env
from wtf.agents.DDPG import DDPGAgent

def load_ddpg(checkpoint_path):
    env = h_env.HockeyEnv()
    ddpg = DDPGAgent(env.observation_space, env.action_space)  
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt= torch.load(checkpoint_path, map_location='cpu')
    ddpg.policy.load_state_dict(ckpt["policy"])
    ddpg.Q.load_state_dict(ckpt["Q"])
    ddpg.policy_target.load_state_dict(ckpt["policy_target"])
    ddpg.Q_target.load_state_dict(ckpt["Q_target"])
    ddpg.optimizer.load_state_dict(ckpt["policy_opt"])
    ddpg.exploit = True
    ddpg._eps = 0.0
    return ddpg

def generate_id() -> str:
    s = secrets.token_urlsafe(5)
    return s.replace('-', 'a').replace('_', 'b')

def fill_buffer(env, ddpg):
    opponent = h_env.BasicOpponent(weak=True)
    obs, _ = env.reset()
    obs_opponent = env.obs_agent_two()
    print("filling the buffer")
    for i in range(100000):
        # opponent plays agent 2
        a_op = opponent.act(obs_opponent)
        a = ddpg.policy.predict(obs)  # or random action
        joint_action = np.hstack([a, a_op])
        obs_next, r, done, trunc, _ = env.step(joint_action)
        ddpg.store_transition((obs, a, r, obs_next, done))
        if i % 10000 == 0:
            print(f"filled {i} transitions")
        obs = obs_next
        obs_opponent = env.obs_agent_two()
        if done or trunc:
            obs, _ = env.reset()
            obs_opponent = env.obs_agent_two()
