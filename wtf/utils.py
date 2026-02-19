import secrets
import torch
import numpy as np
import hockey.hockey_env as h_env
from gymnasium import spaces
from wtf.agents.DDPG import DDPGAgent
from wtf.agents.SAC import SAC

def load_agent(checkpoint_path, which_agent, evaluate):
    env = h_env.HockeyEnv()
    full_action_space = env.action_space 
    n_actions_per_player = full_action_space.shape[0] // 2
    agent_action_space = spaces.Box(low=full_action_space.low[:n_actions_per_player],
                                    high=full_action_space.high[:n_actions_per_player],
                                    dtype=full_action_space.dtype)
    if which_agent == "DDPG":
        agent = DDPGAgent(env.observation_space, agent_action_space)  
    elif which_agent == "SAC":
        agent = SAC(env.observation_space, agent_action_space)
    agent = load_weights(which_agent, agent, checkpoint_path, evaluate = evaluate)
    return agent

def load_weights(which_agent, agent, ckpt_path, evaluate):
    map_location = None
    if evaluate: 
        map_location ="cpu"
    ckpt= torch.load(ckpt_path, map_location=map_location)
    if which_agent == "DDPG":
        agent.policy.load_state_dict(ckpt["policy"])
        agent.Q.load_state_dict(ckpt["Q"])
        agent.policy_target.load_state_dict(ckpt["policy_target"])
        agent.Q_target.load_state_dict(ckpt["Q_target"])
        agent.optimizer.load_state_dict(ckpt["policy_opt"])
        if evaluate: 
            agent._eps = 0.0

    elif which_agent == "SAC":
        agent.policy.load_state_dict(ckpt['policy_state_dict'])
        agent.critic.load_state_dict(ckpt['critic_state_dict'])
        agent.critic_target.load_state_dict(ckpt['critic_target_state_dict'])
        agent.critic_optim.load_state_dict(ckpt['critic_optimizer_state_dict'])
        agent.policy_optim.load_state_dict(ckpt['policy_optimizer_state_dict'])
        if evaluate:
                agent.policy.eval()
                agent.critic.eval()
                agent.critic_target.eval()
        else:
            agent.policy.train()
            agent.critic.train()
            agent.critic_target.train()
    else: 
        print("please specify which agent to load")
    return agent

def create_state_dict(which_agent,agent):
    if which_agent == "DDPG":
        state_dict = {
                "policy": agent.policy.state_dict(),
                "Q": agent.Q.state_dict(),
                "policy_target": agent.policy_target.state_dict(),
                "Q_target": agent.Q_target.state_dict(),
                "policy_opt": agent.optimizer.state_dict(),
            }
    elif which_agent == "SAC":
        state_dict = {
                'policy_state_dict': agent.policy.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'critic_target_state_dict': agent.critic_target.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optim.state_dict(),
                'policy_optimizer_state_dict': agent.policy_optim.state_dict(),
            }
    return state_dict

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
