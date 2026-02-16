import secrets
import numpy as np
import hockey.hockey_env as h_env

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

def get_opponent(players):
    pass