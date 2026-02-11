import numpy as np
import hockey.hockey_env as h_env
import gymnasium as gym
from importlib import reload
import time

np.set_printoptions(suppress=True)

env = h_env.HockeyEnv()

obs, info = env.reset()
obs_agent2 = env.obs_agent_two()
_ = env.render()

print(obs)
print(obs_agent2)