import torch 
import numpy as np
import hockey.hockey_env as hockey_env
from TD3_agent import TD3_Agent

def main():
    env = hockey_env.HockeyEnv()

    TD3 = TD3_Agent(env.observation_space, 
                    )
    
    for t in range(total_env_steps):
        state = 
        # select action (inludes exploration noise if
        # explore = True)
        action = TD3.select_action(observation=state,
                              explore=True)
        # observe r and s_next
        (s_next, r, terminated, truncated, info) = env.step(action) 
        ended = terminated or truncated

        # store transition in replay buffer
        TD3.buffer.add_experience(
            state = state,
            action = action,
            reward = r,
            next_state= s_next,
            ended = ended
        )

        # advance the state
        state = s_next if not ended else env.reset()[0]

        # TODO: alternatively, use the train method directly?
        # training 
        if t >= TD3._params["warmup_steps"]:
            TD3.train_step() 

