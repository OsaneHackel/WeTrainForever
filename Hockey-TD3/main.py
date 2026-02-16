import torch 
import numpy as np
import csv
import gymnasium
from gymnasium import spaces
import hockey.hockey_env as hockey_env
from TD3_agent import TD3_Agent

def make_opponent(opponent_type, 
                  saved_agent_path = None):
    """
    Creates the opponent agent
    opponent_type: "weak", "strong", "current_self", "pretrained_self"
    """
    if opponent_type == "weak":
        return hockey_env.BasicOpponent(weak=True)
    elif opponent_type == "strong":
        return hockey_env.BasicOpponent(weak=False)
    elif opponent_type == "current_self":
        # handle this separately in the training loop
        return None 
    elif opponent_type == "pretrained_self":
        # load previously saved TD3 agent as opponent
        if saved_agent_path is None:
            raise ValueError("Need --saved_agent_path for playing against a pretrained TD3 agent")
        # create a dummy env
        dummy_env = hockey_env.HockeyEnv()
        full_action_space = dummy_env.action_space
        n_per_player = full_action_space.shape[0] // 2
        opponent = TD3_Agent(
            obs_dim = dummy_env.observation_space.shape[0],
            observation_space = dummy_env.observation_space,
            act_dim = n_per_player,
            action_space = spaces.Box(-1,+1, (n_per_player,), dtype=np.float32)
        )
        opponent.load(saved_agent_path)
        return opponent
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    
def get_opponent_action(opponent, 
                        opponent_type,
                        agent, 
                        obs_agent2):
    """
    returns: actions for player2 (opponent) depending on opponent_type
    """
    if opponent_type == "pretrained_self":
        # current agent plays against a pretrained version of TD3
        # keep the actions of this pretrained self deterministic => easier to learn for agent1
        return opponent.select_action(obs_agent2, 
                                      explore = False) 
    elif isinstance(opponent, hockey_env.BasicOpponent):
        return opponent.act(obs_agent2)
    elif opponent_type == "current_self":
        # current agent plays against itself
        return agent.select_action(obs_agent2, explore = True)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

def train():
    env = hockey_env.HockeyEnv()

    full_action_space = env.action_space # actions for player1 ||
    n_actions_per_player = full_action_space.shape[0] // 2
    agent_action_space = spaces.Box(low=full_action_space.low[:n_actions_per_player],
                                    high=full_action_space.high[:n_actions_per_player],
                                    dtype=full_action_space.dtype)
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

            # log rewards

            if t % save_every == 0:
                TD3.save(f"saves/td3_{t}.pt")


if __name__ == "__main__":
    main()