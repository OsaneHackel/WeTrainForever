from __future__ import annotations

import argparse
import uuid
import os
import sys

import hockey.hockey_env as h_env
import numpy as np
from gymnasium import spaces # added
import torch

# for collecting data from games
import pickle
from pathlib import Path


from comprl.client import Agent, launch_client

# TD3 agent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from TD3_agent import TD3_Agent


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, weak: bool) -> None:
        super().__init__()

        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        # NOTE: If your agent is using discrete actions (0-7), you can use
        # HockeyEnv.discrete_to_continous_action to convert the action:
        #
        # from hockey.hockey_env import HockeyEnv
        # env = HockeyEnv()
        # continuous_action = env.discrete_to_continous_action(discrete_action)

        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )

# my agent
class TD3Agent(Agent):
    def __init__(self, saved_agent_path: str) -> None:
        super().__init__()

        # set up dimensions via dummy env
        dummy_env = h_env.HockeyEnv()
        obs_dim = dummy_env.observation_space.shape[0]
        full_action_space = dummy_env.action_space
        agent_act_dim = full_action_space.shape[0] // 2
        agent_action_space = spaces.Box(
            low=full_action_space.low[:agent_act_dim],
            high=full_action_space.high[:agent_act_dim],
            dtype=full_action_space.dtype
        )

        self.hockey_agent = TD3_Agent(
            obs_dim=obs_dim,
            act_dim=agent_act_dim,
            observation_space=dummy_env.observation_space,
            action_space=agent_action_space, 
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # load agent from the saved path
        self.hockey_agent.load(saved_agent_path)
        print(f"Loaded TD3 agent from {saved_agent_path}")

    def get_step(self,
                 observation: list[float]) -> list[float]:
        obs = np.array(observation, dtype=np.float32)
        action = self.hockey_agent.select_action(observation=obs,
                                                 explore=False)
        return action.tolist()
    
    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )
        
class TD3MemoAgent(Agent):
    """
    TD3 agent that records all transitions for offline training.
    """
    def __init__(self, saved_agent_path: str, save_dir: str = "tournament_replay") -> None:
        super().__init__()

        dummy_env = h_env.HockeyEnv()
        obs_dim = dummy_env.observation_space.shape[0]
        full_action_space = dummy_env.action_space
        agent_act_dim = full_action_space.shape[0] // 2
        agent_action_space = spaces.Box(
            low=full_action_space.low[:agent_act_dim],
            high=full_action_space.high[:agent_act_dim],
            dtype=full_action_space.dtype
        )
        self.hockey_agent = TD3_Agent(
            obs_dim=obs_dim,
            act_dim=agent_act_dim,
            observation_space=dummy_env.observation_space,
            action_space=agent_action_space,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.hockey_agent.load(saved_agent_path)
        print(f"Loaded TD3 agent from {saved_agent_path}")

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.prev_state = None
        self.prev_action = None
        self.episode_buffer = []
        self.current_game_id = None
        self.games_collected = 0
        
    def get_step(self, observation: list[float]) -> list[float]:
        state = np.array(observation, dtype=np.float32)

        if self.prev_state is not None:
            self.episode_buffer.append({
                "state": self.prev_state,
                "action": self.prev_action,
                "next_state": state,
                "done": False,
            })

        action = self.hockey_agent.select_action(observation=state, explore=False)
        self.prev_state = state
        self.prev_action = action
        return action.tolist()
    
    def on_start_game(self, game_id) -> None:
        self.prev_state = None
        self.prev_action = None
        self.episode_buffer = []
        self.current_game_id = str(uuid.UUID(int=int.from_bytes(game_id)))
        print(f"Game started (id: {self.current_game_id})")
        
    def on_end_game(self, result: bool, stats: list[float]) -> None:
        if self.prev_state is not None:
            self.episode_buffer.append({
                "state": self.prev_state,
                "action": self.prev_action,
                "next_state": self.prev_state,
                "done": True,
            }) 
             
        self.games_collected += 1
        data = {
            "transitions": self.episode_buffer,
            "won": result,
            "stats": stats,
        }
        save_path = self.save_dir / f"{self.current_game_id}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        text_result = "won" if result else "lost"
        print(f"Game {text_result} | {len(self.episode_buffer)} transitions saved "
              f"| score: {stats[0]:.1f} vs {stats[1]:.1f} "
              f"| total: {self.games_collected}")
        

# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["weak", "strong", "random", "TD3", "TD3_memo"],
        default="weak",
        help="Which agent to use",
    )
    parser.add_argument(
        "--saved_agent_path",
        type=str,
        default=None,
        help="Path to TD3 agent (.pt)"
    )
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    agent: Agent
    if args.agent == "weak":
        agent = HockeyAgent(weak=True)
    elif args.agent == "strong":
        agent = HockeyAgent(weak=False)
    elif args.agent == "random":
        agent = RandomAgent()
    elif args.agent == "TD3":
        if args.saved_agent_path is None:
            raise ValueError("--saved_agent_path is required when using --agent=TD3")
        agent = TD3Agent(saved_agent_path=args.saved_agent_path)
    elif args.agent == "TD3_memo":
        if args.saved_agent_path is None:
            raise ValueError("--saved_agent_path is required when using --agent==TD3_memo")
        agent = TD3MemoAgent(saved_agent_path=args.saved_agent_path)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
