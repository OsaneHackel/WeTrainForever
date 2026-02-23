import torch 
import numpy as np
import gymnasium
from gymnasium import spaces
import hockey.hockey_env as hockey_env
from TD3_agent import TD3_Agent
import os
import sys

# to train 
class SACOpponentWrapper:
    """
    wrapper around Osane's SAC
    """
    def __init__(self, sac_agent):
        self._agent = sac_agent

    def act(self, obs):
        return self._agent.act(obs, eps=0)

class OpponentPool:
    def __init__(self, 
                 seed: int
                 ):
        self._opponents = []
        self._weights = []
        self._rng = np.random.RandomState(seed=seed)

    def add_opponent(self,
                     name: str,
                     opponent,
                     weight: float):
        self._opponents.append((name, opponent))
        self._weights.append(weight)

    def update_TD3_opponent(self, 
                            checkpoint_path: str):
        new_TD3_opponent = _load_frozen_TD3(checkpoint_path)
        new_name = checkpoint_path.split("/")[-1].replace(".pt", "")

        # find existing TD3 opponent
        td3_indices = [idx for idx, (name, _) in enumerate(self._opponents) 
                       if name.startswith("td3")]
        if td3_indices:
            idx = td3_indices[0]
            old_name = self._opponents[idx][0]
            self._opponents[idx] = (new_name, new_TD3_opponent)
            print(f"OpponentPool: replaced '{old_name}' with '{new_name}")
        #else:
        #    self.add_opponent(new_name, new_TD3_opponent,)
        print(self.__current_state__())

    def sample(self):
        # normalize the weight to get probs
        ps = np.array(self._weights, dtype=np.float64)
        ps /= ps.sum()
        idx_sampled = self._rng.choice(len(self._opponents), p=ps)
        return self._opponents[idx_sampled] # (name, opponent)
    
    def __current_state__(self):
        lines = []
        weight_sum = sum(self._weights)
        for (name, _), w in zip(self._opponents, self._weights):
            lines.append(f"{name}: weight={w} ({w/weight_sum:.3%})")
        printout = "OpponentPool:\n" + "\n".join(lines)
        return printout
    
def _load_frozen_TD3(saved_agent_path: str) -> TD3_Agent:
    # create a dummy env
    dummy_env = hockey_env.HockeyEnv()
    full_action_space = dummy_env.action_space
    n_per_player = full_action_space.shape[0] // 2
    TD3agent = TD3_Agent(
        obs_dim = dummy_env.observation_space.shape[0],
        observation_space = dummy_env.observation_space,
        act_dim = n_per_player,
        action_space = spaces.Box(-1,+1, (n_per_player,), dtype=np.float32)
    )
    TD3agent.load(saved_agent_path)
    return TD3agent

def _load_frozen_SAC(checkpoint_path: str, 
                     sac_repo_path: str = None) -> SACOpponentWrapper:
    """
    checkpoint_path : Path to the .pth file.
    """
    if sac_repo_path is None:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        sac_repo_path = os.path.abspath(os.path.join(this_file_dir, ".."))
    
    if sac_repo_path not in sys.path:
        sys.path.insert(0, sac_repo_path)
        print(f"Added '{sac_repo_path}' to sys.path")

    from wtf.agents.SAC import SAC

    env = hockey_env.HockeyEnv()
    full_action_space = env.action_space
    n_per_player = full_action_space.shape[0] // 2
    agent_action_space = spaces.Box(
        low=full_action_space.low[:n_per_player],
        high=full_action_space.high[:n_per_player],
        dtype=full_action_space.dtype,
    )
    sac_agent = SAC(env.observation_space, agent_action_space)

    # load SAC weights
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sac_agent.policy.load_state_dict(ckpt['policy_state_dict'])
    sac_agent.critic.load_state_dict(ckpt['critic_state_dict'])
    sac_agent.critic_target.load_state_dict(ckpt['critic_target_state_dict'])
    sac_agent.critic_optim.load_state_dict(ckpt['critic_optimizer_state_dict'])
    sac_agent.policy_optim.load_state_dict(ckpt['policy_optimizer_state_dict'])
    sac_agent.policy.eval()
    sac_agent.critic.eval()
    sac_agent.critic_target.eval()

    print(f"Loaded SAC checkpoint from {checkpoint_path}")
    return SACOpponentWrapper(sac_agent)

def make_opponent(opponent_type, 
                  saved_agent_path = None,
                  seed = None, 
                  opponent_odds:  dict = None,
                  sac_path: str = None,
                  sac_folder_path: str = None):
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
        opponent = _load_frozen_TD3(saved_agent_path)
        return opponent
    elif opponent_type == "pool_basic_and_frozen_self":
        if seed is None:
            raise ValueError("Need a seed for OppoentPool")
        if opponent_odds is None:
            raise ValueError("Need dict (opponent_name: int) to derive opponent ratios for OppoentPool")
        if saved_agent_path is None:
            raise ValueError("Need --saved_agent_path for pool_basic_and_frozen_self")
        pool = OpponentPool(seed=seed)
        pool.add_opponent("weak", 
                 hockey_env.BasicOpponent(weak=True), 
                 weight = opponent_odds["weak"])
        pool.add_opponent("strong", 
                 hockey_env.BasicOpponent(weak=False), 
                 weight = opponent_odds["strong"])
        # add the frozen self
        frozen_agent = _load_frozen_TD3(saved_agent_path=saved_agent_path)
        name = saved_agent_path.split("/")[-1].replace(".pt", "")
        pool.add_opponent(name, frozen_agent, opponent_odds["frozen_agent"])
        print(pool)
        return pool
    elif opponent_type == "pool_with_sac":
        if seed is None:
            raise ValueError("Need a seed for OpponentPool")
        if opponent_odds is None:
            raise ValueError("Need dict with keys: weak, strong, sac")
        if sac_path is None:
            raise ValueError("Need --sac_path for SAC opponent in pool (.pth)")
        pool = OpponentPool(seed=seed)
        pool.add_opponent("weak",
                 hockey_env.BasicOpponent(weak=True),
                 weight=opponent_odds["weak"])
        pool.add_opponent("strong",
                 hockey_env.BasicOpponent(weak=False),
                 weight=opponent_odds["strong"])
        sac_opponent = _load_frozen_SAC(sac_path, sac_folder_path)
        pool.add_opponent("sac", sac_opponent, weight=opponent_odds["sac"])
        # optionally also add a frozen TD3 self
        if saved_agent_path is not None and "frozen_agent" in opponent_odds:
            frozen_agent = _load_frozen_TD3(saved_agent_path)
            name = saved_agent_path.split("/")[-1].replace(".pt", "")
            pool.add_opponent(name, frozen_agent, weight=opponent_odds["frozen_agent"])
        print(pool.__current_state__())
        return pool
    elif opponent_type == "sac":
        if sac_path is None:
            raise ValueError("Need --sac_path for SAC opponent (.pth)")
        return _load_frozen_SAC(sac_path, sac_folder_path)
    elif opponent_type == "curriculum_basic_and_current_self":
        pass
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    
def get_opponent_action(opponent, 
                        opponent_type,
                        agent, 
                        obs_agent2):
    """
    returns: actions for player2 (opponent) depending on opponent_type
    """
    # current self (TD3 agent) does explore
    if opponent_type == "current_self":
        return agent.select_action(observation = obs_agent2,
                                   explore = True)
    # pretrained self or intermediate TD3 checkpoint
    if isinstance(opponent, TD3_Agent):
        return opponent.select_action(observation = obs_agent2,
                                      explore = False)
    if isinstance(opponent, hockey_env.BasicOpponent):
        return opponent.act(obs= obs_agent2)
    # SAC opponent
    if isinstance(opponent, SACOpponentWrapper):
        return opponent.act(obs_agent2)
    raise ValueError(f"Opponent is of unknown instance: {type(opponent_type)}")
    