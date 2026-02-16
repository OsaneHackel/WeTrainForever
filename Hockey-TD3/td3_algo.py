import numpy as np 
import torch 
import torch.nn as nn
import gymnasium as gym 
#import hockey-env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3():
    def __init__(
            self, gamma, actor, critic
    ) -> None:
        self.gamma = gamma,
        self.update_method = None, # either slowly, periodically
        self.actor = actor,
        self.critic = critic


    def train_agent(self) -> None:
        raise NotImplementedError
        # update periodically not bit by bit
        for t in self.