import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class Replay_DataBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    ended: np.ndarray

class ReplayBuffer:
    """
    Experience replay buffe (NumPy storage)
    - usage breaks correlation in sequential IRL experiences
    - enables sample reuse (off-policy) => less expensive than actual env interactions
    """
    def __init__(self, 
                 obs_dim: int,
                 act_dim: int,
                 max_capacity: int = int(1e6),
                 device = None,
                 *,
                 seed: Optional[int] = None,
                 dtype = np.float32 # default precision
                 ) -> None:
        if max_capacity <= 0:
            raise ValueError("max_capacity must be > 0")
        # max number of transitions in buffer
        self.max_capacity = max_capacity
        self.device = torch.device(device) 
        
        # initialize using seed
        self._rng = np.random.default_rng(seed)

        # buffer
        self._states  = np.zeros((self.max_capacity, obs_dim), 
                                 dtype=dtype)
        self._actions = np.zeros((self.max_capacity, act_dim), 
                                 dtype=dtype)
        self._rewards  = np.zeros((self.max_capacity, 1), 
                                  dtype=dtype)
        self._next_states = np.zeros((self.max_capacity, obs_dim), 
                                     dtype=dtype)
        self._ended  = np.zeros((self.max_capacity, 1), 
                                dtype=dtype)

        # attributes for keeping track of buffer
        self._pointer = 0   # to next index
        self._n_stored = 0  # currently

    def add_experience(self,
                       state, 
                       action, 
                       reward,
                       next_state, 
                       ended) -> None:
        """
        ended: bool-like, True if episode ended at next_state (
        terminated through e..g. goal/end of match? or truncated 
        e.g. due to time limit?
        )
        """
        idx = self._pointer

        # write into buffer
        self._states[idx]  = np.asarray(state, dtype=self._states.dtype)
        self._actions[idx] = np.asarray(action, dtype=self._actions.dtype)
        self._rewards[idx, 0]  = np.asarray(reward, dtype=self._rewards.dtype)
        self._next_states[idx] = np.asarray(next_state, dtype=self._next_states.dtype)
        self._ended[idx, 0] = 1.0 if bool(ended) else 0.0 # TODO: check
    
        # advance the pointer 
        self._pointer = (self._pointer + 1) % self.max_capacity
        # optionally increase the size
        self._n_stored = min(self._n_stored + 1, self.max_capacity)

    def sample(self, batch_size: int) -> Replay_DataBatch:
        if self._n_stored == 0:
            raise RuntimeError("Replay buffer is empty")
        B = min(batch_size, self._n_stored)
        
        # samples indices
        idxs = self._rng.integers(0, self._n_stored, size=B, endpoint=False)
        
        replay_batch = Replay_DataBatch(
            states  = self._states[idxs],
            actions = self._actions[idxs],
            rewards = self._rewards[idxs],
            next_states = self._next_states[idxs],
            ended = self._ended[idxs]
        )
        
        return replay_batch
    
    def create_checkpoint(self) -> dict:
        return {
            "max_capacity": self.max_capacity,
            "_pointer": self._pointer,
            "_n_stored": self._n_stored,
            "_states": self._states,
            "_actions": self._actions,
            "_rewards": self._rewards,
            "_next_states": self._next_states,
            "_ended": self._ended,
            "_rng_state": self._rng.bit_generator.state,
        }
    
    def load_checkpoint(self, checkpoint: dict) -> None:
        self.max_capacity = checkpoint["max_capacity"]
        self._pointer = int(checkpoint["_pointer"])
        self._n_stored = int(checkpoint["_n_stored"])

        self._states[:] = checkpoint["_states"]
        self._actions[:] = checkpoint["_actions"]
        self._rewards[:] = checkpoint["_rewards"]
        self._next_states[:] = checkpoint["_next_states"]
        self._ended[:] = checkpoint["_ended"]

        if not hasattr(self, "_rng") or self._rng is None:
            self._rng = np.random.default_rng()
        self._rng.bit_generator.state = checkpoint["_rng_state"]

    
    def sample_torch(self, batch_size: int, *, pin_memory: bool = False):
        """
        Samples and returns torch tensors on self.device
        Use this only once sure that everything runs (i.e. after debugging)
        """
        replay_batch = self.sample(batch_size)
        def to_torch(x: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(x)  # shares CPU memory with numpy
            if pin_memory and self.device.type == "cuda":
                t = t.pin_memory()
                return t.to(self.device, non_blocking=True)
            return t.to(self.device)
        # TODO: replay buffer is on CPU, but training is on GPU
        return (
            to_torch(replay_batch.states),
            to_torch(replay_batch.actions),
            to_torch(replay_batch.rewards),
            to_torch(replay_batch.next_states),
            to_torch(replay_batch.ended),
        )
    
class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Instead of sampling previous transitions uniformly (as in
    ReplayBuffer), sample transitions proportional to their TD error
    => higher TD error => less expected => replay more often!
    """
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_capacity: int = int(1e6),
                 device = None,
                 *,
                 seed: Optional[int] = None,
                 dtype = np.float32,
                 alpha: float,
                 beta_init: float,
                 beta_n_steps: int
                 ) -> None:
        super().__init__(obs_dim=obs_dim,
                         act_dim=act_dim,
                         max_capacity=max_capacity,
                         device=device, 
                         seed=seed, 
                         dtype=dtype)
        self._alpha = alpha
        self._beta_init = beta_init
        self._beta_n_steps = beta_n_steps
        self._beta_step_counter = 0

        # initialize all samples to have the same priority
        self._priorities = np.ones(max_capacity, dtype=dtype)
        self._max_priority = 1.0

    def add_experience(self, state, action, reward, next_state, ended) -> None:
        # 1. overwrite priority with current max priority ()
        # max priority because no TD error during 
        # real-world interactions (comes during training)
        # will be updated later
        self._priorities[self._pointer] = self._max_priority 
        super().add_experience(state, action, reward, next_state, ended)

    def update_priorities(self, idx_sampled: np.ndarray, 
                          td_errors: np.ndarray,
                          epsilon: float = 1e-6) -> None:
        new_priorities = (np.abs(td_errors) + epsilon).astype(self._priorities.dtype)
        self._priorities[idx_sampled] = new_priorities
        self._max_priority = max(self._max_priority, new_priorities.max()) 

    def sample_prioritized(self,
                           batch_size: int,
                           *,
                           pin_memory: bool = False):
        """
        Returns: 
        - transitions as tensors for the critic forward pass
        - sampled transition indices
        - importance sampling correction weights
        """
        if self._n_stored == 0:
            raise RuntimeError("Replay buffer is empty")
        
        B = min(batch_size, self._n_stored)

        # increase the importance sampling annealing step
        self._beta_step_counter += 1
        candidate_beta = self._beta_init + (1.0-self._beta_init)* self._beta_step_counter/ self._beta_n_steps  
        beta = min(1.0, candidate_beta)

        # sampling probabilities
        priorities = self._priorities[:self._n_stored]
        priorities_alpha = priorities ** self._alpha
        probs = priorities_alpha / priorities_alpha.sum()

        # sample accordingly
        idx_sampled = self._rng.choice(self._n_stored,
                                       size=B,
                                       replace=False,
                                       p=probs)
        
        # importance sampling weights
        # => normalized w_i = (1/(N*P(i)))^beta
        imp_weights = (self._n_stored * probs[idx_sampled]) ** (-beta)
        imp_weights /= imp_weights.max()
        imp_weights = imp_weights.astype(np.float32).reshape(-1,1)

        # return tensors st no additional conversion is necessary for forward pass
        def to_torch(xs: np.ndarray) -> torch.Tensor:
            ts = torch.from_numpy(xs.copy())
            if pin_memory and self.device.type == "cuda":
                ts = ts.pin_memory()
                return ts.to(self.device, non_blocking=True)
            return ts.to(self.device)
        
        transitions = (
            to_torch(self._states[idx_sampled]),
            to_torch(self._actions[idx_sampled]),
            to_torch(self._rewards[idx_sampled]),
            to_torch(self._next_states[idx_sampled]),
            to_torch(self._ended[idx_sampled])
        )
        imp_weights = to_torch(imp_weights) # (B,1)
        return transitions, idx_sampled, imp_weights
    
    def create_checkpoint(self) -> dict:
        checkpoint = super().create_checkpoint()
        checkpoint.update({
            "_priorities": self._priorities.copy(),
            "_max_priority": self._max_priority,
            "_beta_step_counter": self._beta_step_counter # log number of sample steps up to now
        })
        return checkpoint
    
    def load_checkpoint(self, checkpoint: dict) -> None:
        super().load_checkpoint(checkpoint=checkpoint)
        if "_priorities" in checkpoint:
            self._priorities[:] = checkpoint["_priorities"]
            self._max_priority = float(checkpoint["_max_priority"])
            self._beta_step_counter = int(checkpoint["_beta_step_counter"])
