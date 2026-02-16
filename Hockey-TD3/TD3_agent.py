import torch
import torch.nn as nn
import numpy as np
import copy
from typing import Optional
from dataclasses import dataclass
# actor network
# critic network
# td3 agent

class Actor(nn.Module):
    """
    Deterministic policy network pi(s) that maps
    observations to actions (scaled to env's action bounds)
    """
    def __init__(self, 
                 obs_dim: int, 
                 act_dim: int,
                 action_space, 
                 hidden_sizes = (256, 256),
                 ) -> None:
        super().__init__()
        assert hasattr(action_space, "low") and hasattr(action_space, "high"), \
            "Actor expects a gymnasium.spaces.Box as action_space"

        # allow for flexible #hidden layers
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, act_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

        # store action force bounds into (x,y)-direction (make them move with .to(device))
        low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
        self.register_buffer("act_low", torch.from_numpy(low))
        self.register_buffer("act_high", torch.from_numpy(high))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        expects obs: (N_batch_obs, obs_dim)
        returns: (N_batch_obs, act_dim) scaled to [act_low, act_high]

        uses relu and tanh (last layer) as activ. fcts.
        """
        a = self.net(obs) # a = tanh(x) => in [-1,1]

        # scale: out = alpha * a + beta
        # low = -1*alpha + beta; high = 1*alpha +beta 
        alpha = (self.act_high - self.act_low) / 2
        beta = (self.act_high + self.act_low) / 2
        return alpha * a + beta # [action_fx, action_fy, action_torque, action_shoot]

        

class Critic(nn.Module):
    """
    Action-value approximation Q(s,a) network 
    TD3 uses 2 independent critics of the same architecture
    """
    def __init__(self, 
                 obs_dim: int, 
                 act_dim: int, 
                 hidden_sizes = (256, 256)) -> None:
        super().__init__()
        layers = []
        in_dim = obs_dim + act_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        # output unbounded q values in last layer (no act. fct)
        layers += [nn.Linear(in_dim, 1)] 
        self.net = nn.Sequential(*layers)

    def forward(self, 
                obs: torch.Tensor, 
                act: torch.Tensor) -> torch.Tensor:
        obs_act = torch.cat([obs, act], dim=-1)
        q = self.net(obs_act)
        return q

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
        
        self._rng = np.random.default_rng(seed)

        # buffer
        self._states  = np.zeros((self.max_capacity, obs_dim), dtype=dtype)
        self._actions = np.zeros((self.max_capacity, act_dim), dtype=dtype)
        self._rewards  = np.zeros((self.max_capacity, 1), dtype=dtype)
        self._next_states = np.zeros((self.max_capacity, obs_dim), dtype=dtype)
        self._ended  = np.zeros((self.max_capacity, 1), dtype=dtype)

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


#def batch_to_torch(batch, device):


 
class TD3_Agent:
    def __init__(self, 
                 obs_dim: int,
                 act_dim: int, 
                 observation_space,
                 action_space,
                 device = None,
                 **params) -> None:
        self.device = torch.device(device if device is not None 
                            else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        
        self._obs_dim = obs_dim
        self._act_dim = act_dim

        assert hasattr(action_space, "low") and hasattr(action_space, "high"), \
            "Actor expects a gymnasium.spaces.Box as action_space"
        # TODO: improve!!
        low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
        self._act_low  = torch.tensor(low, device=self.device)
        self._act_high = torch.tensor(high, device=self.device)
        # TODO: needed?
        self._action_space = action_space
        self._observation_space = observation_space
        
        # TODO
        # bounds for clipping actions
        # for numpy
        # for torch

        # parameters for learning
        self._params = {
            "actor_hiddens": [256, 256],
            "critic_hiddens": [256, 256],

            "warmup_steps": 1e4,
            "batch_size": 128,
            "gamma": 0.95,
            "tau": 0.005,

            "actor_lr": 1e-3,
            "critic_lr": 1e-3,
            "policy_delay": 3,
            
            "noise_type": "Gaussian",
            "exploration_noise": 0.1,
            "opponent_noise": 0.05,

            "noise_target_policy": 0.2,
            "clip_noise": 0.5 # TODO: check if this is sensible
        }
        # overwrite 
        self._params.update(params)
        
        # initialize the actor, critic1 + 2 and target networks
        self._init_networks_and_optimizers()

        # Replay buffer
        self.buffer = ReplayBuffer(obs_dim = self._obs_dim,
                                   act_dim = self._act_dim,
                                   device = self.device)

        self._update_iters = 0

    # replay buffer

    def _init_networks_and_optimizers(self):
        """
        Initializes the Actor, Critic1, Critic2 and the respective
        target networks, and sets up the optimizers
        """
        # actor network
        self.actor = Actor(
            obs_dim=self._obs_dim,
            act_dim=self._act_dim,
            action_space=self._action_space,
            hidden_sizes=self._params["actor_hiddens"]
        ).to(self.device)
        # actor target network
        self.actor_target = copy.deepcopy(self.actor).to(self.device)

        # critic networks
        self.critic1 = Critic(obs_dim=self._obs_dim, 
                              act_dim=self._act_dim,
                              hidden_sizes=self._params["critic_hiddens"]
                              ).to(self.device)
        self.critic2 = Critic(obs_dim=self._obs_dim, 
                              act_dim=self._act_dim,
                              hidden_sizes=self._params["critic_hiddens"]
                              ).to(self.device)
        # deep copy weights to target networks
        self.critic1_target = copy.deepcopy(self.critic1).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2).to(self.device)

        # target networks are only used for inference
        for net in (self.actor_target, self.critic1_target, self.critic2_target):
            net.eval()
            net.requires_grad_(False) # save memory/time

        if self.device.type == "cuda":
            # fuse computation and optimize memory usage
            self.actor = torch.compile(self.actor)
            self.critic1 = torch.compile(self.critic1)
            self.critic2 = torch.compile(self.critic2)

        # optimizers
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(),
            lr = self._params["actor_lr"])
        self.critic1_opt = torch.optim.Adam(
            self.critic1.parameters(),
            lr = self._params["critic_lr"]
        )
        self.critic2_opt = torch.optim.Adam(
            self.critic2.parameters(),
            lr = self._params["critic_lr"]
        )

    @torch.no_grad()        
    def select_action(self, 
                      observation: np.ndarray,
                      *,
                      explore: bool = True) -> np.ndarray:
        """
        observation: (obs_dim, ) coming from the env (-> numpy)
        explore: True in training mode, False for deterministic action selection
        returns: action of shape (act_dim,) clipped to the env bounds
        """
        # convert to torch tensor
        obs_t = torch.as_tensor(observation, 
                                dtype=torch.float32, 
                                device=self.device).unsqueeze(0)
        # forward 
        action_t = self.actor(obs_t).squeeze(0) # (act_dim,)

        if explore:
            action_t = action_t + self._exploration_noise()
        
        action_t = torch.clamp(action_t, self._act_low, self._act_high)
        # convert back to numpy bc this is what the env expects
        return action_t.cpu().numpy()

    def _exploration_noise(self) -> torch.Tensor:
        # returns (act_dim, ) noise 
        if self._params["noise_type"] == "Gaussian":
            std = float(self._params["exploration_noise"])
            # uncorrelated noise
            noise = std * torch.randn(self._act_dim, device=self.device)
            return noise
            

    def _batch_to_torch(self, batch):
        # helper to convert to torch for the training step
        return (torch.as_tensor(batch.states, device=self.device),
                torch.as_tensor(batch.actions, device=self.device),
                torch.as_tensor(batch.rewards, device=self.device),
                torch.as_tensor(batch.next_states, device=self.device),
                torch.as_tensor(batch.ended, device=self.device)
                )
    
    # soft update helper 
    def _soft_update(self, 
                     network: nn.Module, 
                     target: nn.Module,
                     tau: float) -> None:
        """
        Used to update the target networks slowly 
        (=> smaller TD error)
        """
        with torch.no_grad():
            for params_net, params_target in zip(network.parameters(), target.parameters()):
                # p_target = (1-tau) * p_target + tau * p_net 
                params_target.data.mul_(1.0 - tau).add_(tau * params_net)


    # training logic
    # update critic, update actor delayed
    def train_step(self) -> dict:
        """
        Performs one gradient step for 
        - critic1 and 
        - critic2 
        The actor and all target networks are updated every policy_delay.
        Returns: logging dict (losses, ++++++)
        """
        self._update_iters += 1

        # check if we have a minimal number samples in buffer
        # ideally, we should have at least "warmup_steps" many
        if self.buffer._n_stored < self._params["batch_size"]:
            return {"skipped": 1}
        
        # parameters
        B = int(self._params["batch_size"])
        gamma = float(self._params["gamma"])
        tau = float(self._params["tau"])
        policy_delay = int(self._params["policy_delay"])

        # sample B many transitions from the batch
        # TODO: for debugging, use the normal sample method (returns numpy)
        s, a, r, s_next, ended = self.buffer.sample_torch(B, pin_memory=False)
        # s: (B, obs_dim)
        # a: (B, act_dim)
        # r: (B, 1)
        # ended: (B, 1)

        # Targets make step and compute return
        with torch.no_grad():
            # policy smoothing: add noise to the target policy
            policy_noise = self._params["noise_target_policy"] * torch.randn_like(a)
            # clip the noise
            policy_noise = torch.clamp(policy_noise, 
                                -self._params["clip_noise"], 
                                self._params["clip_noise"])
            a_next_t = self.actor_target(s_next) + policy_noise
            # make sure a_next is within action bounds
            a_next_t = torch.clamp(a_next_t, self._act_low, self.act_high)
            
            # target
            q_c1_target = self.critic1_target(s_next, a_next_t)
            q_c2_target = self.critic2_target(s_next, a_next_t)
            # q_discounted = 0 if the episode has ended
            q_discounted = gamma * torch.min(q_c1_target, q_c2_target) * (1.0 - ended)
            # target return
            y = r + q_discounted


        # *** Update CRITICS ***
            
        # compute loss of the "normal" critics
        q_c1 = self.critic1(s, a)
        q_c2 = self.critic2(s, a)
        loss_c1 = (y - q_c1).pow(2).mean()
        loss_c2 = (y - q_c2).pow(2).mean()

        # update the critics:
        self.critic1_opt.zero_grad(set_to_none=True) # remove gradient tensor
        loss_c1.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad(set_to_none=True)
        loss_c2.backward()
        self.critic2_opt.step()

        info = {
            "skipped": 0,
            "critic1_loss": float(loss_c1.detach().cpu()),
            "critic2_loss": float(loss_c2.detach().cpu()),
            "q_c1_mean": float(q_c1.detach().mean().cpu()),
            "q_c2_mean": float(q_c2.detach().mean().cpu())
        }


        # *** TARGET and ACTOR Updates ***
        if self._update_iters % policy_delay == 0:
            
            # *** Update ACTOR ***
            a_next_a = self.actor(s) # (B, act_dim)
            # policy aims to max E(Q_c1(s,a_next_a))
            loss_actor = - self.critic1(s, a_next_a).mean()
            
            # update
            self.actor_opt.zero_grad(set_to_none=True) # remove all grads
            loss_actor.backward()
            self.actor_opt.step()
            info["actor_loss"] = float(loss_actor.detach().cpu())


            # ** Update TARGET networks ***
            self._soft_update(network=self.actor,
                              target = self.actor_target,
                              tau=tau)
            self._soft_update(self.critic1,
                              self.critic1_target,
                              tau)
            self._soft_update(self.critic2, 
                              self.critic2_target,
                              tau)
            
        return info
