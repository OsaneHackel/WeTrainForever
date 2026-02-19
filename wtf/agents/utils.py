import numpy as np
import torch

class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)

class OUNoise():
    def __init__(self, shape, theta: float = 0.15, dt: float = 1e-2):
        self._shape = shape
        self._theta = theta
        self._dt = dt
        self.noise_prev = np.zeros(self._shape)
        self.reset()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * ( - self.noise_prev) * self._dt
            + np.sqrt(self._dt) * np.random.normal(size=self._shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        self.noise_prev = np.zeros(self._shape)
    
# class to store transitions
class Memory():
    def __init__(self, max_size=100000):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size=int(max_size)

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx,:] = np.asarray(transitions_new, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        self.inds=np.random.choice(range(self.size), size=batch, replace=False)
        return self.transitions[self.inds,:]

    def get_all_transitions(self):
        return self.transitions[0:self.size]
    

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_fun=torch.nn.Tanh(), output_activation=None):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size
        self.output_activation = output_activation
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = torch.nn.ModuleList([activation_fun for l in  self.layers])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(dim) for dim in layer_sizes[1:]])
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_sizes[-1], self.hidden_sizes[-1]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_sizes[-1], self.output_size)
        ) 

    def forward(self, x):
        for layer,activation_fun, norm in zip(self.layers, self.activations, self.norms):
            x = activation_fun(norm(layer(x)))
        if self.output_activation is not None:
            return self.output_activation(self.readout(x))
        else:
            return self.readout(x)

    def predict(self, x, device=None):
        device = 'cpu' if device is None else device
        with torch.no_grad():
            inp = torch.from_numpy(x.astype(np.float32)).to(device)
            action = self.forward(inp)
            return action.cpu().numpy()

class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100,100],
                 learning_rate = 0.0002):
        super().__init__(input_size=observation_dim + action_dim, hidden_sizes=hidden_sizes,
                         output_size=1)
        self.optimizer=torch.optim.Adam(self.parameters(),
                                        lr=learning_rate,
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss()

    def fit(self, observations, actions, targets): # all arguments should be torch tensors
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass

        pred = self.Q_value(observations,actions)
        # Compute Loss
        loss = self.loss(pred, targets)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, observations, actions):
        return self.forward(torch.hstack([observations,actions]))

'''   
def load_agents(which_agent, agent, ckpt_path, opts, evaluate):
    ckpt= torch.load(f"{opts.checkpoint_path}")
    if which_agent == "DDPG":
        agent.policy.load_state_dict(ckpt["policy"])
        agent.Q.load_state_dict(ckpt["Q"])
        agent.policy_target.load_state_dict(ckpt["policy_target"])
        agent.Q_target.load_state_dict(ckpt["Q_target"])
        agent.optimizer.load_state_dict(ckpt["policy_opt"])
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
    return state_dict'''