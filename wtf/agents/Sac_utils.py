'''
Implementation based on https://github.com/pranz24/pytorch-soft-actor-critic
Addded normalizing flow policy using the normflows package https://arxiv.org/pdf/2302.12014
created without AI usage
'''

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import normflows as nf


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()
        # Allow gym spaces
        if not isinstance(num_inputs, int):
            num_inputs = num_inputs.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        if not isinstance(num_inputs, int):
            num_inputs = num_inputs.shape[0]
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        if not isinstance(num_inputs, int):
            num_inputs = num_inputs.shape[0]
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.register_buffer("action_scale", torch.tensor(1.0))
            self.register_buffer("action_bias", torch.tensor(0.0))
        else:
            self.register_buffer(
                "action_scale",
                torch.FloatTensor((action_space.high - action_space.low) / 2.)
            )
            self.register_buffer(
                "action_bias",
                torch.FloatTensor((action_space.high + action_space.low) / 2.)
    )

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, compute_mean=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()  # softplus might better
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        #print(action.shape, log_prob.shape, mean.shape)
        return action, log_prob, mean
    '''
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)'''
    

class FlowPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super().__init__()
        q0 = nf.distributions.DiagGaussian(num_actions, trainable=False)
        latent_size = num_actions
        context_size = num_inputs

        flows = []
        for _ in range(1):
            l1 = nf.flows.MaskedAffineAutoregressive(
                latent_size,
                hidden_dim,
                context_size,
            )
            flows.extend([
                l1,
                nf.flows.LULinearPermute(latent_size)
            ])
        self.flow = nf.ConditionalNormalizingFlow(q0, flows)
        self.register_buffer("action_scale", torch.tensor(1.0))
        self.register_buffer("action_bias", torch.tensor(0.0))


    def sample(self, state, compute_mean=False):
        actions, log_probs = self.flow.sample(num_samples=state.shape[0], context=state)
        log_probs = log_probs.unsqueeze(1)  # B -> B x 1
        mean_action = actions

        actions = torch.tanh(actions) * self.action_scale + self.action_bias  # B x A
        log_dets = torch.log(self.action_scale * (1 - actions.pow(2)) + epsilon) # BxA, log(|det d tanh(actions) / d actions|)
        log_probs = log_probs - log_dets.sum(dim=1, keepdim=True)
        return actions, log_probs, mean_action
    


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.register_buffer("action_scale", torch.tensor(1.0))
            self.register_buffer("action_bias", torch.tensor(0.0))
        else:
            self.register_buffer(
                "action_scale",
                torch.FloatTensor((action_space.high - action_space.low) / 2.)
            )
            self.register_buffer(
                "action_bias",
                torch.FloatTensor((action_space.high + action_space.low) / 2.)
            )

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state, compute_mean=False):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
    

'''
def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs'''

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
