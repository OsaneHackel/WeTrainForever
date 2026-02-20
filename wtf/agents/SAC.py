import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from gymnasium import spaces

from wtf.line_search.sls import Sls
from wtf.agents.Sac_utils import soft_update, hard_update, GaussianPolicy, QNetwork, DeterministicPolicy
from wtf.agents.utils import UnsupportedSpace, Memory

class SAC(object):
    def __init__(self, observation_space, action_space, **args):
        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.box.Box):
           raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Require Box)'.format(action_space, self))
        
        default_args = {
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "batch_size": 128,
            "buffer_size": int(2e6),
            "policy": "Gaussian",
            "target_update_interval": 1,
            "automatic_entropy_tuning": True,
            "hidden_size": 256,
            "lr": 3e-4,
            "cuda": torch.cuda.is_available(),
            "critic_optimizer":"ADAM",
            "policy_optimizer":"ADAM",
        }

        default_args.update(args)
        args = default_args
        self.gamma = args["gamma"]
        self.tau = args["tau"]
        self.alpha = args["alpha"]
        self.batch_size = args["batch_size"]

        self.policy_type = args["policy"]
        self.target_update_interval = args["target_update_interval"]
        self.automatic_entropy_tuning = args["automatic_entropy_tuning"]
        self.buffer =Memory(max_size=args['buffer_size'])

        self.device = torch.device("cuda" if args["cuda"] else "cpu")
        
        self.critic_optim_name =args["critic_optimizer"]
        self.policy_optim_name =args["policy_optimizer"]

        self.observation_space = observation_space
        self.action_space = action_space
        self.critic = QNetwork(observation_space, action_space.shape[0], args["hidden_size"]).to(device=self.device)
        if self.critic_optim_name =="ADAM":
            self.critic_optim = Adam(self.critic.parameters(), lr=args["lr"])
        elif self.critic_optim_name == "SLS":
            self.critic_optim = Sls(self.critic.parameters())

        self.critic_target = QNetwork(observation_space, action_space.shape[0], args["hidden_size"]).to(self.device)
        #to keep the checkpoint interface the same
        self.Q = self.critic
        self.Q_target = self.critic_target
        self.policy_target = self.critic_target  # SAC doesn't have separate policy target
        ##
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args["lr"])

            self.policy = GaussianPolicy(observation_space, action_space.shape[0], args["hidden_size"], action_space).to(self.device)
            if self.policy_optim_name =="ADAM":
                self.policy_optim = Adam(self.policy.parameters(), lr=args["lr"])
                self.policy_optim_name = "ADAM"
            elif self.policy_optim_name == "SLS":
                self.policy_optim = Sls(self.policy.parameters())
                self.policy_optim_name = "SLS"

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(observation_space, action_space.shape[0], args["hidden_size"], action_space).to(self.device)
            if self.policy_optim_name =="ADAM":
                self.policy_optim = Adam(self.policy.parameters(), lr=args["lr"])
                self.policy_optim_name = "ADAM"
            elif self.policy_optim_name == "SLS":
                self.policy_optim = Sls(self.policy.parameters())
                self.policy_optim_name = "SLS"

            #self.policy_optim = Adam(self.policy.parameters(), lr=args["lr"])

    def store_transition(self, transition):
        state, action, reward, next_state, done = transition
        mask = 1.0 - float(done)
        self.buffer.add_transition((state, action, reward, next_state, mask))

    def act(self, state, eps=None):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        stochastic = (eps is None) or (eps > 0)
        #eps tells us whether we want to explore
        with torch.no_grad():
            if stochastic:  # exploration
                action, _, _ = self.policy.sample(state)
            else:   # evaluation
                _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    #just to keep the interface the same
    def reset(self):
        pass

    def clone(self):
        agent = SAC(
            self.observation_space,
            self.action_space,
            gamma=self.gamma,
            tau=self.tau,
            alpha=self.alpha
        )

        agent.policy.load_state_dict(self.policy.state_dict())
        agent.critic.load_state_dict(self.critic.state_dict())
        agent.policy.requires_grad_(False)

        return agent
    '''
    def train(self, iter_fit=1):
        losses = []
        lrs=[]

        def critic_closure(): 
            self.critic_optim.zero_grad()
            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss

            qf_loss.backward()
            return qf_loss

        for i in range(iter_fit):
            data = self.buffer.sample(batch=self.batch_size)

            state_batch = (np.stack(data[:,0])) # s_t
            action_batch = (np.stack(data[:,1])) # a_t
            reward_batch = (np.stack(data[:,2])[:,None]) # rew  (batchsize,1)
            next_state_batch = (np.stack(data[:,3])) # s_t+1
            mask_batch = (np.stack(data[:,4])[:,None]) # done signal  (batchsize,1)

            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device)

            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss
            next_q_value = next_q_value.detach()
            state_batch = state_batch.detach()
            action_batch = action_batch.detach()
            self.critic_optim.zero_grad()
            
            if self.critic_optim_name == "ADAM":
                qf_loss.backward()
                self.critic_optim.step()
            elif self.critic_optim_name == "SLS":
                self.critic_optim.step(closure=critic_closure)
            
            pi, log_pi, _ = self.policy.sample(state_batch)

            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            #start changes
            for p in self.critic.parameters():
                p.requires_grad = False

            def policy_closure():
                self.policy_optim.zero_grad()

                pi, log_pi, _ = self.policy.sample(state_batch)
                q1, q2 = self.critic(state_batch, pi)
                loss = (self.alpha * log_pi - torch.min(q1, q2)).mean()

                loss.backward()
                return loss
            if self.policy_optim_name == "ADAM":
                self.policy_optim.zero_grad()
                policy_loss.backward()
                self.policy_optim.step()
            elif self.policy_optim_name == "SLS":
                policy_loss = self.policy_optim.step(closure=policy_closure)

            # Unfreeze critic
            for p in self.critic.parameters():
                p.requires_grad = True

            #end changes
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()
                alpha_tlogs = self.alpha.clone() # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.).to(self.device)
                alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX log
        
            losses.append((
                qf1_loss.item(),
                qf2_loss.item(),
                policy_loss.item()
            ))

            lrs.append(self.policy_optim.param_groups[0]['lr'])
        #update target network every epoch
        soft_update(self.critic_target, self.critic, self.tau)
        return losses, lrs'''

    def train(self, iter_fit=1):
        c_loss = []
        p_loss =[]
        a_loss = []
        policy_lrs = []
        critic_lrs = []

        for _ in range(iter_fit):
            data = self.buffer.sample(batch=self.batch_size)

            # === Prepare batch ===
            state_batch = torch.FloatTensor(np.stack(data[:, 0])).to(self.device)
            action_batch = torch.FloatTensor(np.stack(data[:, 1])).to(self.device)
            reward_batch = torch.FloatTensor(np.stack(data[:, 2])[:, None]).to(self.device)
            next_state_batch = torch.FloatTensor(np.stack(data[:, 3])).to(self.device)
            mask_batch = torch.FloatTensor(np.stack(data[:, 4])[:, None]).to(self.device)

            # Compute targets 
            with torch.no_grad():
                next_action, next_log_pi, _ = self.policy.sample(next_state_batch)
                q1_t, q2_t = self.critic_target(next_state_batch, next_action)
                min_q_t = torch.min(q1_t, q2_t) - self.alpha * next_log_pi
                next_q_value = reward_batch + mask_batch * self.gamma * min_q_t

            def critic_closure():
                q1, q2 = self.critic(state_batch, action_batch)
                loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)
                return loss

            self.critic_optim.zero_grad()
            if self.critic_optim_name == "ADAM":
                critic_loss = critic_closure()
                critic_loss.backward()
                self.critic_optim.step()
            else:  # SLS
                critic_loss, _ = self.critic_optim.step(closure=critic_closure)


            def policy_closure():
                pi, log_pi, _ = self.policy.sample(state_batch)
                q1, q2 = self.critic(state_batch, pi)
                loss = (self.alpha * log_pi - torch.min(q1, q2)).mean()
                return loss

            self.policy_optim.zero_grad()
            if self.policy_optim_name == "ADAM":
                policy_loss = policy_closure()
                policy_loss.backward()
                self.policy_optim.step()
            else:  # SLS
                policy_loss,_ = self.policy_optim.step(closure=policy_closure)


            if self.automatic_entropy_tuning:
                with torch.no_grad():
                    _, log_pi, _ = self.policy.sample(state_batch)

                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy)).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp().detach()
            else:
                alpha_loss = torch.tensor(0.0, device=self.device)
            c_loss.append(critic_loss.item())
            p_loss.append(policy_loss.item())
            a_loss.append(alpha_loss.item())

            # Some SLS impls don't expose param_groups
            if self.policy_optim_name == "SLS":
                policy_lrs.append(self.policy_optim.state['step_size'])
                critic_lrs.append(self.critic_optim.state['step_size'])        
            else:
                policy_lrs.append(self.policy_optim.param_groups[0]["lr"])
                critic_lrs.append(self.critic_optim.param_groups[0]["lr"])

        soft_update(self.critic_target, self.critic, self.tau)
        return c_loss, p_loss, a_loss, critic_lrs, policy_lrs

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()