import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import MultivariateNormal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device("cpu")
        
        self.cov_var = torch.full((self.action_dim,), action_std_init, device=self.device)
        self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0).to(self.device)

        self.actor = nn.Sequential(
                nn.Linear(self.obs_dim, 500),
                nn.Tanh(),
                nn.Linear(500, 300),
                nn.Tanh(),
                nn.Linear(300, 200),
                nn.Tanh(),
                nn.Linear(200, self.action_dim),
                nn.Tanh()
                )
        
        self.critic = nn.Sequential(
                nn.Linear(self.obs_dim, 500),
                nn.Tanh(),
                nn.Linear(500, 300),
                nn.Tanh(),
                nn.Linear(300, 200),
                nn.Tanh(),
                nn.Linear(200, 1)
                )

    def forward(self, obs):
        raise NotImplementedError
    
    def set_action_std(self, new_action_std):
        self.cov_var = torch.full((self.action_dim,), new_action_std, device=self.device)
        self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0).to(self.device)

    def get_value(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        return self.critic(obs)
    
    def get_action_and_log_prob(self, obs):
        if isinstance(obs, list):
            obs = torch.stack(obs, dim=0)
        elif isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        # obs = obs.view(-1, self.obs_dim)

        obs = obs.to(self.device)

        if obs.nelement() == self.obs_dim:
            obs = obs.view(-1, self.obs_dim)
        else:
            raise ValueError(f"Expected {self.obs_dim} elements but got {obs.nelement()}")

        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.detach(), log_prob.detach()
    
    def evaluate(self, obs, action):
        obs = obs.to(self.device)
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        
        logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        values = self.critic(obs)
        
        return logprobs, values, dist_entropy
