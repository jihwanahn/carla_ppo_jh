import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import MultivariateNormal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, action_std_init, run_name):
        super(ActorCritic, self).__init__()
        self.obs_dim = obs_dim
        # print("Observation dimension:", obs_dim)
        self.action_dim = action_dim
        # print("Action dimension:", action_dim)
        self.device = torch.device("cpu")
        self.cov_var = torch.full((self.action_dim,), action_std_init, device=self.device)
        self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0).to(self.device)

        # if run_name == "VIT":
        #     self.obs_dim = 195
        # else:
        #     self.obs_dim = obs_dim

        self.actor = nn.Sequential(
                nn.Linear(self.obs_dim, 512),
                nn.Tanh(),
                nn.Linear(512, 300),
                nn.Tanh(),
                nn.Linear(300, 200),
                nn.Tanh(),
                nn.Linear(200, self.action_dim),
                nn.Tanh()
                )

        self.critic = nn.Sequential(
                nn.Linear(self.obs_dim, 512),
                nn.Tanh(),
                nn.Linear(512, 300),
                nn.Tanh(),
                nn.Linear(300, 200),
                nn.Tanh(),
                nn.Linear(200, 1)
                )
        # print("ActorCritic initialized with obs_dim:", obs_dim, "action_dim:", action_dim)


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
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
            # print("Converted observation shape:", obs.shape)  # 입력 ndarray를 텐서로 변환한 후의 형태 출력
        # print("Observation shape on device:", obs.shape)  # 입력 텐서의 형태 출력
        mean = self.actor(obs)
        # print(__file__, "Mean action shape from actor:", mean.shape)  # actor로부터 얻어진 평균 행동의 형태 출력
        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # print("Sampled action shape:", action.shape)  # 샘플링된 행동의 형태 출력
        # print("Log prob shape:", log_prob.shape)  # 로그 확률의 형태 출력
        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob  
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach(), log_prob.detach()
    
    def evaluate(self, obs, action):
        # print("Input obs shape in evaluate:", obs.shape)
        # print("Input action shape in evaluate:", action.shape)
        obs = obs.to(self.device)
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        
        logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        values = self.critic(obs)
        
        return logprobs, values, dist_entropy
