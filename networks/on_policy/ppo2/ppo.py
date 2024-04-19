import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, channels, height, width, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device)

        # Dynamically calculate the output size to ensure it matches the expected input size of linear layers
        self.linear_input_size = 25600 #self._get_conv_output(channels, height, width)
        print(f"Calculated linear input size: {self.linear_input_size}")

        # Adjust these layers if necessary based on the output size from convolutional layers
        self.actor = nn.Sequential(
            nn.Linear(self.linear_input_size, 500),  # Ensure this matches the output of conv_layers
            nn.Tanh(),
            nn.Linear(500, 300),
            nn.Tanh(),
            nn.Linear(300, 200),
            nn.Tanh(),
            nn.Linear(200, action_dim)
        ).to(self.device)
        
        self.critic = nn.Sequential(
            nn.Linear(self.linear_input_size, 500),  # Ensure this matches the output of conv_layers
            nn.Tanh(),
            nn.Linear(500, 300),
            nn.Tanh(),
            nn.Linear(300, 200),
            nn.Tanh(),
            nn.Linear(200, 1)
        ).to(self.device)

        self.action_std = torch.full((action_dim,), action_std_init, device=self.device)
        self.cov_mat = torch.diag(self.action_std).unsqueeze(dim=0).to(self.device)
    
    def forward(self, obs):
        raise NotImplementedError

    def _get_conv_output(self, channels, height, width):
        # Use a dummy input to pass through the conv layers to determine the output size
        with torch.no_grad():
            input = torch.zeros(1, channels, height, width).to(self.device)
            output = self.conv_layers(input)
            return output.numel()

    def get_action_and_log_prob(self, obs):
        obs = obs.to(self.device)
        obs = self.conv_layers(obs)  # Process through CNN layers

        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach(), log_prob.detach()

    def evaluate(self, obs, action):
        obs = obs.to(self.device)
        obs = self.conv_layers(obs)  # Process through CNN layers

        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        
        logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        values = self.critic(obs)
        
        return logprobs, values, dist_entropy

    def set_action_std(self, new_action_std):
        self.cov_var = torch.full((self.action_dim,), new_action_std, device=self.device)
        self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0).to(self.device)

    def get_value(self, obs):
        obs = obs.to(self.device)
        obs = self.conv_layers(obs)  # Process through CNN layers

        return self.critic(obs)



def test():
    # Assume 3 channels (RGB), image size 160x120
    model = ActorCritic(channels=3, height=120, width=160, action_dim=4)
    model.to(model.device)
    # Create a dummy input tensor mimicking a single RGB image
    dummy_input = torch.rand(1, 3, 120, 160).to(model.device)
    action_probs, value = model(dummy_input)
    print("Action probabilities:", action_probs)
    print("Value:", value)

if __name__ == "__main__":
    test()
