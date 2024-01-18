import torch
import torch.nn as nn
import torch.optim as optim
from parameters import SAC_LEARNING_RATE, SAC_CHECKPOINT_DIR, MEMORY_SIZE, BATCH_SIZE, GAMMA
from torch.distributions import Normal
import numpy as np
import os
from networks.off_policy.replay_buffer import ReplayBuffer

class Actor(nn.Module):
    def __init__(self, n_actions, max_action):
        super(Actor, self).__init__()
        self.checkpoint_file = os.path.join(SAC_CHECKPOINT_DIR, "actor")
        self.max_action = max_action
        self.n_actions = n_actions

        self.Linear1 = nn.Sequential(
            nn.Linear(95 + 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.mu = nn.Linear(64, self.n_actions)
        self.log_std = nn.Linear(64, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=SAC_LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        fc = self.Linear1(x)
        mu = self.mu(fc)
        log_std = self.log_std(fc)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        return mu, std

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class Critic(nn.Module):
    def __init__(self, n_actions):
        super(Critic, self).__init__()
        self.checkpoint_file = os.path.join(SAC_CHECKPOINT_DIR, "critic")
        self.n_actions = n_actions

        self.Linear1 = nn.Sequential(
            nn.Linear(95 + 5 + self.n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.Linear2 = nn.Sequential(
            nn.Linear(95 + 5 + self.n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.V = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=SAC_LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        fc1 = self.Linear1(x)
        fc2 = self.Linear2(x)
        V = self.V(fc2)
        return V

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class SACAgent(object):
    def __init__(self, n_actions, max_action):
        self.gamma = GAMMA
        self.alpha = SAC_LEARNING_RATE
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = BATCH_SIZE
        self.train_step = 0
        self.actor = Actor(n_actions, max_action)
        self.critic1 = Critic(n_actions)
        self.critic2 = Critic(n_actions)
        self.value = Critic(n_actions)
        self.value_target = Critic(n_actions)
        self.value_target.load_state_dict(self.value.state_dict())
        self.value_target.eval()
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=SAC_LEARNING_RATE)
        self.value_loss = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(MEMORY_SIZE,100, n_actions)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_transition(self, observation, action,  reward, new_observation, done):
        self.replay_buffer.save_transition(observation, action, reward, new_observation, done)

    def get_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
        mu, std = self.actor.forward(observation)
        dist = Normal(mu, std)
        action = dist.sample()
        action = torch.clamp(action, min=-1, max=1)
        return action.item()

    def decrese_epsilon(self):
        pass

    def save_model(self):
        self.actor.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()
        self.value.save_checkpoint()

    def load_model(self):
        self.actor.load_checkpoint()
        self.critic1.load_checkpoint()
        self.critic2.load_checkpoint()
        self.value.load_checkpoint()

    def learn(self):
        if self.replay_buffer.counter < self.batch_size:
            return

        observation, action, reward, new_observation, done = self.replay_buffer.sample_buffer()

        observation = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        new_observation = torch.tensor(new_observation, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done, dtype=torch.float).to(self.actor.device)
