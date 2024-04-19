import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from networks.on_policy.ppo2.ppo import ActorCritic
from parameters import  *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Buffer:
    def __init__(self):
        # Batch data
        self.observation = []  
        self.actions = []         
        self.log_probs = []     
        self.rewards = []         
        self.dones = []

    def clear(self):
        del self.observation[:]    
        del self.actions[:]        
        del self.log_probs[:]      
        del self.rewards[:]
        del self.dones[:]

class PPOAgent(object):
    def __init__(self, channels, height, width, action_dim, action_std_init=0.4):
        self.channels = channels
        self.height = height
        self.width = width
        self.action_dim = action_dim
        self.clip = POLICY_CLIP
        self.gamma = GAMMA
        self.n_updates_per_iteration = 7
        self.lr = PPO_LEARNING_RATE
        self.action_std = action_std_init
        self.memory = Buffer()

        self.policy = ActorCritic(channels, height, width, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr},
            {'params': self.policy.critic.parameters(), 'lr': self.lr}
        ])

        self.old_policy = ActorCritic(channels, height, width, action_dim, action_std_init).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def get_action(self, obs, train=True):
        if isinstance(obs, list):
            # Assuming the first element of the list is the image and the rest is additional data
            image_data = obs[0]
            if isinstance(image_data, np.ndarray):
                image_data = torch.tensor(image_data, dtype=torch.float, device=device).permute(2, 0, 1)  # Rearrange axis to CxHxW
            else:
                raise TypeError(f"Expected image data to be a NumPy ndarray, got {type(image_data)} instead.")
        else:
            raise TypeError(f"Observation must be a list containing an image as NumPy ndarray, got {type(obs)} instead.")

        # Ensure observation is 4D (batch, channels, height, width)
        if image_data.dim() == 3:
            image_data = image_data.unsqueeze(0)  # Add batch dimension if it's missing
        elif image_data.dim() != 4:
            raise ValueError(f"Expected observation image to be a 3D or 4D tensor, but got a tensor with {image_data.dim()} dimensions.")

        with torch.no_grad():
            action, log_prob = self.old_policy.get_action_and_log_prob(image_data)

        if train:
            self.memory.observation.append(image_data)
            self.memory.actions.append(action)
            self.memory.log_probs.append(log_prob)

        return action.cpu().numpy().flatten()



    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(torch.full((self.action_dim,), new_action_std, device=self.device))
        self.old_policy.set_action_std(torch.full((self.action_dim,), new_action_std, device=self.device))

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.action_std > min_action_std:
            self.action_std -= action_std_decay_rate
            self.action_std = max(self.action_std, min_action_std)
        self.set_action_std(self.action_std)

    def learn(self):
        rewards = [0] * len(self.memory.rewards)
        discounted_reward = 0
        for i in reversed(range(len(self.memory.rewards))):
            if self.memory.dones[i]:
                discounted_reward = 0
            discounted_reward = self.memory.rewards[i] + (self.gamma * discounted_reward)
            rewards[i] = discounted_reward
        
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.stack(self.memory.observation).detach()
        old_actions = torch.stack(self.memory.actions).detach()
        old_logprobs = torch.stack(self.memory.log_probs).detach()

        for _ in range(self.n_updates_per_iteration):
            logprobs, values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            values = values.squeeze()

            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())
        self.memory.clear()

    def save(self):
        # Ensure directory exists
        os.makedirs(PPO_CNN_CHECKPOINT_DIR + self.town, exist_ok=True)
        # Increment checkpoint file number based on existing files
        self.checkpoint_file_no = len(os.listdir(PPO_CNN_CHECKPOINT_DIR + self.town))
        checkpoint_file = os.path.join(PPO_CNN_CHECKPOINT_DIR, self.town, f"ppo_policy_{self.checkpoint_file_no}.pth")
        torch.save(self.old_policy.state_dict(), checkpoint_file)
        print(f"Saved checkpoint to {checkpoint_file}")

    def chkpt_save(self):
        # Ensure directory exists
        os.makedirs(PPO_CNN_CHECKPOINT_DIR + self.town, exist_ok=True)
        # Decrement to overwrite the last checkpoint
        files = os.listdir(PPO_CNN_CHECKPOINT_DIR + self.town)
        if files:
            self.checkpoint_file_no = len(files) - 1
        checkpoint_file = os.path.join(PPO_CNN_CHECKPOINT_DIR, self.town, f"ppo_policy_{self.checkpoint_file_no}.pth")
        torch.save(self.old_policy.state_dict(), checkpoint_file)
        print(f"Updated checkpoint to {checkpoint_file}")

    def load(self):
        try:
            files = os.listdir(PPO_CNN_CHECKPOINT_DIR + self.town)
            if files:
                self.checkpoint_file_no = len(files) - 1
                checkpoint_file = os.path.join(PPO_CNN_CHECKPOINT_DIR, self.town, f"ppo_policy_{self.checkpoint_file_no}.pth")
                self.old_policy.load_state_dict(torch.load(checkpoint_file, map_location=device))
                self.policy.load_state_dict(torch.load(checkpoint_file, map_location=device))
                print(f"Loaded checkpoint from {checkpoint_file}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")