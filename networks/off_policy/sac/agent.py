import torch
import numpy as np
from sac.sac import SAC

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.sac = SAC(state_dim, action_dim, hidden_dim)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _ = self.sac.sample_action(state)
        return action.detach().cpu().numpy().flatten()

    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size)
        self.sac.update_parameters(state_batch, action_batch, reward_batch, next_state_batch, mask_batch, updates)

    # 추가적인 메서드 구현 (예: 모델 저장 및 로드)
