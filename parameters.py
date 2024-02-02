"""

    All the much needed hyper-parameters needed for the algorithm implementation. 

"""
import os

MODEL_LOAD = False
SEED = 0
BATCH_SIZE = 1
IM_WIDTH = 160
IM_HEIGHT = 80
GAMMA = 0.99
MEMORY_SIZE = 5000
EPISODES = 1000

#VAE Bottleneck
LATENT_DIM = 95

#Dueling DQN (hyper)parameters
DQN_LEARNING_RATE = 0.0001
EPSILON = 1.00
EPSILON_END = 0.05
EPSILON_DECREMENT = 0.00001

REPLACE_NETWORK = 5
DQN_CHECKPOINT_DIR = 'preTrained_models/ddqn'
os.makedirs(DQN_CHECKPOINT_DIR, exist_ok=True)
MODEL_ONLINE = 'carla_dueling_dqn_online.pth'
MODEL_TARGET = 'carla_dueling_dqn_target.pth'


#Proximal Policy Optimization (hyper)parameters
EPISODE_LENGTH = 7500
TOTAL_TIMESTEPS = 3e6
ACTION_STD_INIT = 0.2
TEST_TIMESTEPS = 5e4
PPO_LEARNING_RATE = 1e-4  
PPO_CHECKPOINT_DIR = 'preTrained_models/ppo/'
os.makedirs(PPO_CHECKPOINT_DIR, exist_ok=True)
POLICY_CLIP = 0.2

#Soft Actor Critic (hyper)parameters
# SAC_LEARNING_RATE = 0.0001
# SAC_CHECKPOINT_DIR = 'preTrained_models/sac/'
# os.makedirs(SAC_CHECKPOINT_DIR, exist_ok=True)
# MEMORY_SIZE = int(1e6)
# BATCH_SIZE = 256
# POLICY_UPDATE = 1
# POLICY_DELAY = 2
# POLICY_CLIP = 0.5
# POLICY_NOISE = 0.2
# POLICY_NOISE_CLIP = 0.5
# GAMMA = 0.99
# TAU = 0.005
# ALPHA = 0.2
# AUTO_ENTROPY = True
