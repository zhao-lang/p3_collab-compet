import numpy as np
import random

from models import Actor, Critic
from utils import *

import torch
import torch.optim as optim
import torch.nn.functional as F


GAMMA = 0.99            # discount factor
LR = 1e-4               # learning rate 
EPSILON = 5.0


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_size,
        action_size,
        seed,
        policy_lr=LR,
        critic_lr=LR):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        
        random.seed(seed)

        # Networks
        self.policy_local = Actor(self.state_size, self.action_size, seed)
        self.policy_target = Actor(self.state_size, self.action_size, seed)
        self.critic_local = Critic(self.state_size + self.action_size, self.action_size, seed)
        self.critic_target = Critic(self.state_size + self.action_size, self.action_size, seed)

        # initialize target networks weights
        for target_param, param in zip(self.policy_target.parameters(), self.policy_local.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(param.data)

        # optimizer
        self.policy_optimizer = optim.Adam(self.policy_local.parameters(), lr=policy_lr)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=critic_lr)

        self.epsilon = EPSILON

        # Noise process
        self.noise = OUNoise(action_size, seed)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(device)
        self.policy_local.eval()
        with torch.no_grad():
            action_values = self.policy_local(state).cpu().data.numpy()
        self.policy_local.train()
        if add_noise:
            action_values += self.epsilon * self.noise.sample()

        return np.clip(action_values, -1, 1)
    
    def reset(self):
        self.noise.reset()

                       

    


