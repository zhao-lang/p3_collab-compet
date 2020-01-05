import torch  
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(self, obs_size, action_size, seed, hidden1_size=400, hidden2_size=300):
        super(Actor, self).__init__()
        torch.manual_seed(seed)

        self.linear1 = nn.Linear(obs_size, hidden1_size)
        self.linear2 = nn.Linear(hidden1_size, hidden2_size)
        self.linear3 = nn.Linear(hidden2_size, action_size)
    
    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        self.linear3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        out = F.relu(self.linear1(state))
        out = F.relu(self.linear2(out))
        return F.tanh(self.linear3(out))


class Critic(nn.Module):
    def __init__(self, obs_size, action_size, seed, hidden1_size=400, hidden2_size=300):
        super(Critic, self).__init__()
        torch.manual_seed(seed)

        self.linear1 = nn.Linear(obs_size, hidden1_size)
        self.linear2 = nn.Linear(hidden1_size, hidden2_size)
        self.linear3 = nn.Linear(hidden2_size, action_size)

    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        self.linear3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        out = torch.cat([state, action], 1)
        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
        return self.linear3(out)
