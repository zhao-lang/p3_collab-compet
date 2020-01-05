# ADAPTED FROM MADDPG LAB

# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from agent import DDPGAgent
from utils import *

import torch
import torch.nn.functional as F

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-4               # learning rate
UPDATE_EVERY = 10        # how often to update the network
UPDATE_TIMES = 20       # how many time to learn for each update step
EPSILON_DECAY = 0.999

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPG:
    def __init__(self, state_size, action_size, seed, discount_factor=GAMMA, tau=TAU):
        super(MADDPG, self).__init__()

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # critic input = obs_full + actions = 14+2+2+2=20
        self.maddpg_agent = [
            DDPGAgent(state_size, action_size, seed), 
            DDPGAgent(state_size, action_size, seed)
        ]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.t_step = 0
        

    def reset(self):
        for _, ddpg_agent in enumerate(self.maddpg_agent):
            ddpg_agent.noise.reset()

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                for _ in range(UPDATE_TIMES):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        for i, agent in enumerate(self.maddpg_agent):
        
            # update critic
            Q_expected = agent.critic_local(states, actions)
            next_actions = agent.policy_target(next_states)
            Q_targets_next = agent.critic_target(next_states, next_actions)
            Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
            critic_loss = F.mse_loss(Q_expected, Q_targets)

            # print("CRITIC LOSS:", critic_loss)

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
            agent.critic_optimizer.step()

            # update actor
            predicted_actions = agent.policy_local(states)
            policy_loss = -agent.critic_local(states, predicted_actions).mean()

            # print("POLICY LOSS:", policy_loss)

            agent.policy_optimizer.zero_grad()
            policy_loss.backward()
            agent.policy_optimizer.step()

            agent.epsilon *= EPSILON_DECAY

            # ------------------- update target networks ------------------- #
            self.soft_update(agent.policy_local, agent.policy_target, TAU)   
            self.soft_update(agent.critic_local, agent.critic_target, TAU)

            self.iter += 1


    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.policy_local for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.policy_target for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, add_noise=True):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, add_noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            




