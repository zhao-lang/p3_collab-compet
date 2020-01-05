from unityagents import UnityEnvironment
import numpy as np

from maddpg import MADDPG

import torch


def play():
    env = UnityEnvironment(file_name='./Tennis.app')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))

    # create agent
    maddpg_agent = MADDPG(state_size=state_size, action_size=action_size, seed=0)

    # load weights
    for i, agent in enumerate(maddpg_agent.maddpg_agent):
        agent.policy_local.load_state_dict(torch.load('models/checkpoint_actor_{}.pth'.format(i)))

    # reverse weights so agent 1 is on the left instead
    # for i, agent in enumerate(reversed(maddpg_agent.maddpg_agent)):
    #     agent.policy_local.load_state_dict(torch.load('models/checkpoint_actor_{}.pth'.format(i)))

    env_info = env.reset(train_mode=False)[brain_name]         # reset the environment    
    states = env_info.vector_observations                      # get the current state (for each agent)
    scores = np.zeros(num_agents)                              # initialize the score (for each agent)
    while True:
        actions = maddpg_agent.act(states, add_noise=False)    # select an action (for each agent)
        env_info = env.step(actions)[brain_name]               # send all actions to tne environment
        next_states = env_info.vector_observations             # get next state (for each agent)
        rewards = env_info.rewards                             # get reward (for each agent)
        dones = env_info.local_done                            # see if episode finished
        scores += rewards                                      # update the score (for each agent)
        states = next_states                                   # roll over states to next time step
        if np.any(dones):                                      # exit loop if episode finished
            break

    print('Agent 0 score this episode: {}'.format(scores[0]))
    print('Agent 0 score this episode: {}'.format(scores[1]))

    env.close()


if __name__ == "__main__":
    play()
