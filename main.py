# main function that sets up environments
# perform training loop

from unityagents import UnityEnvironment

from collections import deque
from utils import ReplayBuffer
from maddpg import MADDPG

import torch
import numpy as np
import os
import matplotlib.pyplot as plt


N_EPISODES = 2000
EPISODE_LEN = 1000

GOAL_SCORE = 0.5
SCORE_WINDOW = 100
LOG_EVERY = 1


def train(env, model_dir, n_episodes=150, max_t=1000):
    seed = 12345

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

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
    print('The state for the first agent looks like:', states[0])
    
    # initialize policy and critic
    maddpg_agent = MADDPG(state_size=state_size, action_size=action_size, seed=seed)

    scores = []
    avg_scores = []
    scores_window = deque(maxlen=SCORE_WINDOW)

    for i_episode in range(1, n_episodes+1):

        env_info = env.reset(train_mode=True)[brain_name]
        maddpg_agent.reset()

        states = env_info.vector_observations  
        tmp_scores = np.zeros(num_agents)

        for t in range(max_t):
            actions = maddpg_agent.act(states, add_noise=True)

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                maddpg_agent.step(state, action, reward, next_state, done)

            states = next_states
            tmp_scores += rewards

            if np.any(dones):
                break 
        
        score = np.max(tmp_scores)
        scores_window.append(score)
        scores.append(score)
        avg_scores.append(np.mean(scores_window))

        if i_episode % LOG_EVERY == 0:
            print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(
                i_episode, score, np.mean(scores_window)))

    for i, agent in enumerate(maddpg_agent.maddpg_agent):
        torch.save(agent.policy_local.state_dict(), '{}/checkpoint_actor_{}.pth'.format(model_dir, i))
        torch.save(agent.critic_local.state_dict(), '{}/checkpoint_critic_{}.pth'.format(model_dir, i))
                  
    return scores, avg_scores


def main():
    env = UnityEnvironment(file_name="Tennis.app")

    model_dir = os.getcwd()+"/models"
    
    os.makedirs(model_dir, exist_ok=True)
    
    scores, avg_scores = train(env, model_dir, N_EPISODES, EPISODE_LEN)

    env.close()

    # plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores, label='score')
    plt.plot(np.arange(len(avg_scores)), avg_scores, c='r', label='avg score')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(loc='upper left')
    plt.savefig('scores.png')
    

if __name__=='__main__':
    main()
