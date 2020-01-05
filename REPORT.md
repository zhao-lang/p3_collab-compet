[//]: # (Image References)

[scores]: ./scores.png "Scores"

# Project 3: Collaboration and Competition

# Project Implementation

## Algorithm

I solved this task by adapting the DDPG agent from [p2_continuous-control](https://github.com/zhao-lang/p2-continuous-control) for multi-agent learning.

## Model

The actor and critic network models are defined in `model.Actor` and `model.Critic`,
with 2 fully-connected layers of (400, 300) neurons with RELU activation, followed by
another fully connected layer of `action_size` neurons.
For the actor networks the final layer usees tanh activation to limit output to (-1, 1).
For the critic networks, the final layer directly outputs action-values.

## Agent

The DDPG agent is defined in `agent.DDPGAgent`. The DDPG agent encapsulates the actor and critic
networks, and handles actions from state inputs. During learning,
Ornstein–Uhlenbeck process noise is added to the actions to encourage exploration.

The MADDPG agent is defined in `maddpg.MADDPG`. The MADDPG agent handles the learning process
for all agents in `MADDPG.learn()`, where the update steps for the actor and critic networks take place.

During training, the agents learn from observations independently of whether the player is on
the left or right, but during play, the agents only use observations local to itself.
This produces agents that can play on the left or right.

## Hyperparameters

Following hyperparameters were used:
* replay buffer size: 1e5
* minibatch size: 64 
* discount factor: 0.99 
* target network soft update factor: 1e-3  
* learning rate: 1e-4 
* update every n steps: 20
* number of learning steps for each update step: 10
* starting epsilon: 1.0
* epsilon decay: 0.999

## Results

The environment was solved in ~800 episodes, but was not stable. Max score stabilized at
`>1.1` after ~1400 episodes.

![Scores][scores]

## Observation

I originally thought that the agents were not learning as the scores were not improving
after hundreds of episodes, but that was actually due to the limited amount of training
episodes in the replay buffer. I could have used a warm-up period for the replay buffer
to become more filled before starting the training.

## Future improvements

Using batch normalization may be helpful in addressing some of the early stability issues.
In this particular environment, it doesn't really matter if the agent is on the left or right,
so in theory I could have trained 2 agents sharing the same actor network, which reduces
some of the redundancies in the learning process.

Last but not least more rigorous hyperparamter tuning would provide better results.
