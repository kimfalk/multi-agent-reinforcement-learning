[//]: # (Image References)

# Udacity DRL project 3: Multi agent reinforcement learning

In the following, the learning algorithm is described, along with the chosen hyperparameters. It also describes the 
model architectures for the neural networks used.

Environment solved in 863 episodes. Average Score: 0.50

[image1]: https://github.com/kimfalk/multi-agent-reinforcement-learning/blob/main/images/reward_chart.png?raw=true "Trained Agent"
          
## Simulation Environment
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, 
it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a 
reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each 
agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward 
(or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 
consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. 
This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.

This yields a single score for each episode. 

## The Learning Algorithm 

The agent is implemented using the Deep Deterministic Policy Gradient (DDPG) algorithm [1]. The DDPG builds upon the 
Deterministic Policy Gradient (DPG)[2]. The DPG was special because it was the first policy gradient algorithm not to 
be stochastic. It handles exploration by adding noise to the action values, which will make the agent explore different 
values in the action space. The DDPG builds on top of the DPG and adding learnings from the DQN algorithm [3]. 

The DDPG comprises an actor and a critic part. That actor will approximate the policy, while the critic approximates an 
advantage function. Our goal is to create a policy which will recommend the right actions in each state, so why the 
advantage function then, you might ask. The advantage function helps the agent to understand how valuable the decision 
was. If your agent is in a state where all actions are reasonable, it should put too much value into which action it 
had chosen there. So the critic is there to make the policy understand when it did good or bad. 

One of the learnings used from the DQN is that of having two neural networks for each model, one which is the target 
network, and one is the online network. This trick enables the agent to learn more steadily and not fluctuate. 

## The Neural Networks Architecture

The Actor network(s) comprises two hidden layers, with 512 and 256 nodes respectively both using the RELU activation 
function. Both layers are preceded with a batch normalisation. The actor network output layer is as big as the action 
space and uses the tanh activation function. 

The Critic network(s) are a bit different, first layer is a linear layer with 512 nodes using the RELU activation 
function, the layer is followed by batch normalisation. The following layer combines the output of the first layer 
with the action. This layer also contains 256 nodes and uses RELU activation. Lastly, the output layer contains only 
one node and doesn't have an activation method. 

The challenge in this environment is that we have two agents in each step. But since this in an off-policy trained 
algorithm, it can be solved simply by training one agent that reacts to each agent based on the state. 

## Hyperparameters

``` python
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```

## Further work

There are many ways you could improve on this. I believe that the solution that I have is pretty good. But I would 
have liked to test out other algorithms and see if I could have done it even better. Staying with the DDPG, I would 
start out focusing on how the noise is applied to the actions. It could be interesting to see if it would be worth 
adding decay to the noise, or something like momentum which could be tied to the reward. I would also like to try
the MADDPG[4], to verify if that would improve the solution. 

[1 - Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
[2 - Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)
[3 - Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
[4 - Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
