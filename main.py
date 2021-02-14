import sys
import random
import argparse
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from unityagents import UnityEnvironment

import matplotlib.pyplot as plt

from agent import Agent


def create_parser():
    """Initialize command line parser using arparse.

    defaults:

        BUFFER_SIZE = int(1e6)  # replay buffer size
        BATCH_SIZE = 512        # minibatch size
        GAMMA = 0.99            # discount factor
        TAU = 1e-3              # for soft update of target parameters
        LR_ACTOR = 1e-4         # learning rate of the actor
        LR_CRITIC = 3e-4        # learning rate of the critic
        WEIGHT_DECAY = 0        # L2 weight decay

    Returns:
        An argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-file', type=str, required=False,
                        default='./data/Tennis.app',
                        help='environment file')

    parser.add_argument('--replay-buffer-size', type=str, required=False,
                        default=int(1e6),
                        help='agent replay buffer size')

    parser.add_argument('--replay-batch-size', type=str, required=False,
                        default=512,
                        help='agent replay buffer size')

    parser.add_argument('--random-seed', type=str, required=False,
                        default=42,
                        help='random-seed to ensure reproducibility')

    parser.add_argument('--learning-rate-actor', type=str, required=False,
                        default=1e-4,
                        help='actor learning rate')

    parser.add_argument('--learning-rate-critic', type=str, required=False,
                        default=3e-4,
                        help='critic learning rate')

    parser.add_argument('--agent-gamma', type=str, required=False,
                        default=0.99,
                        help='gamma - please refer to algorithm doc for more information')

    parser.add_argument('--agent-tau', type=str, required=False,
                        default=1e-3,
                        help='tau - please refer to algorithm doc for more information')
    return parser


def initialize_env(file_name='./data/Tennis.app'):
    env = UnityEnvironment(file_name)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    return env, brain, brain_name


def train(agent,
          env,
          n_episodes=10000,
          max_t=2000,
          print_every=100,
          num_agents=2):
    """
    Connects the simulation to the agent.
    :param agent:
    :param env:
    :param n_episodes:
    :param max_t:
    :param print_every:
    :param num_agents:
    :return:
    """

    scores_deque = deque(maxlen=print_every)
    scores = []
    avg_scores = []
    brain_name = env.brain_names[0]

    for i_episode in range(1, n_episodes + 1):
        agent.reset()

        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations  # get next state (for each agent)

        score = np.zeros(num_agents)

        for t in range(max_t):

            pred_actions = np.array(agent.act(states))

            env_infos = env.step(pred_actions)[brain_name]

            rewards = env_infos.rewards  # get reward (for each agent)
            dones = env_infos.local_done  # see if episode finished
            score += env_infos.rewards  # update the score (for each agent)
            next_states = env_infos.vector_observations  # get next state (for each agent)

            for i in range(len(states)):
                agent.step(states[i], pred_actions[i], rewards[i], next_states[i], dones[i])

            states = next_states

            if any(dones):
                break

        episode_score = np.max(score)
        scores_deque.append(episode_score)
        scores.append(episode_score)
        average_score = np.mean(scores_deque)
        avg_scores.append(average_score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score), end="")

        if i_episode % print_every == 0:
            print('\rEpisode {}\t\tAverage Score: {:.2f}'.format(i_episode, average_score))

        if average_score > 0.5:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

            print('\rEnvironment solved in {} episodes. Average Score: {:.2f}'.format(i_episode, average_score))
            break

    return scores, avg_scores


def plot_scores(scores, avg_scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores, label="score")
    plt.plot(np.arange(1, len(avg_scores) + 1), avg_scores, label="average score")
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    now = datetime.now()
    plt.savefig('images/{}-episode_score'.format(now.strftime("%Y-%m-%d-%H-%M")))


def main(argv=None):

    argv = sys.argv if argv is None else argv
    args = create_parser().parse_args(args=argv[1:])

    random.seed(args.random_seed)
    env, brain, brain_name = initialize_env()

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

    agent = Agent(state_size=state_size,
                  action_size=action_size,
                  random_seed=args.random_seed,
                  buffer_size=args.replay_buffer_size,
                  batch_size=args.replay_batch_size,
                  learning_rate_actor=args.learning_rate_actor,
                  learning_rate_critic=args.learning_rate_critic,
                  gamma=args.agent_gamma,
                  tau=args.agent_tau)
    scores, avg_scores = train(agent, env)

    plot_scores(scores, avg_scores)


if __name__ == "__main__":
    main()
