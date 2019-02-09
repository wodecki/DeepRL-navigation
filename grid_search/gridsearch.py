#!/usr/bin/env python
# coding: utf-8

# # Navigation - Deep Q-Network implementation

# import necessary packages
from unityagents import UnityEnvironment
import sys
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


# Instantiate the Environment and Agent

env = UnityEnvironment(file_name="Banana_linux/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of actions
action_size = brain.vector_action_space_size

# examine the state space
state = env_info.vector_observations[0]
state_size = len(state)

#import the Agent
from dqn_agent import Agent


# Train the Agent with DQN
def dqn(LR, GAMMA, TAU, BUFF, UPD, n_episodes=1000, max_t=100, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    agent = Agent(state_size, action_size, LR, GAMMA, TAU, BUFF, UPD, seed=0)
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
#        if np.mean(scores_window)>=13.0:
#            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            #break
#    return scores
    return np.mean(scores_window)



# choose Your hyperparameters below
br = np.linspace(5e4, 5e5, num=5)
ur = np.linspace(2, 10, num=5)


with open('results.txt', 'a') as output:
    for b in br:
        for u in ur:
            scores = dqn(1e-4, 0.99, 0.002575, b, u)
            print(b, u, 'scores: ', scores)
            output.writelines(str(b)+','+str(u)+','+str(scores)+'\n')
            output.flush()


"""
# The hyperparameter setup for "1. lrate gamma tau.txt"
lr = np.linspace(1e-5, 1e-4, num=5)
gr = np.linspace(0.90, 0.99, num=5)
tr = np.linspace(1e-4, 1e-2, num=5)

with open('results.txt', 'a') as output:
    for l in lr:
        for g in gr:
            for t in tr:
                scores = dqn(l, g, t)
                print(l, g, t, 'scores: ', scores)
                output.writelines(str(l)+','+str(g)+','+str(t)+','+str(scores)+'\n')
                output.flush()
"""
