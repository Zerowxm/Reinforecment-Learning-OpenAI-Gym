%matplotlib inline

import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys


if "../" not in sys.path:
  sys.path.append("../") 

from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')

def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env, num_episodes, discount_factor=0.9, alpha=0.618, epsilon=0.1):
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()
        state = preprocess(state)
        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            
            # Take a step
            state = np.reshape(state,-1)
            state = tuple(state)
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

#             Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            next_state = preprocess(next_state)
            next_state = np.reshape(next_state, -1)
            next_state = tuple(next_state)
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            if done:
                break
                
            state = next_state
    
    return Q, stats


def random_exploration(env, num_episodes):
    reward = None
    done = None

    g = 0
    episodes = num_episodes
    rewardTracker = []
    env.render()
    
    while episodes:
        state, reward, done, info = env.step(env.action_space.sample())
        g += reward
        if done == True:
            rewardTracker.append(g)
            state = env.reset()
            episodes -= 1
            
    print("Reached goal after {} episodes with a average return of {}".format(num_episodes, sum(rewardTracker)/len(rewardTracker)))
