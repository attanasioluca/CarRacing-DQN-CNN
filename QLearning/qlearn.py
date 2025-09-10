import gymnasium as gym
from gymnasium import Env
import numpy as np
from collections import defaultdict
from policies import EpsGreedyPolicy


def qlearn(env: gym.Env, alpha0: float, gamma: float, max_steps: int):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = EpsGreedyPolicy(Q)
    done = True
    dist = 10
    
    for step in range(max_steps):
        if done:
            dist = 10
            obs = env.reset()[0]

        eps =  1 - step / max_steps # 1 to 0
        action = policy(obs, eps)
        obs2, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        Q[obs][action] = (1-alpha0)*Q[obs][action] + alpha0*(rew + gamma*np.max(Q[obs2]))
        obs = obs2
    return Q
    
'''
        new_dist = (abs((int(obs2/8)**2 + int(obs2%8)**2) - 2*49))**(1/2)
        if terminated and obs2 != 63:
            rew = -1
        if not (done):
            rew = 0.01 if dist > new_dist else -0.01 
            dist = new_dist
        '''