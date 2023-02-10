import random
import time
import torch
import gym
import gym_hybrid
# import gym_goal
import gym_HFO
import numpy as np
# from datetime import datetime
env = gym.make('Moving-v0', seed=0, penalty=0, max_step=100)
for i in range(50):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
