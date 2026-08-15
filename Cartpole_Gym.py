import random
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()

for i in range(100):
    env.render()

    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )

    if terminated or truncated:
        observation, info = env.reset()

env.close()
