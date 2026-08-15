import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")

print(env.action_space)
print(env.observation_space)
print(env.action_space.sample())

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

env = gym.make("CartPole-v1")

observation, info = env.reset()

terminated = False
truncated = False

while not (terminated or truncated):
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )

    print(f"{observation} -> {reward}")

env.close()

print(env.observation_space.low)
print(env.observation_space.high)


def discretize(x):
    return tuple((x / np.array([0.25, 0.25, 0.01, 0.1])).astype(int))


def create_bins(interval, num):
    return np.arange(num + 1) * (interval[1] - interval[0]) / num + interval[0]


intervals = [
    (-5, 5),
    (-2, 2),
    (-0.5, 0.5),
    (-2, 2),
]

nbins = [20, 20, 10, 10]

bins = [create_bins(intervals[i], nbins[i]) for i in range(4)]


def discretize_bins(x):
    return tuple(np.digitize(x[i], bins[i]) for i in range(4))


observation, info = env.reset()

terminated = False
truncated = False

while not (terminated or truncated):
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )

    print(discretize(observation))

env.close()

Q = {}

actions = (0, 1)


def qvalues(state):
    return [Q.get((state, action), 0) for action in actions]


alpha = 0.3
gamma = 0.9
epsilon = 0.10


def q_probabilities(values, eps=1e-4):
    values = values - values.min() + eps
    return values / values.sum()


state = discretize(observation)
print(q_probabilities(np.array(qvalues(state))))

# Modernization
Qmax = 0

rewards = []
average_rewards = []

for epoch in range(100_000):
    observation, info = env.reset()

    terminated = False
    truncated = False
    cumulative_reward = 0

    while not (terminated or truncated):
        state = discretize(observation)

        # Exploration vs exploitation
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            values = np.array(qvalues(state))
            action = int(np.argmax(values))

        # Take action
        next_observation, reward, terminated, truncated, info = env.step(action)

        cumulative_reward += reward

        next_state = discretize(next_observation)

        # Q-learning update
        if terminated or truncated:
            target = reward
        else:
            target = reward + gamma * max(qvalues(next_state))

        Q[(state, action)] = (1 - alpha) * Q.get((state, action), 0) + alpha * target

        observation = next_observation

    rewards.append(cumulative_reward)

    if (epoch + 1) % 5000 == 0:
        average_reward = np.mean(rewards[-5000:])

        print(
            f"{epoch + 1}: "
            f"average reward = {average_reward:.2f}, "
            f"alpha = {alpha}, "
            f"epsilon = {epsilon}"
        )

        if average_reward > Qmax:
            Qmax = average_reward
            Qbest = Q.copy()


plt.plot(rewards)


def running_average(x, window):
    return np.convolve(
        x,
        np.ones(window) / window,
        mode="valid",
    )


plt.plot(running_average(rewards, 100))
plt.xlabel("Episode")
plt.ylabel("Average reward")
plt.show()

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()

terminated = False
truncated = False

while not (terminated or truncated):
    state = discretize(observation)

    env.render()

    values = np.array(qvalues(state))
    action = int(np.argmax(values))

    observation, reward, terminated, truncated, info = env.step(action)

env.close()
