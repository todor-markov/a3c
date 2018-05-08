import numpy as np
from gym import spaces
from time import sleep

def render_policy(policy, env, pause=None):
    total_reward = 0
    observation = env.reset()
    while True:
        observation = observation.astype(np.float32)
        env.render()
        if pause is not None:
            sleep(pause)
        action = policy.choose_actions(np.asarray([observation]))[0]
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    return total_reward


def get_policy_rewards(policy, env, n_iter=10):
    cumulative_reward = 0
    rewards = []
    for i_episode in range(n_iter):
        observation = env.reset()
        while True:
            observation = observation.astype(np.float32)
            action = policy.choose_actions(np.asarray([observation]))[0]
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward
            if done:
                rewards.append(cumulative_reward)
                cumulative_reward = 0
                break

    return rewards


def discount_rewards(rewards, dones, discount_factor):
    discounted_rewards = []
    curr_reward = 0
    for j in range(len(rewards) - 1, -1, -1):
        curr_reward = rewards[j] + curr_reward * discount_factor * (1 - dones[j])
        discounted_rewards.append(curr_reward)

    return discounted_rewards[::-1]

def stack_space(space, stack_size, stack_axis):
    # We assume the space is of type Box
    if np.isscalar(space.low):
        shape = list(space.shape)
        shape[stack_axis] *= stack_size
        shape = tuple(shape)
        stacked_space = spaces.Box(
            low=space.low, high=space.high, shape=shape, dtype=space.dtype)
    else:
        low = np.concatenate(
            [space.low for _ in range(stack_size)], axis=stack_axis)
        high = np.concatenate(
            [space.high for _ in range(stack_size)], axis=stack_axis)
     
        stacked_space = spaces.Box(low=low, high=high, dtype=space.dtype)

    return stacked_space
