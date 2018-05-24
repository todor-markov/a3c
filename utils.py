import os
import numpy as np
import tensorflow as tf
import random
import argparse
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
        curr_reward = (
            rewards[j] + curr_reward * discount_factor * (1 - dones[j]))
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


def get_learning_rate(init_rate, schedule, curr_iter, total_num_iters):
    if schedule == 'constant':
        return init_rate
    elif schedule == 'linear':
        return init_rate * (1 - float(curr_iter) / total_num_iters)
    else:
        raise NotImplementedError


def set_global_seeds(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_a2c_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument(
        '--num-workers',
        help='Number of environment workers',
        type=int,
        default=16)
    parser.add_argument(
        '--num-iter', help='Number of iterations', type=int, default=625000)
    parser.add_argument(
        '--time_horizon',
        help='Time horizon for policy rollouts',
        type=int,
        default=5)
    parser.add_argument(
        '--learning-rate', help='Learning rate', type=float, default=7e-4)
    parser.add_argument(
        '--lrschedule',
        help='Learning rate schedule',
        choices=['constant', 'linear'],
        default='linear')
    parser.add_argument(
        '--log-interval', help='Log at this interval', type=int, default=100)
    parser.add_argument(
        '--log-dir', help='Logging directory', default='logging/')
    parser.add_argument(
        '--save-interval',
        help='Save the model at this interval',
        type=int,
        default=125000)
    parser.add_argument(
        '--save-path',
        help='Path where the model is saved',
        default='policies/Atari/')
    parser.add_argument(
        '--restore-path',
        help='Path from which to restore the model',
        default=None)
    parser.add_argument(
        '--mode',
        help='Training or evaluation',
        choices=['train', 'eval'],
        default='train')

    return parser


class Logger(object):

    def __init__(self, log_dir):
        self._log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._logtxt = open(log_dir + 'log.txt', 'w+')
        self._losscsv = open(log_dir + 'loss.csv', 'w+')
        self._losscsv.write(
            'iteration,policy_loss,value_loss,entropy_loss,total_loss\n')

    def log(self, message):
        self._logtxt.write(message)

    def log_loss(self, i, pg_loss, vf_loss, entropy_loss, total_loss):
        self._losscsv.write('{0},{1},{2},{3},{4}\n'.format(
            i, pg_loss, vf_loss, entropy_loss, total_loss))

    def close(self):
        self._logtxt.close()
        self._losscsv.close()
