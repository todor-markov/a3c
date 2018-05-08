import os
import time
import sys
import gym
import gym_snake
import numpy as np
import tensorflow as tf
from train import a2c_train
from env_wrappers import StackedEnv, WarpFrame, make_atari, wrap_deepmind
from policies import ActorCriticD, MlpActorCriticD, CnnActorCriticD
from utils import get_policy_rewards, render_policy


if __name__ == '__main__':
    t = time.time()

    env_name = sys.argv[1]
    mode = sys.argv[2]
    path = sys.argv[3] if sys.argv[3] != 'None' else None
    restore_path = None
    if len(sys.argv) == 5 and mode == 'train':
        restore_path = sys.argv[4]

    if env_name == 'Snake-v0':
        env = StackedEnv(
            WarpFrame(gym.make('Snake-v0')), stack_size=4, stack_axis=-1)

        policy = CnnActorCriticD(
            'policy', env.observation_space.shape, env.action_space.n)
    else:
        env = wrap_deepmind(make_atari(env_id))
        policy = CnnActorCriticD(
            'policy', env.observation_space.shape, env.action_space.n)

    if mode == 'train':
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        if restore_path is not None:
            sess = tf.Session()
            sess.__enter__()
            policy.restore(restore_path)

        a2c_train(
            policy,
            env,
            num_env_workers=16, 
            optimizer=optimizer,
            max_grad_norm=None,
            time_horizon=5,
            num_iter=500,
            log_interval=100,
            save_interval=500,
            save_path=path)

        if restore_path is not None:
            sess.close()

        print('\nElapsed time: {}'.format(time.time()-t))

    if mode == 'eval':
        sess = tf.Session()
        sess.__enter__()
        policy.restore(path)
        rewards = get_policy_rewards(policy, env, n_iter=20)
        print('Average reward: {0}\nReward std: {1}\n'
              'Min reward: {2}\nMax reward: {3}\n\n{4}\n'
              .format(np.mean(rewards), np.std(rewards),
                      min(rewards), max(rewards), rewards))

        for _ in range(1):
            print(render_policy(policy, env, pause=0.05))

        sess.close()

    env.close()
