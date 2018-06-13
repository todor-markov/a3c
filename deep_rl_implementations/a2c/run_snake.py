import time
import gym
import gym_snake
import numpy as np
import tensorflow as tf
from train import a2c_train
from env_wrappers import StackedEnv, WarpFrame, ScaledFloatFrame
from policies import CnnActorCriticD
from util import get_policy_rewards, render_policy, set_global_seeds
from util import create_a2c_parser


if __name__ == '__main__':
    parser = create_a2c_parser()
    args = parser.parse_args()

    t = time.time()

    set_global_seeds(args.seed)

    env_generator = lambda: StackedEnv(
        ScaledFloatFrame(WarpFrame(gym.make('Snake-v0'))),
        stack_size=4,
        stack_axis=-1)

    env = env_generator()
    policy = CnnActorCriticD(
        'policy', env.observation_space.shape, env.action_space.n)

    if args.mode == 'train':
        optimizer_class = tf.train.AdamOptimizer
        optimizer_kwargs = {}

        if args.restore_path is not None:
            sess = tf.Session()
            sess.__enter__()
            policy.restore(args.restore_path)

        a2c_train(
            policy,
            env_generator,
            num_env_workers=args.num_workers,
            num_iter=args.num_iter,
            time_horizon=args.time_horizon,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            init_learning_rate=args.learning_rate,
            learning_rate_schedule=args.lrschedule,
            max_grad_norm=0.5,
            seed=args.seed,
            log_interval=args.log_interval,
            log_dir=args.log_dir,
            save_interval=args.save_interval,
            save_path=args.save_path)

        if args.restore_path is not None:
            sess.close()

        print('\nElapsed time: {}'.format(time.time()-t))

    if args.mode == 'eval':
        sess = tf.Session()
        sess.__enter__()
        policy.restore(args.restore_path)
        rewards = get_policy_rewards(policy, env, n_iter=20)
        print('Average reward: {0}\nReward std: {1}\n'
              'Min reward: {2}\nMax reward: {3}\n\n{4}\n'
              .format(np.mean(rewards), np.std(rewards),
                      min(rewards), max(rewards), rewards))

        for _ in range(1):
            print(render_policy(policy, env))

        sess.close()

    env.close()
