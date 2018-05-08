import gym
import numpy as np
import tensorflow as tf
from multiprocess_env import MultiprocessEnv
from comp_graph_policies import ActorCriticD
from train import a2c_rollout
# env = gym.make('Pendulum-v0')
# menv = MultiprocessEnv(env, num_envs=4)
# menv.seed([1,None,3,None])

# for i_episode in range(1):
#     observation = menv.reset()
#     for t in range(5):
#         print(observation)
#         actions = [env.action_space.sample() for _ in range(4)]
#         observation, reward, done, info = menv.step(actions)
#         # print(observation, reward, done, info)

policy = ActorCriticD(name='policy_vars', obs_shape=(4,), action_dims=2)
env = gym.make('CartPole-v1')
menv = MultiprocessEnv(env, 4)
sess = tf.Session()
sess.__enter__()
tf.get_default_session().run(tf.variables_initializer(policy.get_variables()))
a2c_rollout(policy, menv)
