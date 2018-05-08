import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import gym
import gym_snake
from utils import render_policy, get_policy_rewards, discount_rewards
from multiprocess_env import MultiprocessEnv, StackedEnv, WarpFrame
from policies import DiscreteActor, ContinuousActor, Critic

SEEDS = [12122, 90121, 28029, 77440, 13220, 78492, 62352, 27096, 30654,
         85554,   948, 19103, 18936, 85137,  7393, 29512, 44751,  9535,
         3440,  98157, 42169, 44271, 32326, 53868, 57938, 64561, 52987,
         81458, 80909, 40040, 63889, 26850, 73621, 20854, 16559, 28053,
         57269, 41736, 65976, 60886,  5526,  6741, 72993, 53906, 43238,
         26718, 99521,   608,  6146, 45512, 61641, 80748, 86341, 53213,
         65830,  9705, 87879,   768, 29154, 21465, 23496, 97233, 68134]


class A2COptimizer(object):

    def __init__(self, actor_optimizer, critic_optimizer):
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

    def actor_loss_discrete(self,
                            actor,
                            observations,
                            actions,
                            r_values,
                            critic_state_values,
                            entropy_coeff=0.01):

        logits = actor.get_action_logits(observations)
        negative_likelihoods = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=actions, logits=logits)

        weighted_negative_likelihoods = (
            negative_likelihoods * (r_values - critic_state_values))

        action_probabilities = tf.nn.softmax(logits)
        entropy = entropy_coeff * tf.reduce_mean(tf.reduce_sum(
            action_probabilities * -tf.log(action_probabilities),
            axis=1))

        return tf.reduce_mean(weighted_negative_likelihoods) - entropy

    def actor_loss_continuous(self,
                              actor,
                              observations,
                              actions,
                              r_values,
                              critic_state_values,
                              entropy_coeff=0.01):

        action_mean_vals = actor.get_action_values(observations)
        squared_diff = tf.square((actions - action_mean_vals))
        negative_likelihoods = (
            0.5 * (squared_diff * tf.exp(-2 * actor.log_stdevs)) +
            actor.log_stdevs)

        weighted_negative_likelihoods = (
            negative_likelihoods * (r_values - critic_state_values))

        entropy = entropy_coeff * tf.reduce_mean(actor.log_stdevs)

        return tf.reduce_mean(weighted_negative_likelihoods) - entropy

    def critic_loss(self, critic, observations, r_values):
        state_values = tf.reshape(
                critic.get_state_values(observations),
                shape=[observations.shape[0]])

        return tf.losses.mean_squared_error(r_values, state_values)

    def parallel_rollout(self,
                         actor,
                         critic,
                         env,
                         curr_observations=None,
                         time_horizon=10,
                         discount_factor=0.99):
        mb_observations = []
        mb_actions = []
        mb_rewards = []
        mb_dones = []
        mb_state_values = []
        mb_critic_states = []
        action_dtype = np.int32 if actor.type == 'discrete' else np.float32

        mb_observations_shape = np.concatenate(
            ((env.num_envs * time_horizon,), env.observation_space.shape))

        if curr_observations is None:
            observations = env.reset()
        else:
            observations = curr_observations

        # total_wins = 0
        for t in range(time_horizon):
            actions = actor.choose_actions(
                np.array(observations, dtype=np.float32))
            mb_observations.append(observations)
            mb_actions.append(actions)

            observations, rewards, dones, infos = env.step(actions)
            # for i in range(len(dones)):
            #     if dones[i] and mb_observations[-1][i][0] > 0.4:
            #         total_wins += 1
            mb_rewards.append(rewards)
            mb_dones.append(dones)

        # if total_wins:
        #     print('Total wins: {}'.format(total_wins))

        mb_observations = (np.asarray(mb_observations, dtype=np.float32)
                           .swapaxes(1, 0)
                           .reshape(mb_observations_shape))
        mb_actions = np.asarray(mb_actions, dtype=action_dtype).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        last_values = critic.get_state_values(observations)

        for i in range(env.num_envs):
            curr_env_rewards = mb_rewards[i].tolist()
            curr_env_dones = mb_dones[i].tolist()
            curr_env_last_value = last_values.numpy()[i, 0]
            if curr_env_dones[-1] == 0:
                curr_env_rewards = discount_rewards(
                    curr_env_rewards + [curr_env_last_value],
                    curr_env_dones + [0],
                    discount_factor)[:-1]
            else:
                curr_env_rewards = discount_rewards(
                    curr_env_rewards,
                    curr_env_dones,
                    discount_factor)

            mb_rewards[i] = curr_env_rewards

        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()

        return (
            mb_observations,
            np.array(mb_actions),
            np.array(mb_rewards, dtype=np.float32),
            observations)

    def optimize_policy(self,
                        actor,
                        critic,
                        env,
                        num_envs,
                        time_horizon=10,
                        n_iter=1000,
                        verbose=1):

        if actor.type == 'discrete':
            actor_loss = self.actor_loss_discrete
        if actor.type == 'continuous':
            actor_loss = self.actor_loss_continuous

        actor_grads = tfe.implicit_gradients(actor_loss)
        critic_grads = tfe.implicit_gradients(self.critic_loss)
        curr_obs = None

        multiproc_env = MultiprocessEnv(env, num_envs, seeds=None)

        for i in range(n_iter):
            observations, actions, r_values, curr_obs = self.parallel_rollout(
                actor,
                critic,
                multiproc_env,
                curr_observations=curr_obs,
                time_horizon=time_horizon)

            observations = tf.convert_to_tensor(observations, dtype=tf.float32)
            critic_state_values = tf.reshape(
                critic.get_state_values(observations),
                shape=[observations.shape[0]])

            if verbose >= 1 and (i+1) % 100 == 0:
                # mean_reward = get_policy_mean_reward(actor, env, n_iter=4)
                # mean_reward = 'N/A'
                # loss_actor = actor_loss(
                #     actor,
                #     observations,
                #     actions,
                #     r_values,
                #     critic_state_values)

                # loss_critic = self.critic_loss(critic, observations, r_values)
                # print('Iteration {0}.\nActor Loss: {1}. '
                #       'Critic loss: {2}.\nAverage reward: {3}\n'
                #       .format(i+1, loss_actor, loss_critic, mean_reward))

                # print(r_values[time_horizon-8:1*time_horizon])
                # print()
                # print(critic_state_values[time_horizon-8:1*time_horizon])
                # print()
                # print(actions[time_horizon-8:1*time_horizon])
                # print()
                # print(observations[time_horizon-8:time_horizon])
                # print()
                # if actor.type == 'continuous':
                #     print(actions[time_horizon-8:1*time_horizon])
                #     print(actor.log_stdevs)
                # if actor.type == 'discrete':
                #     probs = tf.nn.softmax(actor.get_action_logits(observations))
                #     print(probs[time_horizon-8:time_horizon, :])
                print(i)
                print('---------------------------------------------')

            # if verbose >= 2 and (i+1) % 1000 == 0:
            #     render_policy(actor, env, pause=0.01)

            self.actor_optimizer.apply_gradients(
                actor_grads(actor,
                            observations,
                            actions,
                            r_values,
                            critic_state_values)
            )

            self.critic_optimizer.apply_gradients(
                critic_grads(critic, observations, r_values))

        multiproc_env.close()
        return n_iter


if __name__ == '__main__':
    tf.enable_eager_execution()
    t = time.time()

    env_name = sys.argv[1]

    if env_name != 'Snake-v0':
        env = gym.make(env_name)
        stacked_env = StackedEnv(env, stack_size=4, stack_axis=0)

        actor_layers = [
            tf.layers.Dense(units=32, activation=tf.nn.tanh),
        ]

        if isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = env.action_space.n
            actor_layers.append(tf.layers.Dense(units=action_dim))
            actor = DiscreteActor(actor_layers)
        elif isinstance(env.action_space, gym.spaces.Box):
            action_dim = env.action_space.shape[0]
            actor_layers.append(tf.layers.Dense(units=action_dim))
            actor = ContinuousActor(actor_layers, action_dim)
        else:
            raise TypeError('Actor classes can only handle action spaces of'
                            ' type Discrete or Box')

        critic_layers = [
            tf.layers.Dense(units=64, activation=tf.nn.relu),
            tf.layers.Dense(units=64, activation=tf.nn.relu),
            tf.layers.Dense(units=1),
        ]
    else:
        env = WarpFrame(gym.make('Snake-v0'))

        actor_layers = [
            tf.layers.Conv2D(
                filters=16,
                kernel_size=8,
                strides=(4, 4),
                activation=tf.nn.relu),
            tf.layers.Conv2D(
                filters=32,
                kernel_size=4,
                strides=(2, 2),
                activation=tf.nn.relu),
            tf.layers.Flatten(),
            tf.layers.Dense(units=256, activation=tf.nn.relu),
            tf.layers.Dense(units=3)
        ]

        critic_layers = [
            tf.layers.Conv2D(
                filters=16,
                kernel_size=8,
                strides=(4, 4),
                activation=tf.nn.relu),
            tf.layers.Conv2D(
                filters=32,
                kernel_size=4,
                strides=(2, 2),
                activation=tf.nn.relu),
            tf.layers.Flatten(),
            tf.layers.Dense(units=256, activation=tf.nn.relu),
            tf.layers.Dense(units=1)
        ]

        actor = DiscreteActor(actor_layers)


    critic = Critic(critic_layers)
    actor_optimizer = tf.train.AdamOptimizer(1e-4)
    critic_optimizer = tf.train.AdamOptimizer()

    policy_optimizer = A2COptimizer(actor_optimizer, critic_optimizer)

    n_iter = policy_optimizer.optimize_policy(
        actor,
        critic,
        stacked_env,
        num_envs=16,
        time_horizon=16,
        n_iter=1000,
        verbose=2)

    print('\nElapsed time: {}'.format(time.time()-t))
    print('Number of iterations: {}'.format(n_iter))
    print('Average reward: {}'
          .format(np.mean(get_policy_rewards(actor, stacked_env, n_iter=100))))

    for _ in range(1):
        render_policy(actor, stacked_env)

    env.close()
