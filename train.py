import os
import numpy as np
import tensorflow as tf
from multiprocess_env import MultiprocessEnv
from utils import discount_rewards


def a2c_rollout(policy,
                menv,
                curr_observations=None,
                time_horizon=16,
                gamma=0.99):

    mb_observations = []
    mb_actions = []
    mb_rewards = []
    mb_dones = []
    # action_dtype = np.int32 if actor.type == 'discrete' else np.float32
    action_dtype = np.int32

    mb_observations_shape = np.concatenate(
        ((menv.num_envs * time_horizon,), menv.observation_space.shape))

    if curr_observations is None:
        observations = menv.reset()
    else:
        observations = curr_observations

    
    for t in range(time_horizon):
        actions = policy.choose_actions(
            np.array(observations, dtype=np.float32))
        mb_observations.append(observations)
        mb_actions.append(actions)

        observations, rewards, dones, infos = menv.step(actions)
        mb_rewards.append(rewards)
        mb_dones.append(dones)

    mb_observations = (np.asarray(mb_observations, dtype=np.float32)
                       .swapaxes(1, 0)
                       .reshape(mb_observations_shape))
    mb_actions = np.asarray(mb_actions, dtype=action_dtype).swapaxes(1, 0)
    mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
    mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

    last_values = policy.get_values(observations)

    for i in range(menv.num_envs):
        curr_env_rewards = mb_rewards[i].tolist()
        curr_env_dones = mb_dones[i].tolist()
        curr_env_last_value = last_values[i, 0]
        if curr_env_dones[-1] == 0:
            curr_env_rewards = discount_rewards(
                curr_env_rewards + [curr_env_last_value],
                curr_env_dones + [0],
                gamma)[:-1]
        else:
            curr_env_rewards = discount_rewards(
                curr_env_rewards,
                curr_env_dones,
                gamma)

        mb_rewards[i] = curr_env_rewards

    mb_rewards = mb_rewards.flatten()
    mb_actions = mb_actions.flatten()

    return (
        mb_observations,
        np.array(mb_actions),
        np.array(mb_rewards, dtype=np.float32),
        observations)


def a2c_train(policy,
              env,
              num_env_workers,
              optimizer,
              max_grad_norm=0.5,
              time_horizon=16,
              num_iter=1000,
              log_interval=100,
              save_interval=1000,
              save_path=None,
              create_new_session=False):

    R = tf.placeholder(dtype=tf.float32, shape=[None])
    A = tf.placeholder(dtype=tf.int32, shape=[None])
    logits = policy.get_logits()
    values = policy.get_values()
    probabilities = policy.get_probabilities()

    negative_likelihoods = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=A, logits=logits)

    weighted_negative_likelihoods = tf.multiply(
        negative_likelihoods,
        tf.subtract(R, values))

    entropy = tf.reduce_sum(
        tf.multiply(probabilities, -tf.log(probabilities + 1e-9)),
        axis=1)

    entropy_coeff = tf.constant(0.01)
    vf_coeff = tf.constant(0.5)
    pg_loss = tf.reduce_mean(weighted_negative_likelihoods)
    entropy_loss = tf.reduce_mean(entropy)
    vf_loss = tf.losses.mean_squared_error(R, tf.squeeze(values))

    loss = pg_loss + vf_coeff * vf_loss - entropy_coeff * entropy_loss
    policy_trainable_vars = policy.get_trainable_variables()
    grads = tf.gradients(loss, policy_trainable_vars)
    if max_grad_norm is not None:
        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
    else:
        grad_norm = tf.constant(0)
    grads_and_vars = list(zip(grads, policy_trainable_vars))

    train = optimizer.apply_gradients(
        grads_and_vars,
        global_step=tf.train.get_or_create_global_step())

    new_session_created = False
    if tf.get_default_session() is None or create_new_session:
        sess = tf.Session()
        sess.__enter__()
        sess.run(tf.global_variables_initializer())
        new_session_created = True
    else:
        # Initialize the optimizer variables
        vars_to_be_initialized = list(
            set(tf.global_variables()) - set(policy_trainable_vars))
        tf.get_default_session().run(
            tf.variables_initializer(vars_to_be_initialized))

    menv = MultiprocessEnv(env, num_env_workers)
    curr_obs = None

    count = 0
    for i in range(1, num_iter+1):
        obs, actions, rewards, curr_obs = a2c_rollout(
            policy,
            menv,
            curr_observations=curr_obs,
            time_horizon=time_horizon)

        (logits_np, values_np, probabilities_np,
         pg_loss_np, vf_loss_np, entropy_loss_np, loss_np,
         _, entropy_np, grad_norm_np) = tf.get_default_session().run(
         [logits, values, probabilities,
          pg_loss, vf_loss, entropy_loss, loss, train, entropy, grad_norm],
         feed_dict={policy.X: obs, A: actions, R: rewards})

        if i == 1 or i % log_interval == 0:
            print('Iteration {0}\n'
                  'Policy gradients loss: {1}\n'
                  'Value function loss: {2}\n'
                  'Entropy loss: {3}\n'
                  'Total loss: {4}\n'.format(
                    i, pg_loss_np, vf_loss_np, entropy_loss_np, loss_np))

            # print(logits_np)
            # print()
            # print(rewards)
            # print(values_np)
            # print()
            print(np.mean(probabilities_np, axis=0))
            print(probabilities_np.min(axis=0))
            # print(grad_norm_np / max_grad_norm)
            print()
            # print(entropy_np)
            # print()

        if i % save_interval == 0 and save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            policy.save(save_path)

        # if probabilities_np.min(axis=0).max() > 0.9999:
        #     count += 1
        #     if count >= 300:
        #         break

    if new_session_created:
        sess.close()
