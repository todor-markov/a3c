import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize

class ActorCriticD(object):
    """A generic ActorCritic class for environments with discrete actions
    Actual ActorCritic policy classes inherit from this class
    """
    def __init__(self, name, obs_shape, action_dims):
        with tf.variable_scope(name):
            self._init(obs_shape, action_dims)
            self.scope = tf.get_variable_scope().name

        
        self.saver = tf.train.Saver(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope))

    def _init(self, obs_shape, action_dims):
        input_shape = [None]
        input_shape.extend(obs_shape)
        self.X = tf.placeholder(dtype=tf.float32, shape=input_shape)

        self._setup_actor_critic(action_dims)
        self.probabilities = tf.nn.softmax(self.logits)

    def _setup_actor_critic(self, action_dims):
        raise NotImplementedError

    def get_logits(self, observations=None):
        if observations is None:
            return self.logits
        else:
            return tf.get_default_session().run(
                self.logits, feed_dict={self.X: observations})

    def get_values(self, observations=None):
        if observations is None:
            return self.values
        else:
            return tf.get_default_session().run(
                self.values, feed_dict={self.X: observations})

    def get_probabilities(self, observations=None):
        if observations is None:
            return self.probabilities
        else:
            return tf.get_default_session().run(
                self.probabilities, feed_dict={self.X: observations})

    def choose_actions(self, observations):
        probabilities = tf.get_default_session().run(
            self.probabilities, feed_dict={self.X: observations})

        # np.random.multinomial implicitly casts to float64, which can create
        # errors in the multinomial from the sum of probabilities being > 1.
        # to prevent that, we preemptively cast to float64 and normalize
        probabilities = normalize(
            probabilities.astype(np.float64),
            norm='l1',
            axis=1)

        return [np.argmax(np.random.multinomial(1, probabilities[i]))
            for i in range(observations.shape[0])]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def save(self, save_path):
        return self.saver.save(
            tf.get_default_session(),
            save_path,
            global_step=tf.train.get_global_step())

    def restore(self, save_path):
        return self.saver.restore(tf.get_default_session(), save_path)


class MlpActorCriticD(ActorCriticD):
    """An actor-critic policy with multilayer perceptron networks for both
    the actor and the critic.

    Note that this class currently assumes that the environment observations
    are 1-dimensional arrays.  
    """
    def __init__(self, name, obs_shape, action_dims):
        super().__init__(name, obs_shape, action_dims)

    def _setup_actor_critic(self, action_dims):
        with tf.variable_scope('actor'):
            net = tf.layers.dense(inputs=self.X, units=32, activation=tf.nn.tanh)
            self.logits = tf.layers.dense(inputs=net, units=action_dims)
            self.actor_scope = tf.get_variable_scope().name

        with tf.variable_scope('critic'):
            values = tf.layers.dense(
                inputs=self.X,
                units=64,
                activation=tf.nn.relu)

            values = tf.layers.dense(
                inputs=values,
                units=64,
                activation=tf.nn.relu)

            self.values = tf.layers.dense(inputs=values, units=1)
            self.critic_scope = tf.get_variable_scope().name

    def get_trainable_actor_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.actor_scope)

    def get_trainable_critic_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.critic_scope)


class CnnActorCriticD(ActorCriticD):
    """An actor-critic policy with convolutional neural nets for the actor and
    critic. The nets follow the architecture outlined in Mnih et al. 2013:

    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

    The nets do not share parameters.
    The observation batches are assumed to have shape N x 84 X 84 x C
    """
    def __init__(self, name, obs_shape, action_dims):
        super().__init__(name, obs_shape, action_dims)

    def _setup_actor_critic(self, action_dims):
        net = tf.layers.conv2d(
            inputs=self.X,
            filters=16,
            kernel_size=8,
            strides=(4, 4),
            activation=tf.nn.relu)

        net = tf.layers.conv2d(
            inputs=net,
            filters=32,
            kernel_size=4,
            strides=(2, 2),
            activation=tf.nn.relu)

        net = tf.layers.flatten(net)
        net = tf.layers.dense(inputs=net, units=256, activation=tf.nn.relu)
        self.logits = tf.layers.dense(inputs=net, units=action_dims)

        # values = tf.layers.conv2d(
        #     inputs=self.X,
        #     filters=16,
        #     kernel_size=8,
        #     strides=(4, 4),
        #     activation=tf.nn.relu)

        # values = tf.layers.conv2d(
        #     inputs=values,
        #     filters=32,
        #     kernel_size=4,
        #     strides=(2, 2),
        #     activation=tf.nn.relu)

        # values = tf.layers.flatten(values)
        # values = tf.layers.dense(inputs=values, units=256, activation=tf.nn.relu)
        self.values = tf.layers.dense(inputs=net, units=1)
