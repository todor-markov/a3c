import random
import numpy as np
import cv2
import gym
from multiprocessing import Process, Pipe
from collections import deque
from deep_rl_implementations.common.util import stack_space


def env_worker(pipe, env):
    while True:
        command, action = pipe.recv()
        if command == 'step':
            observation, reward, done, info = env.step(action)
            if done:
                observation = env.reset()
            pipe.send((observation, reward, done, info))
        elif command == 'reset':
            observation = env.reset()
            pipe.send(observation)
        elif command == 'close':
            env.close()
            pipe.close()
            break
        elif command == 'seed':
            env.seed(action)
            pipe.send(True)
        else:
            raise NotImplementedError


class MultiprocessEnv(object):

    def __init__(self, env_generator, num_envs, seeds=None):
        self.closed = False
        self.num_envs = num_envs
        self.processes = []
        self.pipes = []

        env = env_generator()
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        for i in range(num_envs):
            pipe_local, pipe_remote = Pipe()
            curr_env = env_generator()
            if seeds is not None:
                curr_env.seed(seeds[i])
            else:
                curr_env.seed(random.randint(1, 1000000000))
            process = Process(
                target=env_worker,
                args=(pipe_remote, curr_env),
                daemon=True)

            process.start()
            self.processes.append(process)
            self.pipes.append(pipe_local)

    def seed(self, seeds):
        assert len(seeds) == len(self.pipes), (
            "Error: number of seeds provided different from number of envs")

        updated_env_indices = []
        for i in range(len(seeds)):
            if seeds[i] is None:
                continue

            self.pipes[i].send(('seed', seeds[i]))
            updated_env_indices.append(i)

        for i in updated_env_indices:
            self.pipes[i].recv()

    def step(self, actions):
        for i in range(self.num_envs):
            self.pipes[i].send(('step', actions[i]))

        observations = []
        rewards = []
        dones = []
        infos = []

        for pipe in self.pipes:
            observation, reward, done, info = pipe.recv()
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (observations, rewards, dones, infos)

    def reset(self):
        for pipe in self.pipes:
            pipe.send(('reset', None))

        observations = [pipe.recv() for pipe in self.pipes]
        return observations

    def close(self):
        if self.closed:
            return
        for pipe in self.pipes:
            pipe.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work.

        This class was taken directly from the OpenAI baselines code
        Link to source file:
        https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.

        This class was taken directly from the OpenAI baselines code
        Link to source file:
        https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.

        This class was taken directly from the OpenAI baselines code
        Link to source file:
        https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing.

        This class was taken directly from the OpenAI baselines code
        Link to source file:
        https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
        """
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame

        This class was taken directly from the OpenAI baselines code
        Link to source file:
        https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)

            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs

            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        """
        This class was taken directly from the OpenAI baselines code
        Link to source file:
        https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
        """
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class StackedEnv(gym.Wrapper):

    def __init__(self, env, stack_size, stack_axis):
        """Stack environment observations along a given axis

        This class was inspired by the OpenAI baselines FrameStack class
        Link to source file for FrameStack class:
        https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
        """
        gym.Wrapper.__init__(self, env)
        self.stack_size = stack_size
        self.stack_axis = stack_axis
        self.curr_stack = deque([], maxlen=stack_size)
        self.observation_space = stack_space(
            env.observation_space, stack_size, stack_axis)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.curr_stack.append(obs)
        stacked_obs = np.concatenate(self.curr_stack, axis=self.stack_axis)
        return (stacked_obs, reward, done, info)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.stack_size):
            self.curr_stack.append(obs)

        stacked_obs = np.concatenate(self.curr_stack, axis=self.stack_axis)
        return stacked_obs


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """
        This class was taken directly from the OpenAI baselines code
        Link to source file:
        https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
        """
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


def make_atari(env_id):
    """
    This function was taken directly from the OpenAI baselines code
    Link to source file:
    https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind(env,
                  episode_life=True,
                  clip_rewards=True,
                  frame_stack=True,
                  scale=True):
    """Configure environment for DeepMind-style Atari.

    This function was taken directly from the OpenAI baselines code
    Link to source file:
    https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = StackedEnv(env, 4, -1)
    return env
