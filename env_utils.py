import copy
import random
import numpy as np
import cv2
import gym
from multiprocessing import Process, Pipe
from utils import stack_space


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

    def __init__(self, env, num_envs, seeds=None):
        self.closed = False
        self.num_envs = num_envs
        self.processes = []
        self.pipes = []

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        for i in range(num_envs):
            pipe_local, pipe_remote = Pipe()
            curr_env = copy.deepcopy(env)
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
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class StackedEnv(object):

    def __init__(self,
                 env,
                 stack_size,
                 stack_axis):

        self.env = env
        self.stack_size = stack_size
        self.stack_axis = stack_axis
        self.curr_stack = None
        self.action_space = env.action_space
        self.observation_space = stack_space(
            env.observation_space, stack_size, stack_axis)

    def seed(self, seed=None):
        self.env.seed(seed)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        indices = range(
            self.env.observation_space.shape[self.stack_axis], 
            self.observation_space.shape[self.stack_axis])
        self.curr_stack = np.concatenate(
            [np.take(self.curr_stack, indices, axis=self.stack_axis), obs],
            axis=self.stack_axis)

        return (self.curr_stack, reward, done, info)

    def reset(self):
        observations = []
        obs = self.env.reset()
        observations.append(obs)

        for _ in range(self.stack_size - 1):
            obs, _, done, _ = self.env.step(self.action_space.sample())
            if done:
                print('Warning: episode terminated during the reset')
                self.curr_stack = None
                return None

            observations.append(obs)

        self.curr_stack = np.concatenate(observations, axis=self.stack_axis)
        return self.curr_stack

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
