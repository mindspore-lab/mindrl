# Copyright 2021-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MPEMultiEnvironment class."""
#pylint: disable=E0402
import os
from multiprocessing import Queue
import numpy as np
from gym import spaces

from mindspore.ops import operations as P

from mindspore_rl.environment.space import Space
from mindspore_rl.environment import Environment
from mindspore_rl.environment.env_process import EnvironmentProcess


def _prepare_mpe_env():
    '''prepare mpe env'''
    current_path = os.path.dirname(os.path.normpath(os.path.realpath(__file__)))
    os.chdir(current_path)
    # Clone mpe environment from marlbenchmark
    os.system('git clone https://github.com/marlbenchmark/on-policy.git')
    # Copy mpe folder to current directory
    os.system('cp -r on-policy/onpolicy/envs/mpe ./')
    # Download patch from mindspore_rl
    os.system(
        'wget https://gitee.com/mindspore/reinforcement/raw/master/example/mappo/src/mpe_environment.patch\
             --no-check-certificate')
    # patch mpe folder
    os.system('patch -p0 < mpe_environment.patch')


try:
    from .mpe.MPE_env import MPEEnv

except ModuleNotFoundError:
    _prepare_mpe_env()
    from .mpe.MPE_env import MPEEnv


class MPEMultiEnvironment(Environment):
    """
    This is the wrapper of Multi-Agent Particle Environment(MPE) which is modified by MAPPO author from
    (https://github.com/marlbenchmark/on-policy/tree/main/onpolicy). A simple multi-agent particle world with
    a continuous observation and discrete action space, along with some basic simulated physics.
    Used in the paper Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.

    Args:
        params (dict): A dictionary contains all the parameters which are used in this class.

            +------------------------------+--------------------------------------------------------+
            |  Configuration Parameters    |  Notices                                               |
            +==============================+========================================================+
            |  num                         |  Number of MPEEnvironment                              |
            |------------------------------|--------------------------------------------------------|
            |  name                        |  Name of environment in MPEEnvironment, like           |
            |                              |  simple_spread                                         |
            |------------------------------+--------------------------------------------------------|
            |  proc_num                    |  Number of process used in multi processing            |
            +------------------------------|--------------------------------------------------------+
        env_id (int): A integer which is used to set the seed of this environment.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> env_params = {"num": 10, "name": simple_spread, "proc_num": 32}
        >>> environment = MPEMultiEnvironment(env_params, 0)
        >>> print(environment)
    """

    def __init__(self,
                 params,
                 env_id=0):
        super(MPEMultiEnvironment, self).__init__()
        self.params = params
        self._nums = params["num"]
        self._proc_num = params['proc_num']
        self._env_name = params['name']
        self._num_agent = params['num_agent']
        self._envs = []
        self.env_id = env_id

        class AllArgs:
            def __init__(self, env_name, episode, num_agent, num_landmark):
                self.episode_length = episode
                self.num_agents = num_agent
                self.num_landmarks = num_landmark
                self.scenario_name = env_name

        all_args = AllArgs(self._env_name, 25, self._num_agent, self._num_agent)

        for i in range(self._nums):
            mpe_env = MPEEnv(all_args)
            mpe_env.seed(1 + i * 1000)
            self._envs.append(mpe_env)

        self._state_space = self._space_adapter(self._envs[0].observation_space[0], batch_shape=(self._num_agent,))
        self._action_space = self._space_adapter(self._envs[0].action_space[0], batch_shape=(self._num_agent,))
        self._reward_space = Space((1,), np.float32, batch_shape=(self._num_agent,))
        self._done_space = Space((1,), np.bool_, low=0, high=2, batch_shape=(self._num_agent,))

        step_input_shape = [(self._nums, self._num_agent, self._action_space.num_values)]
        step_output_shape = [((self._nums,) + self._state_space.shape),
                             ((self._nums,) + self._reward_space.shape),
                             ((self._nums,) + self._done_space.shape)]
        reset_output_shape = [((self._nums,) + self._state_space.shape)]

        self.step_ops = P.PyFunc(self._step,
                                 [self._action_space.ms_dtype,],
                                 step_input_shape,
                                 [self._state_space.ms_dtype, self._reward_space.ms_dtype, self._done_space.ms_dtype],
                                 step_output_shape)

        self.reset_ops = P.PyFunc(self._reset, [], [],
                                  [self._state_space.ms_dtype,],
                                  reset_output_shape)

        self.mpe_env_procs = []
        self.action_queues = []
        self.exp_queues = []
        self.init_state_queues = []

        if self._nums < self._proc_num:
            raise ValueError("Environment number can not be smaller than process number")

        avg_env_num_per_proc = int(self._nums / self._proc_num)
        for i in range(self._proc_num):
            action_q = Queue()
            self.action_queues.append(action_q)
            exp_q = Queue()
            self.exp_queues.append(exp_q)
            init_state_q = Queue()
            self.init_state_queues.append(init_state_q)

            assigned_env_num = i * avg_env_num_per_proc
            if assigned_env_num < self._nums:
                env_num = avg_env_num_per_proc
            else:
                env_num = self._nums - assigned_env_num

            env_proc = EnvironmentProcess(
                i, env_num, self._envs[env_num * i:env_num * (i+1)], action_q, exp_q, init_state_q)
            self.mpe_env_procs.append(env_proc)
            env_proc.start()

    @property
    def observation_space(self):
        """
        Get the state space of the environment.

        Returns:
            The state space of environment.
        """

        return self._state_space

    @property
    def action_space(self):
        """
        Get the action space of the environment.

        Returns:
            The action space of environment.
        """

        return self._action_space

    @property
    def reward_space(self):
        """
        Get the reward space of the environment.

        Returns:
            The reward space of environment.
        """
        return self._reward_space

    @property
    def done_space(self):
        """
        Get the done space of the environment.

        Returns:
            The done space of environment.
        """
        return self._done_space

    @property
    def config(self):
        """
        Get the config of environment.

        Returns:
            A dictionary which contains environment's info.
        """
        config_dict = {'global_observation_dim': self._num_agent * self._state_space.shape[-1]}
        return config_dict

    def step(self, action):
        r"""
        Execute the environment step, which means that interact with environment once.

        Args:
            action (Tensor): A tensor that contains the action information.

        Returns:
            - state (Tensor), the environment state after performing the action.
            - reward (Tensor), the reward after performing the action.
            - done (Tensor), whether the simulation finishes or not.
        """
        return self.step_ops(action)

    def reset(self):
        """
        Reset the environment to the initial state. It is always used at the beginning of each
        episode. It will return the value of initial state.

        Returns:
            A tensor which states for the initial state of environment.

        """
        return self.reset_ops()[0]

    def close(self):
        r"""
        Close the environment to release the resource.

        Returns:
            Success(np.bool\_), Whether shutdown the process or threading successfully.
        """
        for env in self._envs:
            env.close()
        for env_proc in self.mpe_env_procs:
            env_proc.terminate()
            env_proc.join()
        return True

    def _step(self, actions):
        """Inner step function"""
        accum_env_num = 0
        for i in range(self._proc_num):
            env_num = self.mpe_env_procs[i].env_num
            self.action_queues[i].put(actions[accum_env_num: accum_env_num+env_num,])
            accum_env_num += env_num
        results = []
        for i in range(self._proc_num):
            result = self.exp_queues[i].get()
            results.extend(result)
        local_obs, rewards, dones, _ = map(np.array, zip(*results))
        if dones.all():
            local_obs = self._reset()
        local_obs = local_obs.astype(np.float32)
        rewards = rewards.astype(np.float32)
        return local_obs, rewards, dones

    def _reset(self):
        """Inner reset function"""
        s0 = []
        for i in range(self._proc_num):
            self.action_queues[i].put('reset')
        for i in range(self._proc_num):
            s0.extend(self.init_state_queues[i].get())
        s0 = np.array(s0, np.float32)
        return s0

    def _space_adapter(self, gym_space, batch_shape=None):
        """Inner space adapter"""
        shape = gym_space.shape
        # The dtype get from gym.space is np.int64, but step() accept np.int32 actually.
        dtype = np.int32 if gym_space.dtype.type == np.int64 else gym_space.dtype.type
        if isinstance(gym_space, spaces.Discrete):
            return Space(shape, dtype, low=0, high=gym_space.n, batch_shape=batch_shape)

        return Space(shape, dtype, low=gym_space.low, high=gym_space.high, batch_shape=batch_shape)
