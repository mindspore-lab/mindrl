# pylint: disable=protected-access
# Copyright 2021 Huawei Technologies Co., Ltd
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
"""MultiEnvironmentWrapper Class"""

#pylint: disable=W0212
from multiprocessing import Queue
import numpy as np

from mindspore.ops import operations as P
from mindspore_rl.environment.env_process import EnvironmentProcess
from mindspore_rl.environment import Environment


class MultiEnvironmentWrapper(Environment):
    """
    The MultiEnvironmentWrapper is a wrapper for multi environment scenario. User implements
    their single environment class and set the environment number larger than 1 in configuration
    file, framework will automatically invoke this class to create a multi environment class.

    Args:
        env_instance (list[Environment]): A list that contains instance of environment (subclass of Environment).
        num_proc (int): Number of processing uses during interacting with environment. Default: None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> env_params = {'name': 'CartPole-v0'}
        >>> multi_env = [GymEnvironment(env_params), GymEnvironment(env_params)]
        >>> wrapper = MultiEnvironmentWrapper(multi_env)
        >>> print(wrapper)
        MultiEnvironmentWrapper<>
    """

    def __init__(self,
                 env_instance,
                 num_proc=None):
        super().__init__()
        self._nums = len(env_instance)
        self._envs = env_instance
        self.num_proc = num_proc
        batch_shape = (self._nums,)

        obs_type = self._envs[0].observation_space.ms_dtype
        action_type = self._envs[0].action_space.ms_dtype
        reward_type = self._envs[0].reward_space.ms_dtype
        done_type = self._envs[0].done_space.ms_dtype

        obs_shape = batch_shape + self._envs[0].observation_space.shape
        action_shape = batch_shape + self._envs[0].action_space.shape
        reward_shape = batch_shape + self._envs[0].reward_space.shape
        done_shape = batch_shape + self._envs[0].done_space.shape

        self._step_op = P.PyFunc(self._step,
                                 [action_type,],
                                 [action_shape,],
                                 [obs_type, reward_type, done_type],
                                 [obs_shape, reward_shape, done_shape])
        self._reset_op = P.PyFunc(self._reset, [], [],
                                  [obs_type,],
                                  [obs_shape,])

        self.mpe_env_procs = []
        if self.num_proc != 1:
            self.action_queues = []
            self.exp_queues = []
            self.init_state_queues = []

            if self._nums < self.num_proc:
                raise ValueError("Environment number can not be smaller than process number.")

            avg_env_num_per_proc = int(self._nums / self.num_proc)
            for i in range(self.num_proc):
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

                env_proc = EnvironmentProcess(i, env_num, self._envs[env_num * i:env_num * (i+1)],
                                              action_q, exp_q, init_state_q)
                self.mpe_env_procs.append(env_proc)
                env_proc.start()

    @property
    def observation_space(self):
        """
        Get the state space of the environment.

        Returns:
            A tuple which states for the space of state.
        """

        return self._envs[0].observation_space

    @property
    def action_space(self):
        """
        Get the action space of the environment.

        Returns:
            A tuple which states for the space of action.
        """

        return self._envs[0].action_space

    @property
    def reward_space(self):
        """
        Get the reward space of the environment.

        Returns:
            A tuple which states for the space of reward.
        """
        return self._envs[0].reward_space

    @property
    def done_space(self):
        """
        Get the done space of the environment.

        Returns:
            A tuple which states for the space of done.
        """
        return self._envs[0].done_space

    @property
    def config(self):
        """
        Get the config of environment.

        Returns:
            A dictionary which contains environment's info.
        """
        return self._envs[0].config

    def render(self):
        """
        Render the game. Only support on PyNative mode.
        """
        try:
            self._envs[0].render()
        except:
            raise RuntimeError("Failed to render, run in PyNative mode and comment the ms_function.")

    def reset(self):
        """
        Reset the environment to the initial state. It is always used at the beginning of each
        episode. It will return the value of initial state of each environment.

        Returns:
            A list of tensor which states for all the initial states of each environment.

        """

        return self._reset_op()[0]

    def step(self, action):
        """
        Execute the environment step, which means that interact with environment once.

        Args:
            action (Tensor): A tensor that contains the action information.

        Returns:
            - state (list(Tensor)), a list of environment state after performing the action.
            - reward (list(Tensor)), a list of reward after performing the action.
            - done (list(Tensor)), whether the simulations of each environment finishes or not.
        """

        return self._step_op(action)

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

    def _reset(self):
        """
        The python(can not be interpreted by mindspore interpreter) code of resetting the
        environment. It is the main body of reset function. Due to Pyfunc, we need to
        capsule python code into a function.

        Returns:
            A list of numpy array which states for the initial state of each environment.
        """
        if self.num_proc != 1:
            s0 = []
            for i in range(self.num_proc):
                self.action_queues[i].put('reset')
            for j in range(self.num_proc):
                s0.extend(self.init_state_queues[j].get())
        else:
            s0 = [env._reset() for env in self._envs]
        return s0

    def _step(self, actions):
        """
        The python(can not be interpreted by mindspore interpreter) code of interacting with the
        environment. It is the main body of step function. Due to Pyfunc, we need to
        capsule python code into a function.

        Args:
            actions(List[numpy.dtype]): The action which is calculated by policy net.
            It could be List[int] or List[float] or other else, according to different environment.

        Returns:
            - s1 (List[numpy.array]), a list of environment state after performing the action.
            - r1 (List[numpy.array]), a list of reward after performing the action.
            - done (List[boolean]), whether the simulations of each environment finishes or not.
        """
        results = []
        if self.num_proc != 1:
            accum_env_num = 0
            for i in range(self.num_proc):
                env_num = self.mpe_env_procs[i].env_num
                self.action_queues[i].put(actions[accum_env_num: accum_env_num+env_num,])
                accum_env_num += env_num
            for j in range(self.num_proc):
                exp = self.exp_queues[j].get()
                results.extend(exp)
        else:
            for i in range(self._nums):
                exp = self._envs[i]._step(actions[i])
                results.append(exp)
        obs, rewards, dones = map(np.array, zip(*results))
        return obs, rewards, dones
