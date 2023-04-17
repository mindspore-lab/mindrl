# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
The PettingZooMPEEnvironment base class.
"""

from gym import spaces
import numpy as np
from mindspore.ops import operations as P
from mindspore_rl.environment.environment import Environment
from mindspore_rl.environment.space import Space


class PettingZooMPEEnvironment(Environment):
    """
    The PettingZooMPEEnvironment class is a wrapper that encapsulates
    `PettingZoo <https://pettingzoo.farama.org/environments/mpe/>`_ to
    provide the ability to interact with PettingZoo environments in MindSpore Graph Mode.

    Args:
        params (dict): A dictionary contains all the parameters which are used in this class.

            +------------------------------+-------------------------------+
            |  Configuration Parameters    |  Notices                      |
            +==============================+===============================+
            |  scenario_name               |  the name of game             |
            +------------------------------+-------------------------------+
            |  num                         |  Number of Environment        |
            +------------------------------+-------------------------------+
            |  continuous_actions          |  type of actions space        |
            +------------------------------+-------------------------------+
        env_id (int, optional): A integer which is used to set the seed of this environment,
            default value means the 0th environment. Default: 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> env_params = {'name': 'simple_spread', 'num': 3, 'continuous_actions': False}
        >>> environment = PettingZooMPEEnvironment(env_params)
        >>> print(environment)
        PettingZooMPEEnvironment<>
    """

    def __init__(self, params, env_id=0):
        super(PettingZooMPEEnvironment, self).__init__()
        try:
            import pettingzoo.mpe as mpe
        except ImportError as error:
            raise ImportError(
                "pettingzoo[mpe] is not installed.\n"
                "please pip install pettingzoo[mpe]==1.17.0"
            ) from error
        self.params = params
        self._name = params.get('name')
        self._num = params.get('num')
        self._continuous_actions = params.get('continuous_actions')
        self._seed = params.get('seed') + env_id * 1000
        supported_env_list = ['simple_spread']
        assert self._name in supported_env_list, 'Env {} not supported, choose from {}'.format(
            self._name, supported_env_list)
        if self._name == 'simple_spread':
            self._env = mpe.simple_spread_v2.parallel_env(N=self._num, local_ratio=0, max_cycles=25,
                                                          continuous_actions=self._continuous_actions)
        else:
            pass
        ## reset the environment
        self._env.reset()

        self.agent_name = list(self._env.observation_spaces.keys())
        self._observation_space = self._space_adapter(
            self._env.observation_spaces['agent_0'], batch_shape=(self._num,))
        self._action_space = self._space_adapter(self._env.action_spaces['agent_0'], batch_shape=(self._num,))
        self._reward_space = Space((1,), np.float32, batch_shape=(self._num,))
        self._done_space = Space((1,), np.bool_, low=0, high=2, batch_shape=(self._num,))

        # reset op
        reset_input_type = []
        reset_input_shape = []
        reset_output_type = [self._observation_space.ms_dtype,]
        reset_output_shape = [self._observation_space.shape,]
        self._reset_op = P.PyFunc(self._reset, reset_input_type,
                                  reset_input_shape, reset_output_type, reset_output_shape)

        # step op
        step_input_type = (self._action_space.ms_dtype,)
        step_input_shape = (self._action_space.shape,)
        step_output_type = (self.observation_space.ms_dtype,
                            self._reward_space.ms_dtype, self._done_space.ms_dtype)
        step_output_shape = (self._observation_space.shape,
                             self._reward_space.shape, self._done_space.shape)
        self._step_op = P.PyFunc(
            self._step, step_input_type, step_input_shape, step_output_type, step_output_shape)
        self.action_dtype = self._action_space.ms_dtype
        self.cast = P.Cast()

    @property
    def observation_space(self):
        """
        Get the state space of the environment.

        Returns:
            The state space of environment.
        """

        return self._observation_space

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
        return {}

    def close(self):
        r"""
        Close the environment to release the resource.

        Returns:
            Success(np.bool\_), Whether shutdown the process or threading successfully.
        """
        self._env.close()
        return True

    def render(self):
        """
        Render the game. Only support on PyNative mode.
        """
        try:
            self._env.render()
        except:
            raise RuntimeError("Failed to render, run in PyNative mode and comment the ms_function.")

    def reset(self):
        """
        Reset the environment to the initial state. It is always used at the beginning of each
        episode. It will return the value of initial state.

        Returns:
            A tensor which states for the initial state of environment.

        """

        return self._reset_op()[0]

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

        # Add cast ops for mixed precision case. Redundant cast ops will be eliminated automatically.
        action = self.cast(action, self.action_dtype)
        return self._step_op(action)

    def _reset(self):
        """
        The python(can not be interpreted by mindspore interpreter) code of resetting the
        environment. It is the main body of reset function. Due to Pyfunc, we need to
        capsule python code into a function.

        Returns:
            A numpy array which states for the initial state of environments.
        """

        s0 = self._env.reset()
        s0 = np.array(np.vstack(list(s0.values()))).astype(np.float32)
        return s0

    def _step(self, action):
        """
        The python(can not be interpreted by mindspore interpreter) code of interacting with the
        environment. It is the main body of step function. Due to Pyfunc, we need to
        capsule python code into a function.

        Args:
            action(int or float): The action which is calculated by policy net.
            It could be integer or float, according to different environment.

        Returns:
            - obs (numpy.array), the environment state after performing the actions.
            - reward (numpy.array), the reward after performing the actions.
            - done (boolean), whether the simulation finishes or not.
        """
        action_dict = dict()
        for i, act in enumerate(action):
            agent = self.agent_name[i]
            if self._continuous_actions:
                assert np.all(((act <= 1.0 + 1e-4), (act >= -1.0 - 1e-4))), \
                    'action should in range [-1, 1], but got {}'.format(act)
                low, high = self._action_space.boundary
                a = np.clip(act, low, high)
                action_dict[agent] = a
            else:
                action_dict[agent] = np.argmax(act)
        obs, reward, done, _ = self._env.step(action_dict)
        out_obs = np.array(np.vstack(list(obs.values()))).astype(np.float32)
        out_reward = np.array(np.vstack(list(reward.values()))).astype(np.float32)
        out_done = np.array(np.vstack(list(done.values()))).astype(np.bool_)
        return out_obs, out_reward, out_done

    def _space_adapter(self, mpe_space, batch_shape=None):
        """Transfer pettingzoo mpe dtype to the dtype that is suitable for MindSpore"""
        shape = mpe_space.shape
        mpe_type = mpe_space.dtype.type
        if mpe_type == np.int64:
            dtype = np.float32
        elif mpe_type == np.float64:
            dtype = np.float32
        else:
            dtype = mpe_type

        if isinstance(mpe_space, spaces.Discrete):
            return Space((mpe_space.n,), dtype, batch_shape=batch_shape)

        return Space(shape, dtype, batch_shape=batch_shape)
