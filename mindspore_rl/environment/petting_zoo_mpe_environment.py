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
# pylint:disable=W0707
# pylint:disable=C0415
import numpy as np

from mindspore_rl.environment.python_environment import PythonEnvironment
from mindspore_rl.environment.space import Space
from mindspore_rl.environment.space_adapter import gym2ms_adapter


class PettingZooMPEEnvironment(PythonEnvironment):
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
        try:
            from pettingzoo import mpe
        except ImportError as error:
            raise ImportError(
                "pettingzoo[mpe] is not installed.\n"
                "please pip install pettingzoo[mpe]==1.17.0"
            ) from error
        self.params = params
        self._name = params.get("name")
        self._num = params.get("num")
        self._continuous_actions = params.get("continuous_actions")
        self._seed = params.get("seed") + env_id * 1000
        supported_env_list = ["simple_spread"]
        assert (
            self._name in supported_env_list
        ), f"Env {self._name} not supported, choose from {supported_env_list}"
        if self._name == "simple_spread":
            self._env = mpe.simple_spread_v2.parallel_env(
                N=self._num,
                local_ratio=0,
                max_cycles=25,
                continuous_actions=self._continuous_actions,
            )
        else:
            pass
        # reset the environment
        self._env.reset()
        self.agent_name = list(self._env.observation_spaces.keys())
        observation_space = gym2ms_adapter(list(self._env.observation_spaces.values()))
        env_action_space = self._env.action_spaces["agent_0"]
        action_space = Space(
            (env_action_space.n,), env_action_space.dtype.type, batch_shape=(self._num,)
        )
        reward_space = Space((1,), np.float32, batch_shape=(self._num,))
        done_space = Space((1,), np.bool_, low=0, high=2, batch_shape=(self._num,))

        super().__init__(action_space, observation_space, reward_space, done_space)

    def close(self):
        r"""
        Close the environment to release the resource.

        Returns:
            Success(np.bool\_), Whether shutdown the process or threading successfully.
        """
        self._env.close()
        return True

    def _render(self):
        """
        Render the game. Only support on PyNative mode.
        """
        try:
            self._env.render()
        except BaseException:
            raise RuntimeError(
                "Failed to render, run in PyNative mode and comment the ms.jit."
            )

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
        action_dict = {}
        for i, act in enumerate(action):
            agent = self.agent_name[i]
            if self._continuous_actions:
                assert np.all(
                    ((act <= 1.0 + 1e-4), (act >= -1.0 - 1e-4))
                ), f"action should in range [-1, 1], but got {act}"
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

    def _set_seed(self, seed_value: int) -> bool:
        """Inner set seed"""
        raise ValueError(
            "PettingZooMPEEnvironment does not support set seed. Please pass seed through params"
        )
