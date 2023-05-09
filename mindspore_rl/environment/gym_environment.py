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
"""
The GymEnvironment base class.
"""

from operator import itemgetter
from typing import Union

# pylint: disable=W0223
# pylint: disable=W0707
import gym
import numpy as np
from packaging import version

from mindspore_rl.environment.python_environment import PythonEnvironment
from mindspore_rl.environment.space_adapter import gym2ms_adapter

is_old_gym = version.parse(gym.__version__) < version.parse("0.26.0")


class GymEnvironment(PythonEnvironment):
    """
    The GymEnvironment class is a wrapper that encapsulates `Gym <https://gym.openai.com/>`_ to
    provide the ability to interact with Gym environments in MindSpore Graph Mode.

    Args:
        params (dict): A dictionary contains all the parameters which are used in this class.

            +------------------------------+----------------------------+
            |  Configuration Parameters    |  Notices                   |
            +==============================+============================+
            |  name                        |  the name of game in Gym   |
            +------------------------------+----------------------------+
            |  seed                        |  seed used in Gym          |
            +------------------------------+----------------------------+
        env_id (int, optional): A integer which is used to set the seed of this environment,
            default value means the 0th environment. Default: ``0`` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> env_params = {'name': 'CartPole-v0'}
        >>> environment = GymEnvironment(env_params, 0)
        >>> print(environment)
        GymEnvironment<>
    """

    def __init__(self, params, env_id=0):
        # Obtain parameters
        self._name = params["name"]
        self._info_key = params.get("info_key", None)
        self._render_mode = params.get("render_mode", "rgb_array")
        self._render_kwargs = params.get("render_kwargs", {})
        self._need_auto_reset = params.get("need_auto_reset", False)

        # Create environment instance and adapt gym space to mindspore space
        if is_old_gym:
            self._env = gym.make(self._name)
        else:
            self._env = gym.make(self._name, render_mode=self._render_mode)
            self._seed = params.get("seed")
            if self._seed is not None:
                self._seed = self._seed + env_id * 1000
        observation_space = gym2ms_adapter(self._env.observation_space)
        action_space = gym2ms_adapter(self._env.action_space)
        config = self._env.spec.__dict__
        super().__init__(
            action_space,
            observation_space,
            config=config,
            need_auto_reset=self._need_auto_reset,
        )

    def close(self):
        r"""
        Close the environment to release the resource.

        Returns:
            Success(np.bool\_), Whether shutdown the process or threading successfully.
        """
        self._env.close()
        return True

    def _render(self) -> Union[np.ndarray]:
        """
        Render the game. Only support on PyNative mode.
        """
        try:
            if is_old_gym:
                img = self._env.render(self._render_mode, **self._render_kwargs)
            else:
                img = self._env.render(**self._render_kwargs)
        except BaseException:
            raise RuntimeError(
                "Failed to render, run in PyNative mode and comment the ms.jit."
            )
        return img

    def _reset(self):
        """
        The python(can not be interpreted by mindspore interpreter) code of resetting the
        environment. It is the main body of reset function. Due to Pyfunc, we need to
        capsule python code into a function.

        Returns:
            A numpy array which states for the initial state of environment.
        """
        info = {}
        if is_old_gym:
            s0 = self._env.reset()
        else:
            s0, info = self._env.reset(seed=self._seed)
        if (self._info_key is None) or is_old_gym:
            reset_out = s0
        else:
            info_value = map(np.array, itemgetter(*self._info_key)(info))
            reset_out = (s0, *info_value)
        return reset_out

    def _step(self, action):
        """
        The python(can not be interpreted by mindspore interpreter) code of interacting with the
        environment. It is the main body of step function. Due to Pyfunc, we need to
        capsule python code into a function.

        Args:
            action(int or float): The action which is calculated by policy net. It could be integer
            or float, according to different environment.

        Returns:
            - s1 (numpy.array), the environment state after performing the action.
            - r1 (numpy.array), the reward after performing the action.
            - done (boolean), whether the simulation finishes or not.
        """
        if is_old_gym:
            s, r, done, info = self._env.step(action)
        else:
            s, r, term, trunc, info = self._env.step(action)
            done = term or trunc
        r = np.array(r)
        done = np.array(done)
        if self._info_key is None:
            step_out = (s, r, done)
        else:
            info_value = map(np.array, itemgetter(*self._info_key)(info))
            step_out = (s, r, done, *info_value)
        return step_out
