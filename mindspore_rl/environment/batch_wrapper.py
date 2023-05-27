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
"""Sync Parallel Wrapper"""

# pylint:disable=W0106
from typing import Callable, Sequence

import numpy as np
from mindspore import Tensor

from mindspore_rl.environment.environment import Environment
from mindspore_rl.environment.wrapper import Wrapper

EnvCreator = Callable[[], Environment]


class BatchWrapper(Wrapper):
    r"""
    Execute environment synchronously in parallel. The result will be returned when all the environment are
    finished.

    Args:
        env_creators (Sequence[EnvCreator]):  A list of env creator.
    """

    def __init__(
        self,
        env_creators: Sequence[EnvCreator],
    ):
        type_check = [not callable(env_creator) for env_creator in env_creators]
        if any(type_check):
            raise TypeError(
                f"The input env_creators must be a list of callable, but got {env_creators}"
            )
        super().__init__(env_creators)

    def _reset(self):
        """
        Reset the environment to the initial state. It is always used at the beginning of each
        episode. It will return the value of initial state or other initial information.

        Returns:
            - state (Union[np.ndarray, Tensor]), A numpy array or Tensor which states for
                the initial state of environment.
            - args (Union[np.ndarray, Tensor], optional), Support arbitrary outputs, but user needs to ensure the
                dtype. This output is optional.
        """
        reset_out = []
        for env in self.environment:
            reset_out.append(env.reset())
        if self._num_reset_out == 1:
            reset_out = np.array(reset_out)
        else:
            s0, *others = map(np.array, zip(*reset_out))
            reset_out = (s0, *others)
        return reset_out

    def _step(self, action):
        r"""
        Execute the environment step, which means that interact with environment once.

        Args:
            action (Union[Tensor, np.ndarray]): A tensor that contains the action information.

        Returns:
            - state (Union[np.ndarray, Tensor]), The environment state after performing the action.
            - reward (Union[np.ndarray, Tensor]), The reward after performing the action.
            - done (Union[np.ndarray, Tensor]), Whether the simulation finishes or not.
            - args (Union[np.ndarray, Tensor], optional), Support arbitrary outputs, but user needs to ensure the
                dtype. This output is optional.
        """

        step_out = []
        for i, env in enumerate(self.environment):
            step_out.append(env.step(action[i]))
        obs, rewards, dones, *others = map(np.array, zip(*step_out))
        step_out = (obs, rewards, dones, *others)
        return step_out

    def _send(self, action: Tensor, env_id: Tensor):
        r"""
        Execute the environment step asynchronously. User can obtain result by using recv.

        Args:
            action (Tensor): A tensor or array that contains the action information.
            env_id (Tensor): Which environment these actions will interact with.

        Returns:
            Success (Tensor): True if the action is successfully executed, otherwise False.
        """
        raise ValueError("BatchWrapper does not support send yet")

    def _recv(self):
        r"""
        Receive the result of interacting with environment.

        Returns:
            - state (Tensor), The environment state after performing the action.
            - reward (Tensor), The reward after performing the action.
            - done (Tensor), whether the simulation finishes or not.
            - env_id (Tensor), Which environments are interacted.env
            - args (Tensor, optional), Support arbitrary outputs, but user needs to ensure the
                dtype. This output is optional.
        """
        raise ValueError("BatchWrapper does not support recv yet")
