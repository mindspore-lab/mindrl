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
"""Action Normalization Wrapper"""
from typing import Callable

import numpy as np

from mindspore_rl.environment.environment import Environment
from mindspore_rl.environment.space import Space
from mindspore_rl.environment.wrapper import Wrapper

EnvCreator = Callable[[], Environment]


class ActionNormWrapper(Wrapper):
    """
    ActionNormaWrapper is a wrapper for normalizing the input action. It will also modify the action space of
    environment.

    Args:
        env_creator (EnvCreator): A callable which creates a new environment instance.
    """

    def __init__(self, env_creator: EnvCreator):
        if not callable(env_creator):
            raise TypeError(
                f"The input env_creators must be a list of callable, but got {env_creator}"
            )
        super().__init__(env_creator)
        low, high = self.environment.action_space.boundary
        self._mask = np.logical_and(np.isfinite(low), np.isfinite(high))
        self._low = np.where(self._mask, low, -1)
        self._high = np.where(self._mask, high, 1)

    @property
    def action_space(self) -> Space:
        origin_action_space = self.environment.action_space
        return Space(
            origin_action_space.shape,
            origin_action_space.np_dtype,
            self._low,
            self._high,
        )

    def _step(self, action):
        """Inner step function"""
        action = np.where(
            self._mask,
            (action + 1) / 2 * (self._high - self._low) + self._low,
            action,
        )
        return self.environment.step(action)

    def _reset(self):
        """Inner reset function"""
        return self.environment.reset()

    def _send(self, action, env_id):
        """Inner send function"""
        action = np.where(
            self._mask,
            (action + 1) / 2 * (self._high - self._low) + self._low,
            action,
        )
        return self.environment.send(action, env_id)

    def _recv(self):
        """Inner recv function"""
        return self.environment.recv()
