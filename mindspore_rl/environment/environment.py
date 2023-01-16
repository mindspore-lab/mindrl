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
The environment base class.
"""

import mindspore.nn as nn


class Environment(nn.Cell):
    r"""
    The virtual base class for the environment. This class should be overridden before calling in the model.
    """

    def __init__(self):
        super(Environment, self).__init__(auto_prefix=False)

    @property
    def action_space(self):
        """
        Get the action space of the environment.

        Returns:
            The action space of environment.
        """
        raise NotImplementedError("Method should be overridden by subclass.")

    @property
    def observation_space(self):
        """
        Get the state space of the environment.

        Returns:
            The state space of environment.
        """
        raise NotImplementedError("Method should be overridden by subclass.")

    @property
    def reward_space(self):
        """
        Get the reward space of the environment.

        Returns:
            The reward space of environment.
        """
        raise NotImplementedError("Method should be overridden by subclass.")

    @property
    def done_space(self):
        """
        Get the done space of the environment.

        Returns:
            The done space of environment.
        """
        raise NotImplementedError("Method should be overridden by subclass.")

    @property
    def config(self):
        """
        Get the config of environment.

        Returns:
            A dictionary which contains environment's info.
        """
        raise NotImplementedError("Method should be overridden by subclass.")

    def reset(self):
        """
        Reset the environment to the initial state. It is always used at the beginning of each
        episode. It will return the value of initial state or other initial information.

        Returns:
            A tensor which states for the initial state of environment or a tuple contains
            initial information, such as new state, action, reward, etc.

        """
        raise NotImplementedError("Method should be overridden by subclass.")

    def step(self, action):
        r"""
        Execute the environment step, which means that interact with environment once.

        Args:
            action (Tensor): A tensor that contains the action information.

        Returns:
            A tuple of Tensor which contains information after interacting with environment.
        """
        raise NotImplementedError("Method should be overridden by subclass.")

    def close(self):
        r"""
        Close the environment to release the resource.

        Returns:
            Success(np.bool\_), Whether shutdown the process or threading successfully.
        """
        return True
