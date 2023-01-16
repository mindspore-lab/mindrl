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
The MindSpore reinforcement learning built-in environment class.
"""

from mindspore_rl.environment.environment import Environment
from mindspore_rl.environment.registration import Registry


registry = Registry(domain="MindSpore")


def ms_register(name, env_class):
    '''register an environment to MindSpore domain'''
    return registry.register(name, env_class)


def ms_create(kwargs):
    '''create a MindSpore domain environment instance'''
    return registry.create(**kwargs)


class MsEnvironment(Environment):
    r"""
    Class encapsulates built-in environment.

    Args:
        kwargs (dict): The dictionary of environment specific configurations. See below table for details:

            +--------------------+------------------------------+------------------+----------------------------+
            | Environment name   |  Configuration Parameters    |   Default value  |  Notices                   |
            +====================+==============================+==================+============================+
            | Tag                |  seed                        |   42             |  random seed               |
            |                    +------------------------------+------------------+----------------------------+
            |                    |  environment_num             |   2              |  number of environments    |
            |                    +------------------------------+------------------+----------------------------+
            |                    |  predator_num                |   10             |  number of predators       |
            |                    +------------------------------+------------------+----------------------------+
            |                    |  max_timestep                |   100            |  max timestep per episode  |
            |                    +------------------------------+------------------+----------------------------+
            |                    |  map_length                  |   100            |  length of map             |
            |                    +------------------------------+------------------+----------------------------+
            |                    |  map_width                   |   100            |  width of map              |
            |                    +------------------------------+------------------+----------------------------+
            |                    |  wall_hit_penalty            |   0.1            |  agent wall hit penalty    |
            |                    +------------------------------+------------------+----------------------------+
            |                    |  catch_reward                |   10             |  predator catch reward     |
            |                    +------------------------------+------------------+----------------------------+
            |                    |  caught_penalty              |   5              |  prey caught penalty       |
            |                    +------------------------------+------------------+----------------------------+
            |                    |  step_cost                   |   0.01           |  step cost                 |
            +--------------------+------------------------------+------------------+----------------------------+

    Supported Platforms:
        ``GPU``

    Examples:
        >>> config = {'name': 'Tag', 'predator_num': 4}
        >>> env = MsEnvironment(config)
        >>> observation = env.reset()
        >>> action = Tensor(env.action_space.sample())
        >>> observation, reward, done = env.step(action)
        >>> print(observation.shape)
        (2, 5, 21)
    """

    def __init__(self, kwargs=None):
        super(MsEnvironment, self).__init__()
        self.env = ms_create(kwargs)

    def reset(self):
        r"""
        Reset the environment to initial observation and return the initial observation.

        Inputs:
            No inputs.

        Returns:
            Tensor, the initial observation.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> config = {'name': 'Tag', 'predator_num': 4}
            >>> env = MsEnvironment(config)
            >>> observation = env.reset()
            >>> print(observation.shape)
            (2, 5, 21)
        """
        return self.env.reset()

    def step(self, action):
        r"""
        Run one timestep of environment to interact with environment.

        Args:
            action (Tensor): Action provided by the all of agents.

        Returns:
            Tuple of 3 tensors, the observation, the reward and the done.

            - **observation** (Tensor) - Observations of all agents after action.
            - **reward** (Tensor) - Amount of reward returned by the environment.
            - **done** (Tensor) - Whether the episode has ended.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> config = {'name': 'Tag', 'predator_num': 4}
            >>> env = MsEnvironment(config)
            >>> observation = env.reset()
            >>> action = Tensor(env.action_space.sample())
            >>> observation, reward, done = env.step(action)
            >>> print(observation.shape)
            (2, 5, 21)
        """
        return self.env.step(action)

    @property
    def action_space(self):
        r"""
        Get the valid action space of the environment.

        Returns:
            The action space of environment.
        """
        return self.env.action_space

    @property
    def observation_space(self):
        r"""
        Get the valid observation space of the environment.

        Returns:
            The state space of environment.
        """
        return self.env.observation_space

    @property
    def reward_space(self):
        r"""
        Get the valid reward space of the environment.

        Returns:
            The reward space of environment.
        """
        return self.env.reward_space

    @property
    def done_space(self):
        r"""
        Get the valid done space of the environment.

        Returns:
            The done space of environment.
        """
        return self.env.done_space

    @property
    def config(self):
        r"""
        Get environment configuration.

        Returns:
            The configuration of environment.
        """
        return self.env.config
