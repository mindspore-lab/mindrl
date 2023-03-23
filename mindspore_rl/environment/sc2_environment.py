# Copyright 2022 Huawei Technologies Co., Ltd
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
The starcraft 2 environment.
"""

#pylint: disable=C0111
import importlib
import numpy as np
import mindspore as ms
from mindspore.ops import operations as P
from mindspore_rl.environment.environment import Environment
from mindspore_rl.environment.space import Space


class StarCraft2Environment(Environment):
    """
    StarCraft2Environment is a wrapper of SMAC. SMAC is WhiRL's environment for research in the
    field of collaborative multi-agent reinforcement learning (MARL) based on Blizzard's
    StarCraft II RTS game. SMAC makes use of Blizzard's StarCraft II Machine Learning API and
    DeepMind's PySC2 to provide a convenient interface for autonomous agents to interact with
    StarCraft II, getting observations and performing actions. More detail please have a look
    at the official github of SMAC: https://github.com/oxwhirl/smac.

    Args:
        params (dict): A dictionary contains all the parameters which are used in this class.

            +------------------------------+--------------------------------------------------------+
            |  Configuration Parameters    |  Notices                                               |
            +==============================+========================================================+
            |  sc2_args                    |  a dict which contains key value that is used to create|
            |                              |  instance of SMAC, such as map_name. For more detail   |
            |                              |  please have a look at its official github.            |
            +------------------------------+--------------------------------------------------------+
        env_id (int, optional): A integer which is used to set the seed of this environment,
            default value means the 0th environment. Default: 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> env_params = {'sc2_args': {'map_name': '2s3z'}}
        >>> environment = StarCraft2Environment(env_params, 0)
        >>> print(environment)
    """

    def __init__(self, params, env_id=0):
        super().__init__()
        sc2_args = params['sc2_args']
        if sc2_args.get('seed'):
            sc2_args['seed'] = sc2_args['seed'] + env_id * 1000
        sc2_creator = importlib.import_module("smac.env")
        self._env = sc2_creator.StarCraft2Env(**sc2_args)
        self.env_info = self._env.get_env_info()
        self._num_agent = self.env_info['n_agents']
        action_dim = self.env_info['n_actions']
        obs_dim = self.env_info['obs_shape']
        self._global_obs_dim = self.env_info['state_shape']

        self.step_info = {}

        self._observation_space = Space(
            (obs_dim,), np.float32, batch_shape=(self._num_agent,))
        self._action_space = Space(
            (1,), np.int32, low=0, high=action_dim, batch_shape=(self._num_agent,))
        self._reward_space = Space((1,), np.float32)
        self._done_space = Space((1,), np.bool_, low=0, high=2)

        # reset op
        reset_input_type = []
        reset_input_shape = []
        reset_output_type = (self._observation_space.ms_dtype, self._observation_space.ms_dtype,
                             self._action_space.ms_dtype)
        reset_output_shape = ((self._num_agent, obs_dim), (self._global_obs_dim,),
                              (self._num_agent, self._action_space.num_values))
        self._reset_op = P.PyFunc(self._reset, reset_input_type, reset_input_shape,
                                  reset_output_type, reset_output_shape)

        # step op
        step_input_type = (self._action_space.ms_dtype,)
        step_input_shape = (self._action_space.shape,)
        step_output_type = (self._observation_space.ms_dtype, self._reward_space.ms_dtype,
                            self._done_space.ms_dtype, self._observation_space.ms_dtype,
                            self._action_space.ms_dtype)
        step_output_shape = ((self._num_agent, obs_dim), self._reward_space.shape,
                             self._done_space.shape, (self._global_obs_dim,),
                             (self._num_agent, self._action_space.num_values))
        self._step_op = P.PyFunc(self._step, step_input_type, step_input_shape,
                                 step_output_type, step_output_shape)

        self._get_step_info_ops = P.PyFunc(self._get_step_info, (), (),
                                           (ms.int32, ms.int32, ms.int32),
                                           ((1,), (1,), (1,)))

    def reset(self):
        """
        Reset the environment to the initial state. It is always used at the beginning of each
        episode. It will return the value of initial state, the global observation and
        an new available action.

        Returns:
            A tuple of Tensor contains initial state, global observation and available actions.

        """
        return self._reset_op()

    def step(self, action):
        r"""
        Execute the environment step, which means that interact with environment once.

        Args:
            action (Tensor): A tensor that contains the action information.

        Returns:
            - state (Tensor), the environment state after performing the action.
            - reward (Tensor), the reward after performing the action.
            - done (mindspore.bool\_), whether the simulation finishes or not.
            - global\_obs, the global observation of this environment.
            - avail\_actions, the available actions in this state.
        """
        return self._step_op(action)

    def get_step_info(self):
        r"""
        Get the information after interacting with environment.

        Returns:
            - battle\_won, whether this game is won or not.
            - dead\_allies, how many allies are dead.
            - dead\_enemies, how many enemies are dead.
        """
        return self._get_step_info_ops()

    def close(self):
        r"""
        Close the environment to release the resource.

        Returns:
            Success(np.bool\_), Whether shutdown the process or threading successfully.
        """
        self._env.close()
        return True

    def _step(self, action):
        reward, done, self.step_info = self._env.step(action)
        new_state = np.array(self._env.get_obs(),
                             self._observation_space.np_dtype)
        reward = np.array([reward], self._reward_space.np_dtype)
        done = np.array([done])
        global_obs = self._env.get_state()
        avail_actions = np.array(
            self._env.get_avail_actions(), self._action_space.np_dtype)

        return new_state, reward, done, global_obs, avail_actions

    def _reset(self):
        local_obs, global_obs = np.array(self._env.reset())
        avail_actions = np.array(
            self._env.get_avail_actions(), self._action_space.np_dtype)
        return local_obs, global_obs, avail_actions

    def _get_step_info(self):
        if self.step_info:
            battle_won = np.array(self.step_info['battle_won'], np.int32)
            dead_allies = np.array(self.step_info['dead_allies'], np.int32)
            dead_enemies = np.array(self.step_info['dead_enemies'], np.int32)
        else:
            battle_won = np.array(0, np.int32)
            dead_allies = np.array(5, np.int32)
            dead_enemies = np.array(0, np.int32)
        return battle_won, dead_allies, dead_enemies

    @property
    def action_space(self):
        """
        Get the action space of the environment.

        Returns:
            A tuple which states for the space of observation.
        """
        return self._action_space

    @property
    def observation_space(self):
        """
        Get the state space of the environment.

        Returns:
            A tuple which states for the space of state.
        """
        return self._observation_space

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
        return {"global_observation_dim": self._global_obs_dim,
                "episode_limit": self.env_info['episode_limit'],
                "num_agent": self._num_agent}
