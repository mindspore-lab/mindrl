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

# pylint: disable=C0111
import importlib

import numpy as np

from mindspore_rl.environment.python_environment import PythonEnvironment
from mindspore_rl.environment.space import Space


class StarCraft2Environment(PythonEnvironment):
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
        sc2_args = params["sc2_args"]
        if sc2_args.get("seed"):
            sc2_args["seed"] = sc2_args["seed"] + env_id * 1000
        sc2_creator = importlib.import_module("smac.env")
        self._env = sc2_creator.StarCraft2Env(**sc2_args)
        self._env.reset()
        valid_action_mask = self._env.get_avail_actions()
        self._info_key = params.get("info_key", None)

        env_info = self._env.get_env_info()
        num_agent = env_info["n_agents"]
        action_dim = env_info["n_actions"]
        obs_dim = env_info["obs_shape"]

        config = {
            **env_info,
            "global_observation_dim": env_info["state_shape"],
            "num_agent": num_agent,
        }

        self.step_info = {}

        observation_space = Space((obs_dim,), np.float32, batch_shape=(num_agent,))
        action_space = Space(
            (1,),
            np.int32,
            low=0,
            high=action_dim,
            batch_shape=(num_agent,),
            mask=valid_action_mask,
        )
        super().__init__(action_space, observation_space, config=config)

    def close(self):
        r"""
        Close the environment to release the resource.

        Returns:
            Success(np.bool\_), Whether shutdown the process or threading successfully.
        """
        self._env.close()
        return True

    def _step(self, action):
        reward, done, step_info = self._env.step(action)
        info_out = []
        if self._info_key is not None:
            for key in self._info_key:
                info_out.append(np.array(step_info[key]))
        new_state = np.array(self._env.get_obs(), self.observation_space.np_dtype)
        reward = np.array(reward, self.reward_space.np_dtype)
        done = np.array(done)
        global_obs = self._env.get_state()
        avail_actions = np.array(
            self._env.get_avail_actions(), self._action_space.np_dtype
        )
        if len(info_out) > 0:
            step_out = (new_state, reward, done, global_obs, avail_actions, *info_out)
        else:
            step_out = (new_state, reward, done, global_obs, avail_actions)
        return step_out

    def _reset(self):
        local_obs, global_obs = self._env.reset()
        avail_actions = np.array(
            self._env.get_avail_actions(), self._action_space.np_dtype
        )
        return np.array(local_obs), np.array(global_obs), avail_actions

    def _render(self) -> np.ndarray:
        img = self._env.render()
        return img

    def _set_seed(self, seed_value: int) -> bool:
        """Inner set seed"""
        raise ValueError(
            "StarCraft2Environment does not support set seed. Please pass seed through params"
        )
