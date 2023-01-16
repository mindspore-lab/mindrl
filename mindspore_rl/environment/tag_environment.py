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
The environment registration class.
"""

import numpy as np
from mindspore_rl.environment.space import Space
from mindspore_rl.environment.environment import Environment
from mindspore.ops.operations._rl_inner_ops import EnvCreate, EnvReset, EnvStep


class TagEnvironment(Environment):
    """The tag environment."""
    def __init__(self, **kwargs):
        super(TagEnvironment, self).__init__()
        # default config
        self._config = {"seed": 42,
                        "environment_num": 2,
                        "predator_num": 10,
                        "max_timestep": 100,
                        "map_length": 100,
                        "map_width": 100,
                        "wall_hit_penalty": 0.1,
                        "catch_reward": 10.0,
                        "caught_penalty": 5.0,
                        "step_cost": 0.01,
                        "partially_observation": False}

        # Update default config
        for key in kwargs:
            if key not in self._config:
                raise 'key{} not supported.'.format(key)
            self._config[key] = kwargs[key]

        # Create environment instance.
        handle = EnvCreate('Tag', **self.config)().asnumpy().item(0)

        environment_num = self._config['environment_num']
        agent_num = self._config['predator_num'] + 1

        # Action space.
        batch_shape = (environment_num, agent_num)
        self._action_space = Space((), np.int32, low=0, high=5, batch_shape=batch_shape)

        # Observation space.
        obs_per_agent = (6,) if self._config['partially_observation'] else (4 * agent_num + 1,)
        self._observation_space = Space(obs_per_agent, np.float32, low=0, high=1, batch_shape=batch_shape)

        # Reward space.
        self._reward_space = Space((), np.float32, low=0, high=1, batch_shape=batch_shape)

        # Done space
        self._done_space = Space((), np.int32, low=0, high=2, batch_shape=(environment_num,))

        # Environment Operators.
        self._reset_op = EnvReset(handle, self._observation_space.shape, self._observation_space.ms_dtype)
        self._step_op = EnvStep(handle, self._observation_space.shape, self._observation_space.ms_dtype,
                                self._reward_space.shape, self._reward_space.ms_dtype)

    def reset(self):
        return self._reset_op()

    def step(self, action):
        return self._step_op(action)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def reward_space(self):
        return self._reward_space

    @property
    def done_space(self):
        return self._done_space

    @property
    def config(self):
        return self._config
