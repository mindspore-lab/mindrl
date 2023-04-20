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
"""Random Environment """

import numpy as np

from mindspore_rl.environment.python_environment import PythonEnvironment
from mindspore_rl.environment.space import Space


class RandomEnvironment(PythonEnvironment):
    """
    This is a random environment that returns random values for step and reset. It is used for mocking the
    environment, and reduce the environment cost in the algorithm.
    """

    def __init__(self, params):
        self._step_input_shape = params.get("step_input_shape")
        self._step_input_type = params.get("step_input_type")
        self._step_output_shape = params.get("step_output_shape")
        self._step_output_type = params.get("step_output_type")
        self._reset_output_shape = params.get("reset_output_shape")
        self._reset_output_type = params.get("reset_output_type")
        seed_value = params.get("seed")
        self._seed = np.random.RandomState(seed=seed_value)

        action_space = Space(self._step_input_shape[0], self._step_input_type[0])
        observation_space = Space(
            self._reset_output_shape[0], self._reset_output_type[0]
        )
        super().__init__(action_space, observation_space)

    def _reset(self):
        reset_out = []
        for i, shape in enumerate(self._reset_output_shape):
            reset_out.append(
                self._seed.random(shape).astype(self._reset_output_type[i])
            )
        if len(self._reset_output_shape) > 1:
            return tuple(reset_out)
        return reset_out[0]

    def _step(self, action: np.ndarray):
        step_out = []
        for i, shape in enumerate(self._step_output_shape):
            step_out.append(self._seed.random(shape).astype(self._step_output_type[i]))
        return tuple(step_out)

    def _render(self) -> np.ndarray:
        raise ValueError("RandomEnvironment does not support render.")

    def _set_seed(self, seed_value: int) -> bool:
        raise ValueError(
            "RandomEnvironment does not support set_seed. Please pass the seed to params"
        )
