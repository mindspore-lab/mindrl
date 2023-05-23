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
"""Action Repeat Wrapper"""
import numpy as np

from mindspore_rl.environment.wrapper import Wrapper


class ActionRepeatWrapper(Wrapper):
    """
    This wrapper provides the ability to repeat a same action for multiple steps. According to the input value repeat
    num. ActionRepeatWrapper does not support asynchronous environment.

    Args:
        env_creator (Environment): The environment creator.
        repeat_num (int): The number of steps to repeat the same action. Default: 1.
    """

    def __init__(self, env_creator, repeat_num=1):
        super().__init__(env_creator)
        if repeat_num < 1:
            raise ValueError(f"Repeat num must be greater than 0, but got {repeat_num}")
        self._repeat_num = repeat_num

    def _step(self, action: np.ndarray):
        """Inner step function"""
        done = False
        total_reward = 0
        i = 0
        while i < self._repeat_num and not done:
            step_out = self.environment.step(action)
            total_reward += step_out[1]
            done = step_out[2]
            i += 1
        step_out = (
            step_out[0],
            np.array(total_reward, self.reward_space.np_dtype),
            *step_out[2:],
        )
        return step_out

    def _reset(self):
        """Inner reset function"""
        return self.environment.reset()

    def _send(self, action: np.ndarray, env_id: np.ndarray):
        """Inner send function"""
        raise ValueError("ActionRepeatWrapper does not support send method yet")

    def _recv(self):
        """Inner recv function"""
        raise ValueError("ActionRepeatWrapper does not support recv method yet")
