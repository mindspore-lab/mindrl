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
"""test random environment"""
import numpy as np
import pytest

from mindspore_rl.environment.random_environment import RandomEnvironment


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_random_environment():
    """
    Feature: Random Environment generate random values for step and reset function.
    Description: Random Environment generate random values for step and reset function.
    Expectation: success.
    """

    params = {
        "reset_output_shape": [
            (4,),
        ],
        "reset_output_type": [
            np.float32,
        ],
        "step_input_shape": [
            (),
        ],
        "step_input_type": [
            np.int32,
        ],
        "step_output_shape": [(4,), (), ()],
        "step_output_type": [
            np.float32,
            np.float32,
            np.bool_,
        ],
    }

    random_env = RandomEnvironment(params)
    state = random_env.reset()
    assert state.shape == (4,)
    assert state.dtype == np.float32
    action = random_env.action_space.sample()
    assert action.shape == ()
    assert action.dtype == np.int32
    new_state, reward, done = random_env.step(action)
    assert new_state.shape == (4,)
    assert reward.shape == ()
    assert done.shape == ()
    assert new_state.dtype == np.float32
    assert reward.dtype == np.float32
    assert done.dtype == np.bool_
