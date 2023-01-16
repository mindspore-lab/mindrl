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
'''
Test case for Space.
'''

import pytest
import numpy as np
from mindspore_rl.environment import Space
from mindspore.common import dtype as mstype

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_discrete_space():
    '''
    Feature: Test space class.
    Description: Discrete space
    Expectation: success.
    '''

    low, high = 2, 5
    action_space = Space(feature_shape=(6,), low=low, high=high, dtype=np.int32, batch_shape=(10,))
    assert action_space.is_discrete
    assert action_space.shape == (10, 6)
    assert action_space.np_dtype == np.int32
    assert action_space.ms_dtype == mstype.int32
    assert action_space.num_values == (5 - 2) ** 6
    sample = action_space.sample()
    assert sample.shape, action_space.shape
    assert np.max(sample) < high
    assert np.min(sample) >= low
    print(action_space)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_continuous_space():
    '''
    Feature: Test space class.
    Description: Continuous space
    Expectation: success.
    '''

    low, high = 0., 10.
    action_space = Space(feature_shape=(6,), dtype=np.float32, low=low, high=high, batch_shape=(20, 5))
    assert not action_space.is_discrete
    assert action_space.shape == (20, 5, 6)
    assert action_space.np_dtype == np.float32
    assert action_space.ms_dtype == mstype.float32
    sample = action_space.sample()
    assert sample.shape == action_space.shape
    assert np.max(sample) < high
    assert np.min(sample) >= low
    print(action_space)


if __name__ == "__main__":
    test_discrete_space()
    test_continuous_space()
