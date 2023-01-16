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
Test case for discounted reward.
'''

import pytest
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore_rl.utils import DiscountedReturn

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_discounted_return():
    '''
    Feature: Test discounted reward class.
    Description: Test discounted reward class
    Expectation: success.
    '''

    # [Timestep, Batch, ...]
    net = DiscountedReturn(gamma=0.99)
    reward = Tensor([[1], [1], [1], [1]], dtype=mindspore.float32)
    done = Tensor([[False], [False], [True], [False]])
    last_state_value = Tensor([2.], dtype=mindspore.float32)
    ret = net(reward, done, last_state_value)
    expect = np.array([[2.9701], [1.99], [1.], [2.98]], dtype=np.float32)
    assert np.allclose(ret.asnumpy(), expect)

    # [Timestep, ...]
    reward = Tensor([1, 1, 1, 1], dtype=mindspore.float32)
    done = Tensor([False, True, False, True])
    last_state_value = Tensor(2., dtype=mindspore.float32)
    ret = net(reward, done, last_state_value)
    expect = np.array([1.99, 1, 1.99, 1.], dtype=np.float32)
    assert np.allclose(ret.asnumpy(), expect)

if __name__ == "__main__":
    test_discounted_return()
