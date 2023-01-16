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
Test case for soft update.
'''

import pytest
import numpy as np
import mindspore.nn as nn
from mindspore.common.parameter import ParameterTuple
from mindspore_rl.utils import SoftUpdate

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.behavior_params = ParameterTuple(nn.Dense(10, 20).trainable_params())
        self.target_params = ParameterTuple(nn.Dense(10, 20).trainable_params())
        self.updater = SoftUpdate(0.9, 2, self.behavior_params, self.target_params)

    def construct(self):
        return self.updater()

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_soft_update():
    '''
    Feature: Test soft update class.
    Description: Test soft update class
    Expectation: success.
    '''

    net = Net()
    for _ in range(10):
        net()

    assert np.allclose(net.behavior_params[0].asnumpy(), net.target_params[0].asnumpy(), atol=1e-5)
    assert np.allclose(net.behavior_params[1].asnumpy(), net.target_params[1].asnumpy(), atol=1e-5)

if __name__ == "__main__":
    test_soft_update()
