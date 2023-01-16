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
Test case for OU noise.
'''

import pytest
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore_rl.utils import OUNoise

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ou_noise():
    '''
    Feature: Test ou noise class.
    Description: Test ou noise class
    Expectation: success.
    '''

    action_shape = (6,)
    actions = Tensor(np.ones(action_shape), mindspore.float32)
    noise_net = OUNoise(stddev=0.2, damping=0.15, action_shape=action_shape)
    actions = noise_net(actions)
    print(actions.shape)

if __name__ == "__main__":
    test_ou_noise()
