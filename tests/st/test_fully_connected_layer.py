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
Test case for fully connected layer module
"""

import pytest
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore_rl.network.fully_connected_net import FullyConnectedLayers

class Net(nn.Cell):
    """
    Ground truth network
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(10, 20)
        self.fc2 = nn.Dense(20, 3)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.relu(x)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fully_connected_layer():
    '''
    Feature: Test test_fully_connected_layer class
    Description: Test test_fully_connected_layer class
    Expectation: success.
    '''
    net = Net()
    fc_layers = FullyConnectedLayers(fc_layer_params=[10, 20, 3])
    net_fc1_weight = net.trainable_params()[0]
    net_fc2_weight = net.trainable_params()[2]
    for k, v in fc_layers.parameters_dict().items():
        if 'weight' in k and '0' in k:
            v.set_data(net_fc1_weight)
        if 'weight' in k and '1' in k:
            v.set_data(net_fc2_weight)

    input_x = Tensor(np.random.random([5, 10]).astype(np.float32))
    expected = net(input_x)
    actual = fc_layers(input_x)

    assert np.allclose(expected.asnumpy(), actual.asnumpy(), atol=1e-5)

if __name__ == '__main__':
    test_fully_connected_layer()
