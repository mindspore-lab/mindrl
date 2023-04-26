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
"""
Test case for UniformReplayBuffer.
"""

import mindspore
import numpy as np
import pytest
from mindspore import Tensor, context, set_seed

from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer


class TestURB(mindspore.nn.Cell):
    """Test replay buffer"""

    def __init__(self, capacity, batch_size):
        super().__init__()
        self.capacity = capacity
        state_shape, state_dtype = (17,), mindspore.float32
        action_shape, action_dtype = (6,), mindspore.float32
        shapes = (state_shape, action_shape)
        dtypes = (state_dtype, action_dtype)
        self.urb = UniformReplayBuffer(batch_size, self.capacity, shapes, dtypes)
        self.init_s = Tensor(np.ones(state_shape), state_dtype)
        self.init_a = Tensor(np.ones(action_shape), action_dtype)

    def init(self):
        for i in range(self.capacity):
            state = self.init_s + i
            action = self.init_a + i
            self.urb.insert([state, action])
        return self.urb

    def construct(self):
        return self.urb.sample()[0]


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_uniform_replay_buffer():
    """
    Feature: Test uniform replay buffer
    Description: Test uniform replay buffer
    Expectation: success.
    """

    capacity = 200
    batch_size = 16
    set_seed(10)
    state_shape, state_dtype = (17,), mindspore.float32
    action_shape, action_dtype = (6,), mindspore.int32
    shapes = (state_shape, action_shape)
    dtypes = (state_dtype, action_dtype)
    urb = UniformReplayBuffer(batch_size, capacity, shapes, dtypes)

    # Push 100 timestep transitions to uniform replay buffer.
    for i in range(100):
        state = Tensor(np.ones(state_shape) * i, state_dtype)
        action = Tensor(np.ones(action_shape) * i, action_dtype)
        urb.insert([state, action])

    context.set_context(mode=context.GRAPH_MODE)
    test_urb = TestURB(200, 32)
    test_urb.init()
    s0 = test_urb()
    s1 = test_urb()
    assert not np.allclose(s0.asnumpy(), s1.asnumpy())


if __name__ == "__main__":
    test_uniform_replay_buffer()
