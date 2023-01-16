# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
Test case for ReservoirReplayBuffer.
'''

import pytest
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore_rl.core.reservoir_replay_buffer import ReservoirReplayBuffer


@pytest.mark.skip(reason="Fail on run package upgrade.")
def test_reservoir_replay_buffer():
    '''
    Feature: Test reservoir replay buffer
    Description: Test reservoir replay buffer
    Expectation: success.
    '''

    capacity = 100
    batch_size = 256
    state_shape, state_dtype = (17,), mindspore.float32
    action_shape, action_dtype = (6,), mindspore.int32
    shapes = (state_shape, action_shape)
    dtypes = (state_dtype, action_dtype)
    prb = ReservoirReplayBuffer(capacity, batch_size, shapes, dtypes, seed0=0, seed1=42)

    # Push 100 timestep transitions to reservoir replay buffer.
    for i in range(100):
        state = Tensor(np.ones(state_shape) * i, state_dtype)
        action = Tensor(np.ones(action_shape) * i, action_dtype)
        prb.push(state, action)

    # Sample a batch of transitions.
    states, actions = prb.sample()

    # Expect: The elements along axis 1 are same.
    assert np.allclose(states.asnumpy(), states.asnumpy()[:, 0:1])
    assert np.allclose(actions.asnumpy(), actions.asnumpy()[:, 0:1])
    assert np.allclose(states.asnumpy()[:, 0], actions.asnumpy()[:, 0])


if __name__ == "__main__":
    test_reservoir_replay_buffer()
