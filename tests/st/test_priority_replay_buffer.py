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
Test case for PriorityReplayBuffer.
'''

import pytest
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore_rl.core.priority_replay_buffer import PriorityReplayBuffer


@pytest.mark.skip(reason="Fail on run package upgrade.")
def test_priority_replay_buffer():
    '''
    Feature: Test priority replay buffer
    Description: Test priority replay buffer
    Expectation: success.
    '''

    capacity = 200
    batch_size = 32
    state_shape, state_dtype = (17,), mindspore.float32
    action_shape, action_dtype = (6,), mindspore.int32
    shapes = (state_shape, action_shape)
    dtypes = (state_dtype, action_dtype)
    prb = PriorityReplayBuffer(1., capacity, batch_size, shapes, dtypes, seed0=0, seed1=42)

    # Push 100 timestep transitions to priority replay buffer.
    for i in range(100):
        state = Tensor(np.ones(state_shape) * i, state_dtype)
        action = Tensor(np.ones(action_shape) * i, action_dtype)
        prb.push(state, action)

    # Sample a batch of transitions, the indices should be consist with transition.
    indices, weights, states, actions = prb.sample(1.)
    assert np.all(indices.asnumpy() < 100)
    states_expect = np.broadcast_to(indices.asnumpy().reshape(-1, 1), states.shape)
    actions_expect = np.broadcast_to(indices.asnumpy().reshape(-1, 1), actions.shape)
    assert np.allclose(states.asnumpy(), states_expect)
    assert np.allclose(actions.asnumpy(), actions_expect)

    # Minimize the priority, these transition will not be sampled next time.
    priorities = Tensor(np.ones(weights.shape) * 1e-7, mindspore.float32)
    prb.update_priorities(indices, priorities)

    indices_new, _, states_new, actions_new = prb.sample(1.)
    assert np.all(indices_new.asnumpy() < 100)
    assert np.all(indices.asnumpy() != indices_new.asnumpy())
    states_expect = np.broadcast_to(indices_new.asnumpy().reshape(-1, 1), states.shape)
    actions_expect = np.broadcast_to(indices_new.asnumpy().reshape(-1, 1), actions.shape)
    assert np.allclose(states_new.asnumpy(), states_expect)
    assert np.allclose(actions_new.asnumpy(), actions_expect)


if __name__ == "__main__":
    test_priority_replay_buffer()
