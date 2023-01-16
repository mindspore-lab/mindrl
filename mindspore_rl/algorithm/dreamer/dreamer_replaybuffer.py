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
"""Dreamer cpu replay buffer"""

import numpy as np
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P


class DreamerReplayBuffer(nn.Cell):
    """Dreamer replay buffer"""
    def __init__(self, batch_size, capacity, shapes, types):
        super().__init__()
        self.state = []
        self.action = []
        self.reward = []
        self.discount = []
        self.batch_size = batch_size
        self.capacity = capacity
        self.seed = np.random.RandomState()
        self.insert_ops = P.PyFunc(self._insert, types, shapes, types, shapes)
        sample_out_shape = []
        for _, shape in enumerate(shapes):
            sample_out_shape.append((self.batch_size, 50,) + shape[1:])
        self.sample_ops = P.PyFunc(self._sample, (), (), types, sample_out_shape)

    def insert(self, state, action, reward, discount):
        """insert data into buffer"""
        return self.insert_ops(state, action, reward, discount)

    def sample(self):
        """sample data from, buffer"""
        return self.sample_ops()

    def full(self):
        """whether the buffer is full"""
        return Tensor(False, ms.bool_)

    def reset(self):
        """reset the buffer"""
        self.state = []
        self.action = []
        self.reward = []
        self.discount = []

    def _insert(self, state, action, reward, discount):
        """Python implementation of insert"""
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.discount.append(discount)
        return state, action, reward, discount

    def _sample(self):
        """Python implementation of sample"""
        selected_index = self.seed.choice(len(self.state), self.batch_size)
        out_state = []
        out_action = []
        out_reward = []
        out_discount = []
        for index in selected_index:
            start_index = self.seed.randint(0, self.state[0].shape[0] - 50)
            out_state.append(self.state[index][start_index: start_index + 50])
            out_action.append(self.action[index][start_index: start_index + 50])
            out_reward.append(self.reward[index][start_index: start_index + 50])
            out_discount.append(self.discount[index][start_index: start_index + 50])
        out_state = np.stack(out_state)
        out_action = np.stack(out_action)
        out_reward = np.stack(out_reward)
        out_discount = np.stack(out_discount)
        return out_state, out_action, out_reward, out_discount
