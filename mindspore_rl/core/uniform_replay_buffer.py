# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
Implementation of Replay Buffer class.
"""

import numpy as np
import mindspore as ms
from mindspore import get_seed
from mindspore import context, Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter, ParameterTuple
import mindspore.nn as nn


def _create_tensor(capacity, shapes, types):
    """
    Create tensors of replay buffer, and store them into a list.

    Args:
        capacity (int): capacity of the buffer.
        shapes (List[int]): the shape of the tensor.
        types (List[mindspore.dtype]): the type of the tensor.

    Returns:
        buffer(List[Tensor]), a list of tensor which states for the replay buffer
    """
    buffer = []
    for i in range(len(shapes)):
        buffer.append(Parameter(Tensor(np.zeros(((capacity,) + shapes[i])),
                                       types[i]), name=('buffer_%d' % i), requires_grad=False))
    return buffer


class UniformReplayBuffer(nn.Cell):
    """
    The replay buffer class.
    The replay buffer will store the experience from environment. In replay
    buffer, each element is a list of tensors. Therefore, the constructor of the
    UniformReplayBuffer class takes the shape and type of each tensor as an argument.

    Args:
        batch_size (int): size for sampling from the buffer.
        capacity (int): the capacity of the buffer.
        shapes (list[int]): the shape of each tensor in a buffer element.
        types (list[mindspore.dtype]): the data type of each tensor in a buffer element.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer
        >>> batch_size = 10
        >>> capacity = 10000
        >>> shapes = [(4,), (1,), (1,), (4,)]
        >>> types = [ms.float32, ms.int32, ms.float32, ms.float32]
        >>> replaybuffer = UniformReplayBuffer(batch_size, capacity, shapes, types)
        >>> print(replaybuffer)
        UniformReplayBuffer<>
    """

    def __init__(self, batch_size, capacity, shapes, types):
        nn.Cell.__init__(self)
        self.buffer = ParameterTuple(_create_tensor(capacity, shapes, types))
        self._capacity = capacity
        self.count = Parameter(
            Tensor(
                0,
                ms.int32),
            name="count",
            requires_grad=False)
        self.head = Parameter(
            Tensor(
                0,
                ms.int32),
            name="head",
            requires_grad=False)
        self.zero = Tensor(0, ms.int32)
        self.buffer_append = P.BufferAppend(self._capacity, shapes, types)
        self.buffer_get = P.BufferGetItem(self._capacity, shapes, types)
        seed = get_seed()
        if seed == None:
            seed = 0
        self.buffer_sample = P.BufferSample(
            self._capacity, batch_size, shapes, types, seed)
        if context.get_context('device_target') in ['Ascend']:
            self.buffer_append.add_prim_attr('primitive_target', 'CPU')
            self.buffer_get.add_prim_attr('primitive_target', 'CPU')
            self.buffer_sample.add_prim_attr('primitive_target', 'CPU')

        self.reshape = P.Reshape()
        self.assign = P.Assign()

        self.greater_equal = P.GreaterEqual()
        self.capacity_tensor = Tensor([capacity,], ms.int32)

    def insert(self, exp):
        """
        Insert an element to the buffer. If the buffer is full, FIFO strategy will be used to
        replace the element in the buffer.

        Args:
            exp (list[Tensor]): insert a list of tensor which matches with the initialized shape
                and type into the buffer.

        Returns:
             element (list[Tensor]), return the whole buffer after insertion

        """
        self.buffer_append(self.buffer, exp, self.count, self.head)
        return self.buffer

    def get_item(self, index):
        """
        Get an element from the replaybuffer in specific position(index).

        Args:
            index (int): the location of the item.

        Returns:
            element (List[Tensor]), the element from the buffer.
        """

        return self.buffer_get(self.buffer, self.count, self.head, index)

    def sample(self):
        """
        Sampling the replaybuffer, which means that it will randomly choose a set of element
        and output them.

        Returns:
            data (Tuple(Tensor)), A set of sampled elements from the buffer.
        """

        return self.buffer_sample(self.buffer, self.count, self.head)

    def reset(self):
        """
        Reset the replaybuffer. It changes the value of self.count to zero.

        Returns:
            success (boolean), whether the reset successful or not.
        """

        success = self.assign(self.count, self.zero)
        return success

    def size(self):
        """
        Return the size of the replybuffer.

        Returns:
            size (int), the number of element in the replaybuffer.
        """

        return self.count.asnumpy()

    def full(self):
        """
        Check if the replaybuffer is full or not.

        Returns:
            Full(bool), True if the replaybuffer is full, False otherwise.
        """

        count = self.reshape(self.count, (1,))
        capacity = self.reshape(self.capacity_tensor, (1,))
        return self.greater_equal(count, capacity)
