# Copyright 2022 Huawei Technologies Co., Ltd
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
n step buffer.
"""


import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer


class NStepBuffer(nn.Cell):
    r'''
    NStepBuffer: a buffer which get n step tensors lists.

    Args:
        data_shapes (tuple[int64]): the shape of each element in NStepBuffer.
        data_types (tuple[mindspore.dtype]): the type of each element in NStepBuffer.
        td_step (int): the size of the data you want to obtain
        buffer_size (int): the size of the NStepBuffer. Default: 10000.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> import mindspore.ops as ops
        >>> from mindspore_rl.utils.n_step_buffer import NStepBuffer
        >>> data_shapes, data_types, td_step= [(4,), (1,)], [ms.float32, ms.int32], 2
        >>> test = NStepBuffer(data_shapes,data_types,td_step)
        >>> exp1=[Tensor([2.,3.,1.,2.],ms.float32),Tensor([1],ms.int32)]
        >>> exp2=[Tensor([3.,3.,1.,2.],ms.float32),Tensor([2],ms.int32)]
        >>> test.add(exp1)
        >>> test.add(exp2)
        >>> a, b = test.get_data()
        >>> test.clear()
    '''
    def __init__(self, data_shapes, data_types, td_step, buffer_size=10000):
        """Initialize n step buffer"""
        super(NStepBuffer, self).__init__()
        self.buffer = UniformReplayBuffer(1, buffer_size, data_shapes, data_types)
        self.td_data = UniformReplayBuffer(1, td_step, data_shapes, data_types)
        self.less = P.Less()
        self.false = Tensor((False,), ms.bool_)
        self.true = Tensor((True,), ms.bool_)
        self.td = Tensor(td_step, ms.int64)
        self.zero_value = Tensor(0, ms.int64)

    def push(self, exp):
        """
        add element(tuple(Tensors)) to NStepBuffer in the end of buffer.
        Args:
            element (tuple(Tensor) or list[tensor]): The input element.

        Returns:
            Bool, true.
        """
        self.buffer.insert(exp)
        return True

    def get_data(self):
        """
        Get n elements int the last of the NStepBuffer.
        Returns:
            Bool, whether the data is reasonable.
            tuple(Tensors), the element in NStepBuffer.
        """
        check = self.true
        if self.buffer.count <= self.td:
            check = self.false
        m = self.zero_value
        while self.less(m, self.td) and check:
            keep = self.buffer.get_item(self.buffer.count-self.td+m)
            self.td_data.insert(keep)
            m += 1
        return check, self.td_data.buffer

    def clear(self):
        """
        Clear the created NStepBuffer. Only reset the NStepBuffer, clear the data and reset the size
        in NStepBuffer and keep the instance of this NStepBuffer.
        Returns:
            Bool, true.
        """
        self.buffer.reset()
        self.td_data.reset()
        return True

    def size(self):
        """
        Get the used size of the NStepBuffer.
        Returns:
            size (int), the number of element in the NStepBuffer.
        """
        b_size = self.buffer.size()
        return b_size
        