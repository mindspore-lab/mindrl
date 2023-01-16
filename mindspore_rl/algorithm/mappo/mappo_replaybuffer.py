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
Implementation of MAPPO replaybuffer .
"""
#pylint: disable=W0235
from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer


class MAPPOReplayBuffer(UniformReplayBuffer):
    """MAPPO Replaybuffer implementation"""

    def __init__(self, batch_size, capacity, shapes, types):
        super().__init__(batch_size, capacity, shapes, types)

    def construct(self, exp, command):
        """
        Insert an element to the buffer. If the buffer is full, FIFO strategy will be used to
        replace the element in the buffer.

        Args:
            exp (List[Tensor]): insert a list of tensor which matches with the initialized shape
            and type into the buffer.

        Returns:
             element (List[Tensor]), return the whole buffer after insertion

        """
        if command:
            self.buffer_append(self.buffer, exp, self.count, self.head)
        return self.buffer
