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
Implementation of Priority Replay Buffer class.
"""

import mindspore
from mindspore import Tensor, nn
from mindspore.ops.operations._rl_inner_ops import (
    PriorityReplayBufferCreate,
    PriorityReplayBufferDestroy,
    PriorityReplayBufferPush,
    PriorityReplayBufferSample,
    PriorityReplayBufferUpdate,
)


class PriorityReplayBuffer(nn.Cell):
    """
    PriorityReplayBuffer is experience container used in Deep Q-Networks.
    The algorithm is proposed in `Prioritized Experience Replay <https://arxiv.org/abs/1511.05952>`.
    Same as the normal replay buffer, it lets the reinforcement learning agents remember and reuse experiences from the
    past. Besides, it replays important transitions more frequently and improve sample effciency.

    Args:
        alpha (float): parameter to control degree of prioritization.
            ``0`` means the uniform sampling, ``1`` means priority sampling.
        capacity (int): the capacity of the buffer.
        sample_size (int): size for sampling from the buffer.
        shapes (list[int]): the shape of each tensor in a buffer element.
        types (list[mindspore.dtype]): the data type of each tensor in a buffer element.
        seed0 (int): Seed0 value for random generating. Default: ``0``.
        seed1 (int): Seed1 value for random generating. Default: ``0``.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore_rl.core.priority_replay_buffer import PriorityReplayBuffer
        >>> capacity = 10000
        >>> batch_size = 10
        >>> shapes = [(4,), (1,), (1,), (4,)]
        >>> types = [ms.float32, ms.int32, ms.float32, ms.float32]
        >>> replaybuffer = PriorityReplayBuffer(alpha, capacity, batch_size, shapes, types)
        >>> print(replaybuffer)
        PriorityReplayBuffer<>
    """

    def __init__(self, alpha, capacity, sample_size, shapes, types, seed0=0, seed1=0):
        super().__init__()
        handle = (
            PriorityReplayBufferCreate(capacity, alpha, shapes, types, seed0, seed1)()
            .asnumpy()
            .item()
        )
        self.push_op = PriorityReplayBufferPush(handle).add_prim_attr(
            "side_effect_io", True
        )
        self.sample_op = PriorityReplayBufferSample(handle, sample_size, shapes, types)
        self.update_op = PriorityReplayBufferUpdate(handle).add_prim_attr(
            "side_effect_io", True
        )
        self.destroy_op = PriorityReplayBufferDestroy(handle).add_prim_attr(
            "side_effect_io", True
        )

    def insert(self, *transition):
        """
        Push a transition to the buffer. If the buffer is full, the oldest one will be removed.

        Args:
            transition (List[Tensor]): insert a list of tensor which matches with the initialized shapes
                and dtypes into the buffer.

        Returns:
            handle(Tensor), Priority replay buffer instance handle with dtype int64 and shape :math:`(1,)`.
        """

        return self.push_op(transition)

    def sample(self, beta):
        """
        Samples a batch of transitions from the replay buffer.

        Args:
            beta (float): parameter to control degree of sampling correction.
                ``0`` means the no correction, ``1`` means full correction.

        Returns:
            indices (Tensor), the transition indices in the replay buffer.
            weights (Tensor), the weight used to correct for sampling bias.
            transitions (tuple(Tensor)), transitions with variable-length tensors.
        """

        return self.sample_op(beta)

    def update_priorities(self, indices, priorities):
        """
        Update transition prorities.

        Args:
            indices (Tensor): transition indices. The caller needs to ensure the validity of the indices.
            priorities (Tensor): transition priorities.

        Returns:
            tuple(Tensor), Transition with its indices and correction weights.
        """

        return self.update_op(indices, priorities)

    def full(self):
        """whether the buffer is full"""
        return Tensor(False, mindspore.bool_)

    def reset(self):
        """reset the buffer"""
        raise ValueError("reset() is not supported by PriorityReplayBuffer.")

    def destroy(self):
        r"""
        Destroy the replay buffer.

        Returns:
            Priority replay buffer instance handle with dtype int64 and shape :math:`(1,)`.
        """

        return self.destroy_op()
