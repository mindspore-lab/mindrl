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
Implementation of Reservoir Replay Buffer class.
"""

import mindspore.nn as nn
from mindspore.ops.operations._rl_inner_ops import ReservoirReplayBufferCreate, ReservoirReplayBufferPush
from mindspore.ops.operations._rl_inner_ops import ReservoirReplayBufferSample, ReservoirReplayBufferDestroy


class ReservoirReplayBuffer(nn.Cell):
    """
    ReservoirReplayBufferCreate is experience container used in reinforcement learning.
    The algorithm is proposed in `Random sampling with a reservoir <https://dl.acm.org/doi/pdf/10.1145/3147.3165>`
    which used in `Deep Counterfactual Regret Minimization <https://arxiv.org/abs/1811.00164>`.
    It lets the reinforcement learning agents remember and reuse experiences from the past. Besides, It keeps an
    'unbiased' sample of previous iterations.

    Args:
        capcity (int64): Capacity of the buffer.
        shapes (list[tuple[int]]): The dimensionality of the transition.
        dtypes (list[:class:`mindspore.dtype`]): The type of the transition.
        seed0 (int): Random seed0, must be non-negative. Default: 0.
        seed1 (int): Random seed1, must be non-negative. Default: 0.

    Outputs:
        handle(Tensor): Handle of created replay buffer instance with dtype int64 and shape (1,).

    Raises:
        TypeError: The args not provided.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore_rl.core.reservoir_replay_buffer import ReservoirReplayBuffer
        >>> capacity = 10000
        >>> batch_size = 10
        >>> shapes = [(4,), (1,), (1,), (4,)]
        >>> dtypes = [ms.float32, ms.int32, ms.float32, ms.float32]
        >>> replaybuffer = ReservoirReplayBuffer(alpha, capacity, batch_size, shapes, dtypes)
        >>> print(replaybuffer)
        ReservoirReplayBuffer<>
    """

    def __init__(self, capacity, sample_size, shapes, dtypes, seed0=0, seed1=0):
        super(ReservoirReplayBuffer, self).__init__()
        handle = ReservoirReplayBufferCreate(capacity, shapes, dtypes, seed0, seed1)().asnumpy().item()
        self.push_op = ReservoirReplayBufferPush(handle).add_prim_attr('side_effect_io', True)
        self.sample_op = ReservoirReplayBufferSample(handle, sample_size, shapes, dtypes)
        self.destroy_op = ReservoirReplayBufferDestroy(handle).add_prim_attr('side_effect_io', True)

    def push(self, *transition):
        """
        Push a transition to the buffer.

        Args:
            transition (List[Tensor]): insert a list of tensor which matches with the initialized shapes
                and dtypes into the buffer.

        Returns:
            handle(Tensor), The replay buffer instance handle with dtype int64 and shape (1,).
        """

        return self.push_op(transition)

    def sample(self):
        """
        Samples a batch of transitions from the replay buffer.

        Returns:
            transitions (tuple(Tensor)), transitions with variable-length tensors.
        """

        return self.sample_op()

    def destroy(self):
        r"""
        Destroy the replay buffer.

        Returns:
            The replay buffer instance handle with dtype int64 and shape (1,).
        """

        return self.destroy_op()
