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
Discounted return.
"""

import mindspore as ms
import mindspore.ops.operations._rl_inner_ops as rl_ops
from mindspore import Tensor, context, nn
from mindspore.ops import operations as P


class DiscountedReturn(nn.Cell):
    r"""
    Calculate discounted return.

    Set discounted return as :math:`G`, discounted factor as :math:`\gamma`, reward as :math:`R`,
    timestep as :math:`t`, max timestep as :math:`N`. Then :math:`G_{t} = \Sigma_{t=0}^N{\gamma^tR_{t+1}}`

    For the reward sequence contain multi-episode, :math:`done` is introduced for indicating episode boundary,
    :math:`last\_state\_value` represents value after final step of last episode.

    Args:
        gamma (float): Discounted factor between [0, 1].
        need_bprop (bool): Whether need to calculate the backpropagation of discounted returns. Default: ``False`` .
        dtype (mindspore.dtype): Data type for the output. Default: ``ms.float32`` .

    Inputs:
        - **reward** (Tensor) - The reward sequence contains multi-episode.
          Tensor of shape :math:`(Timestep, Batch, ...)`
        - **done** (Tensor) - The episode done flag. Tensor of shape :math:`(Timestep, Batch)`.
          The data type must be bool.
        - **last_state_value** (Tensor) - The value after final step of last episode.
          Tensor of shape :math:`(Batch, ...)`

    Returns:
        Discounted return.

    Examples:
        >>> net = DiscountedReturn(gamma=0.99)
        >>> reward = Tensor([[1, 1, 1, 1]], dtype=ms.float32)
        >>> done = Tensor([[False, False, True, False]])
        >>> last_state_value = Tensor([2.], dtype=ms.float32)
        >>> ret = net(reward, done, last_state_value)
        >>> print(output.shape)
        (2, 2)
    """

    def __init__(self, gamma, need_bprop=False, dtype=ms.float32):
        super().__init__()
        if gamma > 1.0 or gamma < 0.0:
            raise ValueError(
                f"The discounted factor should be a number in range [0, 1], but got {gamma}."
            )

        # Fused operator only supported in GPU backend so far. Ascend and CPU backends will support it soon.
        self.enable_op_fusion = context.get_context("device_target") in ["GPU"]
        self.need_bprop = need_bprop
        self.fused_op = rl_ops.DiscountedReturn(gamma)

        self.gamma = Tensor([gamma], dtype)
        self.zeros_like = P.ZerosLike()

    def construct(self, reward, done, last_state_value):
        """
        Returns discounted return.
        """

        if self.enable_op_fusion and not self.need_bprop:
            return self.fused_op(reward, done, last_state_value)

        discounted_return = self.zeros_like(reward)
        step = reward.shape[0] - 1
        while step >= 0:
            last_state_value = (
                reward[step] + (1 - done[step]) * self.gamma * last_state_value
            )
            discounted_return[step] = last_state_value
            step -= 1
        return discounted_return
