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
GreedyEpsilonGreedyPolicyPolicy.
"""

from mindspore_rl.policy import policy
from mindspore_rl.policy import GreedyPolicy
from mindspore_rl.policy import RandomPolicy
import mindspore
from mindspore import Tensor
import numpy as np


class EpsilonGreedyPolicy(policy.Policy):
    r"""
    Produces a sample action base on the given epsilon-greedy policy.

    Args:
        input_network (Cell): A network returns policy action.
        size (int): Shape of epsilon.
        epsi_high (float): A high epsilon for exploration betweens [0, 1].
        epsi_low (float): A low epsilon for exploration betweens [0, epsi_high].
        decay (float): A decay factor applied to epsilon.
        action_space_dim (int): Dimensions of the action space.

    Examples:
        >>> state_dim, hidden_dim, action_dim = (4, 10, 2)
        >>> input_net = FullyConnectedNet(state_dim, hidden_dim, action_dim)
        >>> policy = EpsilonGreedyPolicy(input_net, 1, 0.1, 0.1, 100, action_dim)
        >>> state = Tensor(np.ones([1, state_dim]).astype(np.float32))
        >>> step =  Tensor(np.array([10,]).astype(np.float32))
        >>> output = policy(state, step)
        >>> print(output.shape)
        (1,)
    """

    def __init__(self,
                 input_network,
                 size,
                 epsi_high,
                 epsi_low,
                 decay,
                 action_space_dim):
        super(EpsilonGreedyPolicy, self).__init__()
        self._input_network = input_network

        self.sub = mindspore.ops.Sub()
        self.add = mindspore.ops.Add()
        self.div = mindspore.ops.Div()
        self.mul = mindspore.ops.Mul()
        self.exp = mindspore.ops.Exp()
        self.slice = mindspore.ops.Slice()
        self.squeeze = mindspore.ops.Squeeze(1)
        self.less = mindspore.ops.Less()
        self.select = mindspore.ops.Select()
        self.randreal = mindspore.ops.UniformReal()

        self.decay_epsilon = (epsi_high != epsi_low)
        self.epsi_low = epsi_low
        self._size = size
        self._shape = (1,)
        self._elow_arr = np.ones(self._size) * epsi_low
        self._ehigh_arr = np.ones(self._size) * epsi_high
        self._steps_arr = np.ones(self._size)
        self._decay_arr = np.ones(self._size) * decay
        self._mone_arr = np.ones(self._size) * -1

        self._epsi_high = Tensor(self._ehigh_arr, mindspore.float32)
        self._epsi_low = Tensor(self._elow_arr, mindspore.float32)
        self._decay = Tensor(self._decay_arr, mindspore.float32)
        self._mins_one = Tensor(self._mone_arr, mindspore.float32)

        self._action_space_dim = action_space_dim
        self.greedy_policy = GreedyPolicy(self._input_network)
        self.random_policy = RandomPolicy(self._action_space_dim)

    # pylint:disable=W0221
    def construct(self, state, step):
        """
        The interface of the construct function.

        Args:
            state (Tensor): The input tensor for network.
            step (Tensor): The current step, effects the epsilon decay.

        Returns:
            The output action.
        """
        greedy_action = self.greedy_policy(state)
        random_action = self.random_policy()

        if self.decay_epsilon:
            epsi_sub = self.sub(self._epsi_high, self._epsi_low)
            epsi_exp = self.exp(
                self.mul(
                    self._mins_one,
                    self.div(
                        step,
                        self._decay)))
            epsi_mul = self.mul(epsi_sub, epsi_exp)
            epsi = self.add(self._epsi_low, epsi_mul)
            epsi = self.slice(epsi, (0, 0), (1, 1))
            epsi = self.squeeze(epsi)
        else:
            epsi = self.epsi_low

        cond = self.less(self.randreal(self._shape), epsi)
        output_action = self.select(cond, random_action, greedy_action)
        return output_action
