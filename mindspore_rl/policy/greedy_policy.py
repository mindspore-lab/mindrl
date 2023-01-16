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
GreedyPolicy.
"""

import mindspore.ops
from mindspore_rl.policy import policy


class GreedyPolicy(policy.Policy):
    r"""
    Produces a sample action base on the given greedy policy.

    Args:
        input_network(Cell): network used to generate action probs by input state.

    Examples:
        >>> state_dim, hidden_dim, action_dim = 4, 10, 2
        >>> input_net = FullyConnectedNet(state_dim, hidden_dim, action_dim)
        >>> policy = GreedyPolicy(input_net)
        >>> state = Tensor(np.ones([2, 4]).astype(np.float32))
        >>> output = policy(state)
        >>> print(output.shape)
        (2,)
    """

    def __init__(self,
                 input_network):
        super(GreedyPolicy, self).__init__()
        self._input_network = input_network
        self.argmax = mindspore.ops.Argmax()

    # pylint:disable=W0221
    def construct(self, state):
        """
        Returns the best action.

        Args:
            state (Tensor): State tensor as the input of network.

        Returns:
            action_max, the best action.
        """

        actions = self._input_network(state)
        action_max = self.argmax(actions)
        return action_max
