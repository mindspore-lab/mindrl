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
RandomPolicy.
"""

from mindspore_rl.policy import Policy
import mindspore
from mindspore import Tensor


class RandomPolicy(Policy):
    r"""
    Produces a random action betweens [0, `acton_space_dim`).

    Args:
        action_space_dim(int): dimension of the action space.

    Examples:
        >>> action_space_dim = 2
        >>> policy = RandomPolicy(action_space_dim)
        >>> output = policy()
        >>> print(output.shape)
        (1,)
    """

    def __init__(self, action_space_dim):
        super(RandomPolicy, self).__init__()
        self.randint = mindspore.ops.UniformInt()
        self.minval = Tensor(0, mindspore.int32)
        self.maxval = Tensor(action_space_dim, mindspore.int32)
        self.shape = (1,)

    # pylint:disable=W0221
    def construct(self):
        """
        Returns a random number betweens [0, `acton_space_dim`).

        Returns:
            A random integer betweens [0, `acton_space_dim`).
        """

        return self.randint(self.shape, self.minval, self.maxval)
