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
Noise class for exploration.
"""

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
import mindspore.nn.probability.distribution as msd
from mindspore.common.initializer import initializer


class OUNoise(nn.Cell):
    r"""
    Perform Ornstein-Uhlenbeck (OU) noise base on actions.

    Set zero-mean normal distribution as :math:`N(0, stddev)`,
    Then the next temporal value is :math:`x\_next = (1 - damping) * x - N(0, stddev)`,
    The action with OU Noise is :math:`action += x\_next`.

    Args:
        stddev (float): stddev of Ornstein-Uhlenbeck (OU) noise.
        damping (float): damping of Ornstein-Uhlenbeck (OU) noise.
        action_shape(tuple): action shape.

    Inputs:
        - **actions** (Tensor) - Actions before perferming noise.

    Outputs:
        - **actions** (Tensor) - Actions after perferming noise.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore_rl.utils import OUNoise
        >>> action_shape = (6,)
        >>> actions = Tensor(np.ones(action_shape))
        >>> net = OUNoise(stddev=0.2, damping=0.15, action_shape=action_shape)
        >>> actions = net(actions)
        >>> print(actions.shape)
        (6,)
    """
    def __init__(self, stddev, damping, action_shape):
        super(OUNoise, self).__init__()
        self.damping = damping
        self.x = Parameter(initializer(0, action_shape), name="x", requires_grad=False)
        self.normal = msd.Normal(0., stddev)
        self.assign = P.Assign()

    def construct(self, actions):
        noise = self.normal.sample(actions.shape)
        self.x = (1.0 - self.damping) * self.x + noise
        return actions + self.x
