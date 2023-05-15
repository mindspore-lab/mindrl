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
Soft Update.
"""

import mindspore
from mindspore import nn
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P


class SoftUpdate(nn.Cell):
    r"""
    Update target network parameters with moving average algorithm.

    Set target network parameter as :math:`target\_param`, behavior network parameter as :math:`behavior\_param`,
    moving averaget factor as :math:`factor`.
    Then :math:`target\_param = (1. - factor) * behavior\_param + factor * target\_param`.

    Args:
        factor (float): moving average factor between [0, 1].
        update_interval (int): The target network parameters will be updated every `update_interval` steps.
        behavior_params(list(Parameter)): list of behavior network parameters.
        target_params(list(Parameter)): list of target network parameters.

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore.common.parameter import ParameterTuple
        >>> from mindspore_rl.utils import SoftUpdate
        >>> class Net(nn.Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.behavior_params = ParameterTuple(nn.Dense(10, 20).trainable_params())
        >>>         self.target_params = ParameterTuple(nn.Dense(10, 20).trainable_params())
        >>>         self.updater = SoftUpdate(0.9, 2, self.behavior_params, self.target_params)
        >>>     def construct(self):
        >>>         return self.updater()
        >>> net = Net()
        >>> for _ in range(10):
        >>>     net()
        >>> np.allclose(net.behavior_params[0].asnumpy(), net.target_params[0].asnumpy(), atol=1e-5)
        True
    """

    def __init__(self, factor, update_interval, behavior_params, target_params):
        super().__init__()
        self.factor = factor
        self.update_interval = update_interval
        self.behavior_params = ParameterTuple(behavior_params)
        self.target_params = ParameterTuple(target_params)

        self.mod = P.Mod()
        self.assign = P.Assign()
        self.hyper_map = C.HyperMap()
        self.steps = Parameter(
            initializer(0, [1], mindspore.int32), name="steps", requires_grad=False
        )

    def _update(self, factor, behavior_param, target_param):
        new_param = (1.0 - factor) * target_param + factor * behavior_param
        self.assign(target_param, new_param)
        return target_param

    def construct(self):
        if not self.mod(self.steps, self.update_interval):
            updater = F.partial(self._update, self.factor)
            self.hyper_map(updater, self.behavior_params, self.target_params)
        self.steps += 1
        return self.steps
