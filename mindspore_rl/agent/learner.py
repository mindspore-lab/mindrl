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
Implementation of learner class.
"""

import mindspore.nn as nn


class Learner(nn.Cell):
    r"""
    The base class of the learner. Calculate and update the self generated network through the input experience.

    Examples:
        >>> from mindspore_rl.agent.learner import Learner
        >>> from mindspore_rl.network import FullyConnectedNet
        >>> class MyLearner(Learner):
        ...   def init(self):
        ...     super(MyLearner, self).init()
        ...     self.target_network = FullyConnectedNet(4, 10, 2)
        >>> my_learner = MyLearner()
        >>> print(my_learner)
        MyLearner<
        (target_network): FullyConnectedNet<
        (linear1): Dense<input_channels=4, output_channels=10, has_bias=True>
        (linear2): Dense<input_channels=10, output_channels=2, has_bias=True>
        (relu): ReLU<>
        >
    """

    def __init__(self):
        super(Learner, self).__init__(auto_prefix=False)

    def learn(self, experience):
        """
        The interface for the learn function. The behavior of the `learn` function
        depend on the user's implementation. Usually, it takes the `samples` form
        replay buffer or other Tensors, and calculates the loss for updating the networks.

        Args:
            experience(tuple(Tensor)): Sampling from the buffer.

        Returns:
            tuple(Tensor), result which outputs after updating weights
        """

        raise NotImplementedError("Method should be overridden by subclass.")
