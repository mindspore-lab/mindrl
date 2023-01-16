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
The Policy base class.
"""

import mindspore.nn as nn


class Policy(nn.Cell):
    """
    The virtual base class for the policy. This class should be overridden before calling in the model.
    """

    def __init__(self):
        super(Policy, self).__init__(auto_prefix=False)

    def construct(self, *inputs, **kwargs):
        """
        The interface of the construct function. Inherited and used by users.
        Args can refer to 'epsilongreedypolicy', 'randompolicy', etc.

        Args:
            inputs: it's depended on the user definition.
            kwargs: it's depended on the user definition.

        Returns:
             User defined. Usually, it returns an action value or the probability distribution of an action.
        """

        raise NotImplementedError("Method should be overridden by subclass.")
