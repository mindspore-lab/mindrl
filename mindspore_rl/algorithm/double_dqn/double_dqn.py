# Copyright 2023 Huawei Technologies Co., Ltd
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
"""double_dqn dqn"""

from mindspore import ops

from mindspore_rl.algorithm.dqn import DQNLearner


class DoubleDQNLearner(DQNLearner):
    """Double DQN Learner"""

    def __init__(self, params):
        super().__init__(params)

    def learn(self, experience):
        """Model update"""
        s0, a0, r1, s1 = experience
        policy_state_values = self.policy_network(s1)
        max_policy_action = ops.argmax(policy_state_values, dim=1, keepdim=True)
        next_state_values = self.target_network(s1)
        next_state_values = ops.gather_d(
            next_state_values, 1, max_policy_action
        ).squeeze(-1)
        r1 = self.reshape(r1, (-1,))

        y_true = r1 + self.gamma * next_state_values

        # Modify last step reward
        one = self.ones_like(r1)
        y_true = self.select(r1 == -one, one, y_true)
        y_true = self.expand_dims(y_true, 1)

        success = self.policy_network_train(s0, a0, y_true)
        return success
