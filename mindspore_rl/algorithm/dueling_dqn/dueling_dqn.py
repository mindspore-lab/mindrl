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
"""Dueling DQN"""

import mindspore as ms
from mindspore import nn, ops
from mindspore.ops import operations as P

from mindspore_rl.policy import EpsilonGreedyPolicy, GreedyPolicy, RandomPolicy


class DuelingDQNPolicy:
    """Dueling DQN Policy"""

    class DuelingNet(nn.Cell):
        """Dueling Network"""

        def __init__(self, params):
            super().__init__()
            compute_type = params["compute_type"]
            self.fc1 = nn.Dense(
                params["state_space_dim"],
                params["hidden_size"],
                weight_init="XavierUniform",
            ).to_float(compute_type)
            self.relu = nn.ReLU()
            self.advantage_fc = nn.Dense(
                params["hidden_size"],
                params["action_space_dim"],
                weight_init="XavierUniform",
            ).to_float(compute_type)
            self.value_fc = nn.Dense(
                params["hidden_size"], 1, weight_init="XavierUniform"
            ).to_float(compute_type)
            self.reduce_mean = P.ReduceMean(keep_dims=True)

        def construct(self, x):
            x = self.relu(self.fc1(x))
            value = self.value_fc(x)
            advantage = self.advantage_fc(x)
            mean_adv = self.reduce_mean(advantage)
            out = value + (advantage - mean_adv)
            out = ops.cast(out, ms.float32)
            return out

    def __init__(self, params):
        self.policy_network = self.DuelingNet(params)
        self.target_network = self.DuelingNet(params)
        self.init_policy = RandomPolicy(params["action_space_dim"])
        self.collect_policy = EpsilonGreedyPolicy(
            self.policy_network,
            (1, 1),
            params["epsi_high"],
            params["epsi_low"],
            params["decay"],
            params["action_space_dim"],
        )
        self.evaluate_policy = GreedyPolicy(self.policy_network)
