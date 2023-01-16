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
    Components for c51.
"""

from mindspore_rl.algorithm.c51 import config
from mindspore_rl.algorithm.c51.c51_session import CategoricalSession
from mindspore_rl.algorithm.c51.c51_trainer import CategoricalDQNTrainer
from mindspore_rl.algorithm.c51.c51 import CategoricalDQNActor
from mindspore_rl.algorithm.c51.c51 import CategoricalDQNLearner
from mindspore_rl.algorithm.c51.c51 import CategoricalDQNPolicy
from mindspore_rl.algorithm.c51.c51policy import GreedyPolicyForValueDistribution
from mindspore_rl.algorithm.c51.c51policy import EpsilonGreedyPolicyForValueDistribution
from mindspore_rl.algorithm.c51.fullyconnectednet_noisy import FullyConnectedNet

__all__ = ["config", "CategoricalSession", "CategoricalDQNTrainer",
           "CategoricalDQNActor", "CategoricalDQNLearner", "CategoricalDQNPolicy",
           "CategoricalDQNTrainer", "EpsilonGreedyPolicyForValueDistribution",
           "FullyConnectedNet"]
