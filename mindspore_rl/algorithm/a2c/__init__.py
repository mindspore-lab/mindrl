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
    Components for a2c.
"""

from mindspore_rl.algorithm.a2c import config
from mindspore_rl.algorithm.a2c.a2c_session import A2CSession
from mindspore_rl.algorithm.a2c.a2c_trainer import A2CTrainer
from mindspore_rl.algorithm.a2c.a2c import A2CActor, A2CLearner, A2CPolicyAndNetwork

__all__ = ["config", "A2CSession", "A2CActor", "A2CLearner", "A2CPolicyAndNetwork", "A2CTrainer"]
