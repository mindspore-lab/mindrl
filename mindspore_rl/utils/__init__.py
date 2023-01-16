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
Network component used to implement polices.
"""

from mindspore_rl.utils.discounted_return import DiscountedReturn
from mindspore_rl.utils.soft_update import SoftUpdate
from mindspore_rl.utils.noise import OUNoise
from mindspore_rl.utils.callback import CallbackParam
from mindspore_rl.utils.callback import CallbackManager
from mindspore_rl.utils.callback import TimeCallback
from mindspore_rl.utils.callback import LossCallback
from mindspore_rl.utils.callback import EvaluateCallback
from mindspore_rl.utils.callback import CheckpointCallback
from mindspore_rl.utils.utils import update_config
from mindspore_rl.utils.batch_read_write import BatchRead, BatchWrite
from mindspore_rl.utils.tensor_array import TensorArray
from mindspore_rl.utils.tensors_queue import TensorsQueue
from .mcts import VanillaFunc, AlgorithmFunc, MCTS

__all__ = ["DiscountedReturn", "CallbackParam", "CallbackManager", "TimeCallback",
           "LossCallback", "EvaluateCallback", "CheckpointCallback", "SoftUpdate",
           "OUNoise", "VanillaFunc", "AlgorithmFunc", "MCTS", "update_config",
           "BatchRead", "BatchWrite", "TensorArray", "TensorsQueue"]
