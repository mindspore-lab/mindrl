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
Component used to implement custom environments.
"""

from mindspore_rl.environment.gym_environment import GymEnvironment
from mindspore_rl.environment.environment import Environment
from mindspore_rl.environment.tag_environment import TagEnvironment
from mindspore_rl.environment.ms_environment import ms_register, MsEnvironment
from mindspore_rl.environment.space import Space
from mindspore_rl.environment.env_process import EnvironmentProcess
from mindspore_rl.environment.multi_environment_wrapper import MultiEnvironmentWrapper
from mindspore_rl.environment.sc2_environment import StarCraft2Environment
from mindspore_rl.environment.tic_tac_toe_environment import TicTacToeEnvironment
from mindspore_rl.environment.dmc_environment import DeepMindControlEnvironment

__all__ = ["GymEnvironment", "MultiEnvironmentWrapper", "Environment", "Space", "MsEnvironment", "EnvironmentProcess",
           "StarCraft2Environment", "TicTacToeEnvironment", "DeepMindControlEnvironment"]

ms_register('Tag', TagEnvironment)
