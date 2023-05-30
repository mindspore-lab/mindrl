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
A3C config.
"""

from mindspore_rl.environment import GymEnvironment
from mindspore_rl.environment.pyfunc_wrapper import PyFuncWrapper

from .a3c import A3CActor, A3CLearner, A3CPolicyAndNetwork

collect_env_params = {
    "GymEnvironment": {
        "name": "CartPole-v0",
        "seed": 42,
    }
}
eval_env_params = {"GymEnvironment": {"name": "CartPole-v0"}}
policy_params = {
    "state_space_dim": 4,
    "action_space_dim": 2,
    "hidden_size": 256,
    "gamma": 0.99,
}
learner_params = {
    "lr": 0.01,
    "state_space_dim": 4,
    "action_space_dim": 2,
}
algorithm_config = {
    "actor": {
        "number": 3,
        "type": A3CActor,
        "params": policy_params,
        "policies": [],
        "networks": ["a3c_net"],
        "envirnment": True,
        "share_env": False,
    },
    "learner": {
        "number": 1,
        "type": A3CLearner,
        "params": learner_params,
        "networks": ["a3c_net_learn", "a3c_net_copy"],
    },
    "policy_and_network": {"type": A3CPolicyAndNetwork, "params": policy_params},
    "collect_environment": {
        "number": 1,
        "type": GymEnvironment,
        "wrappers": [PyFuncWrapper],
        "params": collect_env_params,
        "share_env": False,
    },
    "eval_environment": {
        "number": 1,
        "type": GymEnvironment,
        "wrappers": [PyFuncWrapper],
        "params": collect_env_params,
        "share_env": True,
    },
}
