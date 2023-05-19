# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
A2C config.
"""
import mindspore

from mindspore_rl.distribution.distribution_policies.async_multi_actor_single_learner_dp import (
    AsyncMultiActorSingleLearnerDP,
)
from mindspore_rl.environment import GymEnvironment
from mindspore_rl.environment.pyfunc_wrapper import PyFuncWrapper

from .a2c import A2CActor, A2CLearner, A2CPolicyAndNetwork

collect_env_params = {
    "name": "CartPole-v0",
    "seed": 42,
}
eval_env_params = {"name": "CartPole-v0"}
policy_params = {
    "lr": 0.01,
    "state_space_dim": 4,
    "action_space_dim": 2,
    "hidden_size": 128,
    "gamma": 0.99,
    "compute_type": mindspore.float32,
}
learner_params = {
    "gamma": 0.99,
    "state_space_dim": 4,
    "action_space_dim": 2,
}
algorithm_config = {
    "actor": {
        "number": 1,
        "type": A2CActor,
        "params": None,
        "policies": [],
        "networks": ["a2c_net"],
    },
    "learner": {
        "number": 1,
        "type": A2CLearner,
        "params": learner_params,
        "networks": ["a2c_net_train", "a2c_net"],
    },
    "policy_and_network": {
        "type": A2CPolicyAndNetwork,
        "params": policy_params,
    },
    "collect_environment": {
        "number": 1,
        "type": GymEnvironment,
        "wrappers": [PyFuncWrapper],
        "params": collect_env_params,
    },
    "eval_environment": {
        "number": 1,
        "type": GymEnvironment,
        "wrappers": [PyFuncWrapper],
        "params": collect_env_params,
    },
}

deploy_config = {
    "auto_distribution": True,
    "distribution_policy": AsyncMultiActorSingleLearnerDP,
    "worker_num": 2,
    "network": "a2c_net",
    "algo_name": "a2c",
    "config": {
        "DATA": [(200, 1), (200, 4), (200, 1), (200, 1)],
        "TYPE": [
            "mindspore.float32",
            "mindspore.float32",
            "mindspore.int32",
            "mindspore.bool_",
        ],
    },
}
