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
    IMPALA config.
"""

from mindspore_rl.environment import GymEnvironment
from mindspore_rl.environment.pyfunc_wrapper import PyFuncWrapper

from .impala import IMPALAActor, IMPALALearner, IMPALANetwork

collect_env_params = {
    "GymEnvironment": {
        "name": "CartPole-v0",
        "seed": 42,
    }
}
eval_env_params = {"GymEnvironment": {"name": "CartPole-v0"}}
network_params = {"state_space_dim": 4, "action_space_dim": 2, "hidden_size": 256}
policy_params = {
    "discount": 0.99,
    "length": 200,
    "random_seed": 233,
    "batch_size": 4,
    "state_space_dim": network_params["state_space_dim"],
    "action_space_dim": network_params["action_space_dim"],
}
learner_params = {
    "lr": 0.0005,
    "discount": policy_params["discount"],
    "clip_rho_threshold": 1.0,
    "clip_pg_rho_threshold": 1.0,
    "clip_cs_threshold": 1.0,
    "baseline_cost": 0.5,
    "entropy_cost": 0.01,
}
trainer_params = {
    "loop_size": policy_params["length"],
    "batch_size": policy_params["batch_size"],
    "num_evaluate_episode": 1,
    "state_space_dim": network_params["state_space_dim"],
    "action_space_dim": network_params["action_space_dim"],
}
algorithm_config = {
    "actor": {
        "number": 3,
        "type": IMPALAActor,
        "params": policy_params,
        "share_env": False,
        "policies": [],
        "networks": ["actor_net"],
        "environment": True,
    },
    "learner": {
        "number": 1,
        "type": IMPALALearner,
        "params": learner_params,
        "networks": ["learner_net"],
    },
    "policy_and_network": {"type": IMPALANetwork, "params": network_params},
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
        "params": eval_env_params,
        "share_env": False,
    },
}
