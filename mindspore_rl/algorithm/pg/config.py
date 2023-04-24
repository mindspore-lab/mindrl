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
"""
PG config.
"""
import mindspore

from mindspore_rl.environment import GymEnvironment
from mindspore_rl.environment.pyfunc_wrapper import PyFuncWrapper

from .pg import PGActor, PGLearner, PGPolicyAndNetwork

collect_env_params = {"name": "CartPole-v0", "seed": 42}
eval_env_params = {"name": "CartPole-v0"}
policy_params = {
    "state_space_dim": 4,
    "action_space_dim": 2,
    "hidden_size": 20,
    "compute_type": mindspore.float32,
}
trainer_params = {
    "num_evaluate_episode": 10,
    "ckpt_path": "./ckpt",
}
learner_params = {
    "state_space_dim": 4,
    "action_space_dim": 2,
    "lr": 0.001,
    "gamma": 1.0,
}
algorithm_config = {
    "actor": {
        "number": 1,
        "type": PGActor,
        "policies": ["collect_policy", "eval_policy"],
    },
    "learner": {
        "number": 1,
        "type": PGLearner,
        "params": learner_params,
        "networks": ["actor_net"],
    },
    "policy_and_network": {
        "type": PGPolicyAndNetwork,
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
