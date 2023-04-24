# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
DQN config.
"""
import mindspore

from mindspore_rl.algorithm.dqn.dqn import DQNActor, DQNLearner
from mindspore_rl.algorithm.dueling_dqn.dueling_dqn import DuelingDQNPolicy
from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer
from mindspore_rl.environment import GymEnvironment
from mindspore_rl.environment.pyfunc_wrapper import PyFuncWrapper

learner_params = {"gamma": 0.99, "lr": 0.001}
trainer_params = {
    "num_evaluate_episode": 10,
    "ckpt_path": "./ckpt",
    "save_per_episode": 50,
    "eval_per_episode": 10,
}

collect_env_params = {"name": "CartPole-v0"}
eval_env_params = {"name": "CartPole-v0"}

policy_params = {
    "epsi_high": 0.1,
    "epsi_low": 0.1,
    "decay": 200,
    "state_space_dim": 0,
    "action_space_dim": 0,
    "hidden_size": 100,
    "compute_type": mindspore.float32,
}

algorithm_config = {
    "actor": {
        "number": 1,
        "type": DQNActor,
        "policies": ["init_policy", "collect_policy", "evaluate_policy"],
    },
    "learner": {
        "number": 1,
        "type": DQNLearner,
        "params": learner_params,
        "networks": ["policy_network", "target_network"],
    },
    "policy_and_network": {"type": DuelingDQNPolicy, "params": policy_params},
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
        "params": eval_env_params,
    },
    "replay_buffer": {
        "number": 1,
        "type": UniformReplayBuffer,
        "capacity": 100000,
        "sample_size": 64,
    },
}
