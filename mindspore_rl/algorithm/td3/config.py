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
TD3 config.
"""

from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer
from mindspore_rl.environment import GymEnvironment
from mindspore_rl.environment.pyfunc_wrapper import PyFuncWrapper

from .td3 import TD3Actor, TD3Learner, TD3Policy

collect_env_params = {"GymEnvironment": {"name": "HalfCheetah-v2"}}
eval_env_params = {"GymEnvironment": {"name": "HalfCheetah-v2"}}

policy_params = {
    "state_space_dim": 0,
    "action_space_dim": 0,
    "hidden_size1": 400,
    "hidden_size2": 300,
}

learner_params = {
    "gamma": 0.995,
    "state_space_dim": 0,
    "action_space_dim": 0,
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "actor_update_interval": 2,
    "target_update_factor": 0.05,
    "target_update_interval": 5,
    "target_action_noise_stddev": 0.2,
    "target_action_noise_clip": 0.5,
}

trainer_params = {
    "init_collect_size": 1000,
    "ckpt_path": "./ckpt",
    "num_eval_episode": 10,
    "save_per_episode": 50,
}

actor_params = {"actor_explore_noise": 0.1}

algorithm_config = {
    "actor": {
        "number": 1,
        "type": TD3Actor,
        "params": actor_params,
        "policies": [],
        "networks": ["actor_net", "init_policy"],
    },
    "learner": {
        "number": 1,
        "type": TD3Learner,
        "params": learner_params,
        "networks": [
            "actor_net",
            "target_actor_net",
            "critic_net_1",
            "critic_net_2",
            "target_critic_net_1",
            "target_critic_net_2",
        ],
    },
    "policy_and_network": {"type": TD3Policy, "params": policy_params},
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

summary_config = {
    "mindinsight_on": False,
    "base_dir": "./summary",
    "collect_interval": 10,
}
