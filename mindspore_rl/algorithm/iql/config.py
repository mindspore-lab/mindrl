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
IQL config.
"""

from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer
from mindspore_rl.environment import GymEnvironment
from mindspore_rl.environment.pyfunc_wrapper import PyFuncWrapper

from .iql import IQLActor, IQLLearner, IQLPolicyAndNetwork

collect_env_params = {
    "GymEnvironment": {
        "name": "walker2d-medium-v2",
        "seed": 10,
    }
}
eval_env_params = {
    "GymEnvironment": {
        "name": "walker2d-medium-v2",
        "seed": 10,
    }
}

trainer_params = {
    "num_evaluate_episode": 5,
    "ckpt_path": "./ckpt",
    "save_per_episode": 5,
    "eval_per_episode": 5,
    "max_ckpt_num": 5,
    "loss_freq": 5,
    "offline_train_steps": 4000,
    "offline_data_length": 10000,
    "n_steps_per_episode": 100,
}

policy_params = {
    "state_space_dim": 27,
    "action_space_dim": 3,
    "hidden_sizes": [256, 256, 256, 256],
    "conditioned_std": False,
}
learner_params = {
    "gamma": 0.99,
    "critic_lr": 3e-4,
    "actor_lr": 3e-4,
    "value_lr": 3e-4,
    "log_alpha": 0,
    "reward_scale_factor": 0.1,
    "critic_loss_weight": 0.5,
    "actor_loss_weight": 1.0,
    "actor_mean_std_reg": False,
    "actor_mean_reg_weight": 1e-3,
    "actor_std_reg_weight": 1e-3,
    "train_alpha_net": False,
    "alpha_loss_weight": 1.0,
    "alpha_lr": 3e-4,
    "target_entropy": -3.0,
    "update_factor": 0.005,
    "update_interval": 1,
    "quantile": 0.8,
    "temperature": 0.1,
}
algorithm_config = {
    "actor": {
        "number": 1,
        "type": IQLActor,
        "params": None,
        "policies": ["actor_net", "eval_policy"],
    },
    "learner": {
        "number": 1,
        "type": IQLLearner,
        "params": learner_params,
        "networks": [
            "actor_net",
            "critic_net1",
            "critic_net2",
            "target_critic_net1",
            "target_critic_net2",
            "value_net",
        ],
    },
    "policy_and_network": {"type": IQLPolicyAndNetwork, "params": policy_params},
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
        "capacity": 2000000,
        "sample_size": 1024,
    },
}
