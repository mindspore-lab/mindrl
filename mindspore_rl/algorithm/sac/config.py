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
SAC config.
"""

from mindspore_rl.algorithm.sac.sac import SACActor, SACLearner, SACPolicy
from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer
from mindspore_rl.environment import GymEnvironment
from mindspore_rl.environment.pyfunc_wrapper import PyFuncWrapper

collect_env_params = {"name": "HalfCheetah-v2"}
eval_env_params = {"name": "HalfCheetah-v2"}

policy_params = {
    "state_space_dim": 0,
    "action_space_dim": 0,
    "hidden_sizes": [256, 256],
    "conditioned_std": True,
}

learner_params = {
    "gamma": 0.99,
    "state_space_dim": 0,
    "action_space_dim": 0,
    "epsilon": 0.2,
    "critic_lr": 3e-4,
    "actor_lr": 3e-4,
    "reward_scale_factor": 0.1,
    "critic_loss_weight": 0.5,
    "actor_loss_weight": 1.0,
    "actor_mean_std_reg": False,
    "actor_mean_reg_weight": 1e-3,
    "actor_std_reg_weight": 1e-3,
    "log_alpha": 0.0,
    "train_alpha_net": True,
    "alpha_loss_weight": 1.0,
    "alpha_lr": 3e-4,
    "target_entropy": -3.0,
    "update_factor": 0.005,
    "update_interval": 1,
}

trainer_params = {
    "duration": 1000,
    "batch_size": 256,
    "save_per_episode": 100,
    "ckpt_path": "./ckpt",
    "num_eval_episode": 30,
}

algorithm_config = {
    "actor": {
        "number": 1,
        "type": SACActor,
        "policies": ["init_policy", "collect_policy", "eval_policy"],
    },
    "learner": {
        "number": 1,
        "type": SACLearner,
        "params": learner_params,
        "networks": [
            "actor_net",
            "critic_net1",
            "critic_net2",
            "target_critic_net1",
            "target_critic_net2",
        ],
    },
    "policy_and_network": {"type": SACPolicy, "params": policy_params},
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
        "capacity": 1000000,
        "sample_size": 256,
    },
}
