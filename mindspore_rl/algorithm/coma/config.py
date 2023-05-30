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
COMA config
"""

# pylint: disable=E0402
import mindspore

from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer
from mindspore_rl.environment import StarCraft2Environment

from .coma import COMAActor, COMALearner, COMAPolicy

BATCH_SIZE = 8
collect_env_params = {
    "StarCraft2Environment": {
        "sc2_args": {"map_name": "2s3z", "state_last_action": False, "seed": 1}
    }
}
eval_env_params = {
    "StarCraft2Environment": {
        "sc2_args": {"map_name": "2s3z", "state_last_action": False}
    }
}

policy_params = {
    "epsi_high": 0.5,
    "epsi_low": 0.01,
    "decay": 200,
    "state_space_dim": 0,
    "action_space_dim": 0,
    "hidden_size": 64,
    "embed_dim": 32,
    "hypernet_embed": 64,
    "time_length": 100000,
    "batch_size": BATCH_SIZE,
    "compute_type": mindspore.float32,
}

learner_params = {
    "actor_lr": 5e-4,
    "critic_lr": 5e-4,
    "gamma": 0.99,
    "decay": 0.99,
    "epsilon": 1e-5,
    "clip_norm": 10.0,
    "batch_size": BATCH_SIZE,
}

trainer_params = {
    "batch_size": BATCH_SIZE,
    "ckpt_path": "./ckpt",
    "save_per_episode": 5000,
}

algorithm_config = {
    "actor": {
        "number": 1,
        "type": COMAActor,
        "policies": ["collect_policy", "eval_policy"],
    },
    "learner": {
        "number": 1,
        "type": COMALearner,
        "params": learner_params,
        "networks": ["policy_net", "critic_net", "target_critic_net"],
    },
    "policy_and_network": {"type": COMAPolicy, "params": policy_params},
    "collect_environment": {
        "number": BATCH_SIZE,
        "num_parallel": BATCH_SIZE,
        "type": StarCraft2Environment,
        "params": collect_env_params,
    },
    "eval_environment": {
        "number": 1,
        "type": StarCraft2Environment,
        "params": eval_env_params,
    },
    "replay_buffer": {
        "number": 1,
        "type": UniformReplayBuffer,
        "capacity": 121,
        "sample_size": BATCH_SIZE,
    },
}
