# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
QMIX config.
"""

# pylint: disable=E0402
import mindspore

from mindspore_rl.algorithm.mappo.mpe_environment import MultiAgentParticleEnvironment
from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer
from mindspore_rl.environment.pyfunc_wrapper import PyFuncWrapper

from .qmix import QMIXActor, QMIXMPELearner, QMIXPolicy

BATCH_SIZE = 32
NUM_AGENT = 3
collect_env_params = {"name": "simple_spread", "num_agent": NUM_AGENT, "seed": 1}
eval_env_params = {"name": "simple_spread", "num_agent": NUM_AGENT}

policy_params = {
    "epsi_high": 1.0,
    "epsi_low": 0.05,
    "decay": 200,
    "state_space_dim": 0,
    "action_space_dim": 0,
    "hidden_size": 64,
    "embed_dim": 32,
    "hypernet_embed": 64,
    "time_length": 50000,
    "batch_size": BATCH_SIZE,
    "compute_type": mindspore.float32,
    "env_name": "MultiAgentParticleEnvironment",
}

learner_params = {
    "lr": 7e-4,
    "gamma": 0.99,
    "optim_alpha": 0.99,
    "epsilon": 1e-5,
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
        "type": QMIXActor,
        "policies": ["collect_policy", "eval_policy"],
    },
    "learner": {
        "number": 1,
        "type": QMIXMPELearner,
        "params": learner_params,
        "networks": [
            "policy_net",
            "mixer_net",
            "target_policy_net",
            "target_mixer_net",
        ],
    },
    "policy_and_network": {"type": QMIXPolicy, "params": policy_params},
    "collect_environment": {
        "number": 1,
        "type": MultiAgentParticleEnvironment,
        "wrappers": [PyFuncWrapper],
        "params": collect_env_params,
    },
    "eval_environment": {
        "number": 1,
        "type": MultiAgentParticleEnvironment,
        "wrappers": [PyFuncWrapper],
        "params": eval_env_params,
    },
    "replay_buffer": {
        "number": 1,
        "type": UniformReplayBuffer,
        "capacity": 5000,
        "sample_size": BATCH_SIZE,
    },
}
