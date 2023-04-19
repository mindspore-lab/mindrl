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
MAPPO config.
"""
# pylint: disable=E0402
import mindspore as ms
from mindspore import context

from mindspore_rl.environment.pyfunc_wrapper import PyFuncWrapper
from mindspore_rl.environment.sync_parallel_wrapper import SyncParallelWrapper

from .mappo_replaybuffer import MAPPOReplayBuffer
from .mpe_environment import MultiAgentParticleEnvironment

if context.get_context("device_target") in ["Ascend"]:
    from .mappo import MAPPOActor, MAPPOAgent, MAPPOLearner, MAPPOPolicy
else:
    from .mappo_vmap import MAPPOActor, MAPPOAgent, MAPPOLearner, MAPPOPolicy


NUM_AGENT = 3
collect_env_params = {
    "name": "simple_spread",
    "num_agent": NUM_AGENT,
    "auto_reset": True,
}
eval_env_params = {"name": "simple_spread", "num_agent": NUM_AGENT, "auto_reset": True}

policy_params = {
    "state_space_dim": 0,
    "action_space_dim": 0,
    "hidden_size": 64,
    "compute_type": ms.float32,
}

learner_params = {
    "learning_rate": 0.0007,
    "gamma": 0.99,
    "td_lambda": 0.95,
    "iter_time": 10,
}

trainer_params = {
    "duration": 25,
    "eval_interval": 20,
    "metrics": False,
    "ckpt_path": "./ckpt",
}

algorithm_config = {
    "agent": {
        "number": NUM_AGENT,
        "type": MAPPOAgent,
    },
    "actor": {
        "number": 1,
        "type": MAPPOActor,
        "policies": ["collect_policy"],
        "networks": ["critic_net"],
    },
    "learner": {
        "number": 1,
        "type": MAPPOLearner,
        "params": learner_params,
        "networks": ["actor_net", "critic_net"],
    },
    "policy_and_network": {"type": MAPPOPolicy, "params": policy_params},
    "replay_buffer": {
        "multi_type_replaybuffer": True,
        "local_replaybuffer": {
            "number": NUM_AGENT,
            "type": MAPPOReplayBuffer,
            "capacity": 26,
        },
        "global_replaybuffer": {
            "number": 1,
            "type": MAPPOReplayBuffer,
            "capacity": 26,
        },
    },
    "collect_environment": {
        "number": 128,
        "num_parallel": 32,
        "type": MultiAgentParticleEnvironment,
        "wrappers": [PyFuncWrapper, SyncParallelWrapper],
        "params": collect_env_params,
        "seed": [1 + i * 1000 for i in range(128)],
    },
    "eval_environment": {
        "number": 1,
        "num_parallel": 1,
        "type": MultiAgentParticleEnvironment,
        "wrappers": [PyFuncWrapper, SyncParallelWrapper],
        "params": eval_env_params,
    },
}
