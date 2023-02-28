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
MADDPG config.
"""
#pylint: disable=E0402
import mindspore as ms
from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer
from mindspore_rl.environment.petting_zoo_mpe_environment import PettingZooMPEEnvironment
from .maddpg import MADDPGAgent, MADDPGActor, MADDPGLearner, MADDPGPolicy

NUM_AGENT = 3
BATCH_SIZE = 1024
CONTINUOUS_ACTIONS = False
SEED = 10

collect_env_params = {
    'name': 'simple_spread',
    'num': NUM_AGENT,
    'continuous_actions': CONTINUOUS_ACTIONS,
    'seed': SEED
}
eval_env_params = collect_env_params

policy_params = {
    'state_space_dim': 18,
    'action_space_dim': 5,
    'hidden_size': 64,
    'num_agent': NUM_AGENT,
    'continuous_actions': CONTINUOUS_ACTIONS,
    'compute_type': ms.float32
}

learner_params = {
    'learning_rate': 0.01,
    'gamma': 0.95,
    'update_factor': 0.01,
    'update_interval': 1,
    'continuous_actions': CONTINUOUS_ACTIONS,
}

trainer_params = {
    'duration': 25,
    'num_eval_episode': 3,
    'init_size': BATCH_SIZE * 25,
    'num_agent': NUM_AGENT,
    'continuous_actions': CONTINUOUS_ACTIONS,
    'metrics': False,
    'ckpt_path': './ckpt',
}

algorithm_config = {
    'agent': {
        'number': NUM_AGENT,
        'type': MADDPGAgent,
    },
    'actor': {
        'number': 1,
        'type': MADDPGActor,
        'policies': ['collect_policy'],
        'networks': ['actor_net', 'target_actor_net'],
    },
    'learner': {
        'number': 1,
        'type': MADDPGLearner,
        'params': learner_params,
        'networks': ['actor_net', 'critic_net', 'target_actor_net', 'target_critic_net']
    },
    'policy_and_network': {
        'type': MADDPGPolicy,
        'params': policy_params
    },

    'replay_buffer': {'number': 1,
                      'type': UniformReplayBuffer,
                      'capacity': 1000000,
                      'sample_size': BATCH_SIZE},
    'collect_environment': {
        'type': PettingZooMPEEnvironment,
        'params': collect_env_params
    },
    'eval_environment': {
        'type': PettingZooMPEEnvironment,
        'params': eval_env_params
    },
}
