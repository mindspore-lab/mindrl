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
#pylint: disable=E0402
from .mappo import MAPPOAgent, MAPPOActor, MAPPOLearner, MAPPOPolicy
from .mappo_replaybuffer import MAPPOReplayBuffer
from .mpe_environment import MPEMultiEnvironment

NUM_AGENT = 3
collect_env_params = {'name': 'simple_spread', 'proc_num': 32, 'num': 128, 'num_agent': NUM_AGENT}
eval_env_params = {'name': 'simple_spread', 'proc_num': 1, 'num': 1}

policy_params = {
    'state_space_dim': 0,
    'action_space_dim': 0,
    'hidden_size': 64,
}

learner_params = {
    'learning_rate': 0.0007,
    'gamma': 0.99,
    'td_lambda': 0.95,
    'iter_time': 10,
}

trainer_params = {
    'duration': 25,
    'eval_interval': 20,
    'metrics': False,
    'ckpt_path': './ckpt',
}

algorithm_config = {

    'agent': {
        'number': NUM_AGENT,
        'type': MAPPOAgent,
    },

    'actor': {
        'number': 1,
        'type': MAPPOActor,
        'policies': ['collect_policy'],
        'networks': ['critic_net'],
    },
    'learner': {
        'number': 1,
        'type': MAPPOLearner,
        'params': learner_params,
        'networks': ['actor_net', 'critic_net']
    },
    'policy_and_network': {
        'type': MAPPOPolicy,
        'params': policy_params
    },

    'replay_buffer': {
        "multi_type_replaybuffer": True,
        'local_replaybuffer': {
            'number': NUM_AGENT,
            'type': MAPPOReplayBuffer,
            'capacity': 26,
        },
        'global_replaybuffer': {
            'number': 1,
            'type': MAPPOReplayBuffer,
            'capacity': 26,
        }

    },

    'collect_environment': {
        'type': MPEMultiEnvironment,
        'params': collect_env_params
    },
    'eval_environment': {
        'type': MPEMultiEnvironment,
        'params': eval_env_params
    },
}
