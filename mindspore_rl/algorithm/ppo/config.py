# Copyright 2021 Huawei Technologies Co., Ltd
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
PPO config.
"""
import mindspore
from mindspore_rl.environment import GymEnvironment
from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer
from mindspore_rl.distribution.distribution_policies.multi_actor_single_learner_dp import MultiActorEnvSingleLearnerDP
from .ppo import PPOActor, PPOLearner, PPOPolicy

collect_env_params = {'name': 'HalfCheetah-v2'}
eval_env_params = {'name': 'HalfCheetah-v2'}

policy_params = {
    'state_space_dim': 0,
    'action_space_dim': 0,
    'hidden_size1': 200,
    'hidden_size2': 100,
    'sigma_init_std': 0.35,
    'compute_type': mindspore.float32,
}

learner_params = {
    'gamma': 0.99,
    'state_space_dim': 0,
    'action_space_dim': 0,
    'iter_times': 25,
    'epsilon': 0.2,
    'lr': 1e-3,
    'critic_coef': 0.5,
}

trainer_params = {
    'duration': 1000,
    'batch_size': 1,
    'ckpt_path': './ckpt',
    'num_eval_episode': 30,
    'num_save_episode': 50,
}

algorithm_config = {
    'actor': {
        'number': 1,
        'type': PPOActor,
        'policies': ['collect_policy', 'eval_policy'],
    },
    'learner': {
        'number': 1,
        'type': PPOLearner,
        'params': learner_params,
        'networks': ['actor_net', 'critic_net']
    },
    'policy_and_network': {
        'type': PPOPolicy,
        'params': policy_params
    },
    'collect_environment': {
        'number': 30,
        'num_parallel': 5,
        'type': GymEnvironment,
        'params': collect_env_params
    },
    'eval_environment': {
        'number': 1,
        'type': GymEnvironment,
        'params': eval_env_params
    },
    'replay_buffer': {
        'number': 1,
        'type': UniformReplayBuffer,
        'capacity': 1000,
    }
}

deploy_config = {
    'auto_distribution': True,
    'distribution_policy': MultiActorEnvSingleLearnerDP,
    'worker_num':2,
    'network': 'actor_net',
    'algo_name': 'ppo',
    'config': {}
}
