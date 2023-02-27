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

'''PPO distribute_policy_1'''
import os
import sys
import argparse
import mindspore
from mindspore import context

from mindspore_rl.core import UniformReplayBuffer
from mindspore_rl.environment import GymEnvironment
from mindspore_rl.core import Session
#pylint: disable=C0413
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../common/'))
from ppo_algo import PPOTrainer, PPOPolicy, PPOActor, PPOLearner # noqa

context.set_context(mode=context.GRAPH_MODE, max_call_depth=100000)

parser = argparse.ArgumentParser(description='Set the number of actor workers.')
parser.add_argument("-n", "--num_actor", type=int, default=2, required=True,
                    help="The number of actor workers. Default: 2.")
parser.add_argument("-e", "--num_collect_environment", type=int, default=30, required=True,
                    help="The number of collect_environment. Default: 30.")
parser.add_argument("-ep", "--num_episode", type=int, default=100, required=True,
                    help="The number of episodes. Default: 100.")
parser.add_argument("-duration", "--duration", type=int, default=1000, required=True,
                    help="The duration. Default: 1000.")
args = parser.parse_args()
actor_number = int(args.num_actor)
environment_number = int(args.num_collect_environment)
EPISODE = int(args.num_episode)
DURATION = int(args.duration)

env_params = {'name': 'HalfCheetah-v2'}
eval_env_params = {'name': 'HalfCheetah-v2'}

policy_params = {
    'epsilon': 0.2,
    'lr': 1e-3,
    'state_space_dim': 0,
    'action_space_dim': 0,
    'hidden_size1': 200,
    'hidden_size2': 100,
    'sigma_init_std': 0.35,
    'critic_coef': 0.5,
}

learner_params = {
    'gamma': 0.99,
    'state_space_dim': 0,
    'action_space_dim': 0,
    'iter_times': 25
}

trainer_params = {
    'duration': DURATION,
    'batch_size': 1,
    'eval_interval': 20,
    'num_eval_episode': 3
}

ACT_NUM = actor_number
COLLECT_ENV_NUM = int(environment_number / actor_number)
ppo_algorithm_config = {
    'actor': {
        'number': ACT_NUM,
        'type': PPOActor,
        'params': None,
        'policies': [],
        'networks': ['actor_net'],
        'environment': True,
        'eval_environment': True,
    },
    'learner': {
        'number': 1,
        'type': PPOLearner,
        'params': learner_params,
        'networks': ['actor_net', 'critic_net', 'ppo_net_train']
    },
    'policy_and_network': {
        'type': PPOPolicy,
        'params': policy_params
    },
    'replay_buffer': {
        'number': 1,
        'type': UniformReplayBuffer,
        'capacity': DURATION,
        'data_shape': [(ACT_NUM, COLLECT_ENV_NUM, 17), (ACT_NUM, COLLECT_ENV_NUM, 6), (ACT_NUM, COLLECT_ENV_NUM, 1),
                       (ACT_NUM, COLLECT_ENV_NUM, 17), (ACT_NUM, COLLECT_ENV_NUM, 6), (ACT_NUM, COLLECT_ENV_NUM, 6)],
        'data_type': [
            mindspore.float32, mindspore.float32, mindspore.float32,
            mindspore.float32, mindspore.float32, mindspore.float32,
        ],
        'sample_size': 1,
    },
    'collect_environment': {
        'number': COLLECT_ENV_NUM,
        'type': GymEnvironment,
        'params': env_params
    },
    'eval_environment': {
        'type': GymEnvironment,
        'params': eval_env_params
    },
}

deploy_config = {'auto_distribution': True, 'worker_num': 2, 'distribution_policy': 'MultiActorLearner',
                 'config': {'0': {'ip': 'tcp://0.0.0.0:4243', 'type': 'learner_with_action'}}}


ppo_session = Session(ppo_algorithm_config, deploy_config, params=trainer_params)
ppo_session.run(class_type=PPOTrainer, episode=EPISODE, duration=DURATION)
