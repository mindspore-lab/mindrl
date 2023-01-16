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
CQL training example.
"""

#pylint: disable=C0413
import argparse
from mindspore_rl.algorithm.cql.cql_trainer import CQLTrainer
from mindspore_rl.algorithm.cql.cql_session import CQLSession
from mindspore import context

parser = argparse.ArgumentParser(description='MindSpore Reinforcement Offline RL: CQL')
parser.add_argument('--episode', type=int, default=1000000, help='total episode(step) numbers.')
parser.add_argument('--device_target', type=str, default='Auto', choices=['CPU', 'GPU', 'Auto'],
                    help='Choose a device to run the CQL example(Default: Auto).')
parser.add_argument('--env_yaml', type=str, default='../env_yaml/hopper-medium-expert-v0.yaml',
                    help='Choose an environment yaml to update (Default: hopper-medium-expert-v0.yaml).')
parser.add_argument('--algo_yaml', type=str, default=None,
                    help='Choose an algo yaml to update the CQL example(Default: None).')
options, _ = parser.parse_known_args()


def train(episode=options.episode):
    if options.device_target != 'Auto':
        context.set_context(device_target=options.device_target)
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(enable_graph_kernel=True)
    cql_session = CQLSession(options.env_yaml, options.algo_yaml)
    cql_session.run(class_type=CQLTrainer, episode=episode)

if __name__ == "__main__":
    train()
