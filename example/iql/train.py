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
IQL training example.
"""

#pylint: disable=C0413
import argparse
from mindspore_rl.algorithm.iql.iql_trainer import IQLTrainer
from mindspore_rl.algorithm.iql.iql_session import IQLSession
from mindspore import context

parser = argparse.ArgumentParser(description='MindSpore Reinforcement IQL')
parser.add_argument('--episode', type=int, default=500, help='total episode numbers.')
parser.add_argument('--device_target', type=str, default='Auto', choices=['CPU', 'GPU', 'Auto'],
                    help='Choose a device to run the iql example(Default: Auto).')
parser.add_argument('--env_yaml', type=str, default='../env_yaml/walker2d-medium-v2.yaml',
                    help='Choose an environment yaml to update (Default: walker2d-medium-v2.yaml).')
parser.add_argument('--algo_yaml', type=str, default=None,
                    help='Choose an algo yaml to update the iql example(Default: None).')
options, _ = parser.parse_known_args()


def train(episode=options.episode):
    """start to train iql algorithm"""
    if options.device_target != 'Auto':
        context.set_context(device_target=options.device_target)
    context.set_context(enable_graph_kernel=True)
    context.set_context(mode=context.GRAPH_MODE)
    iql_session = IQLSession(options.env_yaml, options.algo_yaml)
    iql_session.run(class_type=IQLTrainer, episode=episode)


if __name__ == "__main__":
    train()
