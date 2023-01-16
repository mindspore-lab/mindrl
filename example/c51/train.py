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
C51 training example.
"""
import argparse
from mindspore import context
from mindspore_rl.algorithm.c51.c51_trainer import CategoricalDQNTrainer
from mindspore_rl.algorithm.c51.c51_session import CategoricalSession


parser = argparse.ArgumentParser(description='MindSpore Reinforcement c51')
parser.add_argument('--episode', type=int, default=600, help='total episode numbers.')
parser.add_argument('--device_target', type=str, default='CPU', choices=['CPU'],
                    help='Choose a device to run the c51 example(Default: CPU).')
parser.add_argument('--env_yaml', type=str, default='../env_yaml/CartPole-v0.yaml',
                    help='Choose an environment yaml to update the c51 example(Default: CartPole-v0.yaml).')
parser.add_argument('--algo_yaml', type=str, default=None,
                    help='Choose an algo yaml to update the c51 example(Default: None).')
options, _ = parser.parse_known_args()


def train(episode=options.episode):
    """start to train c51 algorithm"""

    if options.device_target != 'Auto':
        context.set_context(device_target=options.device_target)
    if context.get_context('device_target') in ['CPU']:
        context.set_context(enable_graph_kernel=True)

    context.set_context(mode=context.GRAPH_MODE)
    c51_session = CategoricalSession(options.env_yaml, options.algo_yaml)
    c51_session.run(class_type=CategoricalDQNTrainer, episode=episode)


if __name__ == "__main__":
    train()
