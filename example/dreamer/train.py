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
Dreamer training example.
"""

#pylint: disable=C0413
import argparse
from mindspore_rl.algorithm.dreamer.dreamer_trainer import DreamerTrainer
from mindspore_rl.algorithm.dreamer.dreamer_session import DreamerSession
from mindspore import context

parser = argparse.ArgumentParser(description='MindSpore Reinforcement Dreamer')
parser.add_argument('--episode', type=int, default=1500, help='total episode numbers.')
parser.add_argument('--device_target', type=str, default='GPU', choices=['GPU'],
                    help='Choose a device to run the dreamer example(Default: Auto).')
parser.add_argument('--precision_mode', type=str, default='fp32', choices=['fp32'],
                    help='Precision mode')
parser.add_argument('--env_yaml', type=str, default='../env_yaml/walker_walk.yaml',
                    help='Choose an environment yaml to update the dreamer example(Default: walker_walk.yaml).')
parser.add_argument('--algo_yaml', type=str, default=None,
                    help='Choose an algo yaml to update the dreamer example(Default: None).')
options, _ = parser.parse_known_args()


def train(episode=options.episode):
    '''Dreamer train entry.'''

    context.set_context(mode=context.GRAPH_MODE, max_call_depth=100000)
    dreamer_session = DreamerSession(options.env_yaml, options.algo_yaml)
    dreamer_session.run(class_type=DreamerTrainer, episode=episode)


if __name__ == "__main__":
    train()
