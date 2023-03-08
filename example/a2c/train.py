# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
A2C training example.
"""

#pylint: disable=C0413
import argparse
from mindspore_rl.algorithm.a2c import config
from mindspore_rl.algorithm.a2c.a2c_trainer import A2CTrainer
from mindspore_rl.algorithm.a2c.a2c_session import A2CSession
from mindspore import context
from mindspore import dtype as mstype


parser = argparse.ArgumentParser(description='MindSpore Reinforcement A2C')
parser.add_argument('--episode', type=int, default=10000, help='total episode numbers.')
parser.add_argument('--device_target', type=str, default='Auto', choices=['CPU', 'GPU', 'Ascend', 'Auto'],
                    help='Choose a device to run the ac example(Default: Auto).')
parser.add_argument('--precision_mode', type=str, default='fp32', choices=['fp32', 'fp16'],
                    help='Precision mode')
parser.add_argument('--env_yaml', type=str, default='../env_yaml/CartPole-v0.yaml',
                    help='Choose an environment yaml to update the a2c example(Default: CartPole-v0.yaml).')
parser.add_argument('--algo_yaml', type=str, default=None,
                    help='Choose an algo yaml to update the a2c example(Default: None).')
options, _ = parser.parse_known_args()


def train(episode=options.episode):
    if options.device_target != 'Auto':
        context.set_context(device_target=options.device_target)
    if context.get_context('device_target') in ['CPU', 'GPU']:
        context.set_context(enable_graph_kernel=True)
    context.set_context(mode=context.GRAPH_MODE)
    compute_type = mstype.float32 if options.precision_mode == 'fp32' else mstype.float16
    config.algorithm_config['policy_and_network']['params']['compute_type'] = compute_type
    if compute_type == mstype.float16 and options.device_target != 'Ascend':
        raise ValueError("Fp16 mode is supported by Ascend backend.")
    ac_session = A2CSession(options.env_yaml, options.algo_yaml)
    ac_session.run(class_type=A2CTrainer, episode=episode)

if __name__ == "__main__":
    train()
