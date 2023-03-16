# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
PPO training example.
"""

#pylint: disable=C0413
import os
import argparse
from mindspore_rl.algorithm.ppo import config
from mindspore_rl.algorithm.ppo.ppo_trainer import PPOTrainer
from mindspore_rl.algorithm.ppo.ppo_session import PPOSession
from mindspore import context
from mindspore import dtype as mstype

parser = argparse.ArgumentParser(description='MindSpore Reinforcement PPO')
parser.add_argument('--episode', type=int, default=650, help='total episode numbers.')
parser.add_argument('--device_target', type=str, default='Auto', choices=['Ascend', 'CPU', 'GPU', 'Auto'],
                    help='Choose a device to run the ppo example(Default: Auto).')
parser.add_argument('--precision_mode', type=str, default='fp32', choices=['fp32', 'fp16'],
                    help='Precision mode')
parser.add_argument('--env_yaml', type=str, default='../env_yaml/HalfCheetah-v2.yaml',
                    help='Choose an environment yaml to update the ppo example(Default: HalfCheetah-v2.yaml).')
parser.add_argument('--algo_yaml', type=str, default=None,
                    help='Choose an algo yaml to update the ppo example(Default: None).')
parser.add_argument('--enable_distribute', type=bool, default=False,
                    help='Train in distribute mode (Default: False).')
parser.add_argument('--worker_num', type=int, default=2, help='Worker num (Default: 2).')
options, _ = parser.parse_known_args()


def train(episode=options.episode):
    '''PPO train entry.'''
    if options.device_target != 'Auto':
        context.set_context(device_target=options.device_target)
    if context.get_context('device_target') in ['CPU']:
        context.set_context(enable_graph_kernel=True)
    if context.get_context('device_target') in ['Ascend']:
        os.environ['GRAPH_OP_RUN'] = "1"

    compute_type = mstype.float32 if options.precision_mode == 'fp32' else mstype.float16
    config.algorithm_config['policy_and_network']['params']['compute_type'] = compute_type
    if compute_type == mstype.float16 and options.device_target != 'Ascend':
        raise ValueError("Fp16 mode is supported by Ascend backend.")
    duration = config.trainer_params.get("duration")
    context.set_context(mode=context.GRAPH_MODE, max_call_depth=100000)
    is_distribte = options.enable_distribute
    if is_distribte:
        config.deploy_config['worker_num'] = options.worker_num
    ppo_session = PPOSession(options.env_yaml, options.algo_yaml, is_distribte)
    ppo_session.run(class_type=PPOTrainer, episode=episode, duration=duration)


if __name__ == "__main__":
    train()
