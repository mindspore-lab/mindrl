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
TD3 eval example.
"""
import argparse
from mindspore_rl.algorithm.td3 import config
from mindspore_rl.algorithm.td3.td3_trainer import TD3Trainer
from mindspore_rl.algorithm.td3.td3_session import TD3Session
from mindspore import context
from mindspore import dtype as mstype

parser = argparse.ArgumentParser(description='MindSpore Reinforcement TD3')
parser.add_argument('--device_target', type=str, default='Auto', choices=['Ascend', 'CPU', 'GPU', 'Auto'],
                    help='Choose a device to run the td3 example(Default: Auto).')
parser.add_argument('--ckpt_path', type=str, default=None, help='The ckpt file in eval.')
parser.add_argument('--eval_episodes', type=int, default=10, help='Total episodes for evaluation.')
parser.add_argument('--precision_mode', type=str, default='fp32', choices=['fp32', 'fp16'],
                    help='Precision mode')
parser.add_argument('--env_yaml', type=str, default='../env_yaml/HalfCheetah-v2.yaml',
                    help='Choose an environment yaml to update the td3 example(Default: HalfCheetah-v2.yaml).')
parser.add_argument('--algo_yaml', type=str, default=None,
                    help='Choose an algo yaml to update the td3 example(Default: None).')
args = parser.parse_args()


def td3_eval():
    if args.device_target != 'Auto':
        context.set_context(device_target=args.device_target)
    context.set_context(mode=context.GRAPH_MODE, max_call_depth=100000)
    compute_type = mstype.float32 if args.precision_mode == 'fp32' else mstype.float16
    config.algorithm_config.get('policy_and_network').get('params')['compute_type'] = compute_type
    config.summary_config['mindinsight_on'] = False
    td3_session = TD3Session(args.env_yaml, args.algo_yaml)
    config.trainer_params.update({'ckpt_path': args.ckpt_path, 'eval_episodes': args.eval_episodes})
    td3_session.run(class_type=TD3Trainer, is_train=False)

if __name__ == "__main__":
    td3_eval()
