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
SAC eval example.
"""
import argparse
from src.sac_trainer import SACTrainer
from mindspore_rl.agent.sac import SACConfig as config
from mindspore_rl.core import Session
from mindspore import context

parser = argparse.ArgumentParser(description='MindSpore Reinforcement SAC')
parser.add_argument('--device_target', type=str, default='Auto', choices=['Ascend', 'CPU', 'GPU', 'Auto'],
                    help='Choose a device to run the sac example(Default: Auto).')
parser.add_argument('--ckpt_path', type=str, default=None, help='The ckpt file in eval.')
args = parser.parse_args()


def sac_eval():
    if args.device_target != 'Auto':
        context.set_context(device_target=args.device_target)
    context.set_context(mode=context.GRAPH_MODE, max_call_depth=100000)
    config.trainer_params.update({'ckpt_path': args.ckpt_path})
    sac_session = Session(config.algorithm_config)
    sac_session.run(class_type=SACTrainer, is_train=False, params=config.trainer_params)

if __name__ == "__main__":
    sac_eval()
