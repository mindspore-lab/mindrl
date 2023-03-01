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
"""QMIX Train"""

import argparse
from mindspore_rl.algorithm.qmix import config
from mindspore_rl.algorithm.qmix.qmix_session import QMIXSession
from mindspore import context
from mindspore import dtype as mstype

parser = argparse.ArgumentParser(description='MindSpore Reinforcement QMIX')
parser.add_argument('--episode', type=int, default=400000, help='total episode numbers.')
parser.add_argument('--device_target', type=str, default='Auto', choices=['Ascend', 'CPU', 'GPU', 'Auto'],
                    help='Choose a device to run the qmix example(Default: Auto).')
parser.add_argument('--precision_mode', type=str, default='fp32', choices=['fp32', 'fp16'],
                    help='Precision mode')
parser.add_argument('--env_yaml', type=str, default='../env_yaml/simple_spread.yaml',
                    help='Choose an environment yaml to update the qmix example(Default: simple_spread.yaml).')
parser.add_argument('--algo_yaml', type=str, default=None,
                    help='Choose an algo yaml to update the qmix example(Default: None).')
options, _ = parser.parse_known_args()



def train(episode=options.episode):
    """start to train qmix algorithm"""
    if options.device_target != 'Auto':
        context.set_context(device_target=options.device_target)
    context.set_context(mode=context.GRAPH_MODE)
    compute_type = mstype.float32 if options.precision_mode == 'fp32' else mstype.float16
    config.algorithm_config['policy_and_network']['params']['compute_type'] = compute_type
    if compute_type == mstype.float16 and options.device_target != 'Ascend':
        raise ValueError("Fp16 mode is supported by Ascend backend.")
    qmix_session = QMIXSession(options.env_yaml, options.algo_yaml)
    env_name = qmix_session.msrl.collect_environment.__class__.__name__
    if env_name == "MultiAgentParticleEnvironment":
        from mindspore_rl.algorithm.qmix.qmix_mpe_trainer import QMIXTrainer
    elif env_name == "StarCraft2Environment":
        from mindspore_rl.algorithm.qmix.qmix_smac_trainer import QMIXTrainer
    else:
        raise ValueError(f"The input environment {env_name} does not support yet")
    qmix_session.run(class_type=QMIXTrainer, episode=episode)


if __name__ == "__main__":
    train()
