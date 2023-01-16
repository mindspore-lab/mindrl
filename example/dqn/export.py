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
"""export"""
import argparse
import os
import gym
import mindspore as ms
from mindspore import dtype as mstype
from mindspore import Tensor, export, context, ops
from mindspore import load_checkpoint
from mindspore_rl.algorithm.dqn.config import policy_params
from mindspore_rl.algorithm.dqn.dqn import DQNPolicy

parser = argparse.ArgumentParser(description='MindSpore Reinforcement DQN export')
parser.add_argument('--ckpt_file', type=str, default='./scripts/ckpt/policy_net/policy_net_1000.ckpt',
                    help='ckpt file location.')
options, _ = parser.parse_known_args()


def run_export():
    """export"""
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    env.seed(1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    compute_type = mstype.float16 if context.get_context('device_target') in ['Ascend'] else mstype.float32
    params = policy_params
    params['state_space_dim'] = state_dim
    params['action_space_dim'] = action_dim
    params['compute_type'] = compute_type
    dqn_net = DQNPolicy(params).evaluate_policy
    load_checkpoint(options.ckpt_file, net=dqn_net)
    state = env.reset()
    state = Tensor(state, ms.float32)
    expand_dims = ops.ExpandDims()
    state = expand_dims(state, 0)
    if not os.path.exists("./scripts/onnx/policy_net"):
        os.makedirs("./scripts/onnx/policy_net")
    export(dqn_net, state, file_name="./scripts/onnx/policy_net/policy_net_1000", file_format="ONNX")
    print("export ONNX file at ./scripts/onnx/policy_net/policy_net_1000.onnx")
    env.close()


if __name__ == '__main__':
    run_export()
