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
"""Double DQN Config"""

# pylint: disable=C0413
import argparse

from mindspore import context
from mindspore import dtype as mstype

from mindspore_rl.algorithm.double_dqn import config
from mindspore_rl.algorithm.double_dqn.double_dqn_session import DoubleDQNSession
from mindspore_rl.algorithm.dqn.dqn_trainer import DQNTrainer

parser = argparse.ArgumentParser(description="MindSpore Reinforcement DQN")
parser.add_argument("--episode", type=int, default=650, help="total episode numbers.")
parser.add_argument(
    "--device_target",
    type=str,
    default="Auto",
    choices=["Ascend", "CPU", "GPU", "Auto"],
    help="Choose a device to run the dqn example(Default: Auto).",
)
parser.add_argument(
    "--precision_mode",
    type=str,
    default="fp32",
    choices=["fp32", "fp16"],
    help="Precision mode",
)
parser.add_argument(
    "--env_yaml",
    type=str,
    default="../env_yaml/CartPole-v0.yaml",
    help="Choose an environment yaml to update the dqn example(Default: CartPole-v0.yaml).",
)
parser.add_argument(
    "--algo_yaml",
    type=str,
    default=None,
    help="Choose an algo yaml to update the dqn example(Default: None).",
)
options, _ = parser.parse_known_args()


def train(episode=options.episode):
    """start to train dqn algorithm"""
    if options.device_target != "Auto":
        context.set_context(device_target=options.device_target)
    if context.get_context("device_target") in ["CPU"]:
        context.set_context(enable_graph_kernel=True)
    context.set_context(mode=context.PYNATIVE_MODE)
    compute_type = (
        mstype.float32 if options.precision_mode == "fp32" else mstype.float16
    )
    config.algorithm_config["policy_and_network"]["params"][
        "compute_type"
    ] = compute_type
    if compute_type == mstype.float16 and options.device_target != "Ascend":
        raise ValueError("Fp16 mode is supported by Ascend backend.")
    ddqn_session = DoubleDQNSession(options.env_yaml, options.algo_yaml)
    ddqn_session.run(class_type=DQNTrainer, episode=episode)


if __name__ == "__main__":
    train()
