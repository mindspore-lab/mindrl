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
A3C training example.
"""

# pylint: disable=C0413
import argparse
import os

from mindspore import context

from mindspore_rl.algorithm.a3c import config
from mindspore_rl.algorithm.a3c.a3c_session import A3CSession
from mindspore_rl.algorithm.a3c.a3c_trainer import A3CTrainer

parser = argparse.ArgumentParser(description="MindSpore Reinforcement A3C")
parser.add_argument("--episode", type=int, default=1000, help="total episode numbers.")
parser.add_argument(
    "--device_target",
    type=str,
    default="Auto",
    choices=["Ascend", "GPU", "Auto"],
    help="Choose a device to run the a3c example(Default: Auto).",
)
parser.add_argument("--worker_num", type=int, default=4, help="worker number")
parser.add_argument(
    "--env_yaml",
    type=str,
    default="../env_yaml/CartPole-v0.yaml",
    help="Choose an environment yaml to update the a3c example(Default: CartPole-v0.yaml).",
)
parser.add_argument(
    "--algo_yaml",
    type=str,
    default=None,
    help="Choose an algo yaml to update the a3c example(Default: None).",
)
parser.add_argument(
    "--enable_distribute",
    type=bool,
    default=True,
    help="Train in distribute mode (Default: True).",
)
options, _ = parser.parse_known_args()


def train(episode=options.episode):
    """Train a3c"""
    if options.device_target != "Auto":
        context.set_context(device_target=options.device_target)
    if context.get_context("device_target") in ["Ascend"]:
        os.environ["GRAPH_OP_RUN"] = "1"
    context.set_context(mode=context.GRAPH_MODE)
    config.algorithm_config["actor"]["number"] = options.worker_num - 1
    ac_session = A3CSession(options.env_yaml, options.algo_yaml)
    ac_session.run(class_type=A3CTrainer, episode=episode)


if __name__ == "__main__":
    train()
