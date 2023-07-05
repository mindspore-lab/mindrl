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
IMAPALA training example.
"""

import argparse

from mindspore import context

from mindspore_rl.algorithm.impala import config
from mindspore_rl.algorithm.impala.impala_session import IMPALASession
from mindspore_rl.algorithm.impala.impala_trainer import IMPALATrainer

parser = argparse.ArgumentParser(description="MindSpore Reinforcement IMPALA")
parser.add_argument("--episode", type=int, default=1000, help="total episode numbers.")
parser.add_argument("--worker_num", type=int, default=4, help="worker number")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
parser.add_argument(
    "--env_yaml",
    type=str,
    default="../env_yaml/CartPole-v0.yaml",
    help="Choose an environment yaml to update the IMPALA example(Default: CartPole-v0.yaml).",
)
parser.add_argument(
    "--algo_yaml",
    type=str,
    default=None,
    help="Choose an algo yaml to update the impala example(Default: None).",
)
options, _ = parser.parse_known_args()


def train(episode=options.episode):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    config.algorithm_config["actor"]["number"] = options.worker_num - 1
    config.policy_params["batch_size"] = options.batch_size
    ac_session = IMPALASession(options.env_yaml, options.algo_yaml)
    ac_session.run(class_type=IMPALATrainer, episode=episode)


if __name__ == "__main__":
    train()
