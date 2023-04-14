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
PG training example.
"""

# pylint: disable=C0413
import argparse

from mindspore import context
from mindspore import dtype as mstype

from mindspore_rl.algorithm.pg import config
from mindspore_rl.algorithm.pg.pg_session import PGSession
from mindspore_rl.algorithm.pg.pg_trainer import PGTrainer

parser = argparse.ArgumentParser(description="MindSpore Reinforcement PG")
parser.add_argument("--episode", type=int, default=600, help="total episode numbers.")
parser.add_argument(
    "--device_target",
    type=str,
    default="Auto",
    choices=["Ascend", "CPU", "GPU", "Auto"],
    help="Choose a device to run the pg example(Default: Auto).",
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
    help="Choose an environment yaml to update the pg example(Default: CartPole-v0.yaml).",
)
parser.add_argument(
    "--algo_yaml",
    type=str,
    default=None,
    help="Choose an algo yaml to update the pg example(Default: None).",
)
options, _ = parser.parse_known_args()


def train(episode=options.episode):
    """PG train entry."""
    if options.device_target != "Auto":
        context.set_context(device_target=options.device_target)
    compute_type = (
        mstype.float32 if options.precision_mode == "fp32" else mstype.float16
    )
    config.algorithm_config["policy_and_network"]["params"][
        "compute_type"
    ] = compute_type
    if compute_type == mstype.float16 and options.device_target != "Ascend":
        raise ValueError("Fp16 mode is supgrted by Ascend backend.")
    duration = config.trainer_params.get("duration")
    context.set_context(mode=context.GRAPH_MODE, max_call_depth=100000)

    pg_session = PGSession(options.env_yaml, options.algo_yaml)
    pg_session.run(class_type=PGTrainer, episode=episode, duration=duration)


if __name__ == "__main__":
    train()
