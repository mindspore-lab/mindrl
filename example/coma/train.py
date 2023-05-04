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
COMA training example.
"""

import argparse

from mindspore import context
from mindspore import dtype as mstype

from mindspore_rl.algorithm.coma import config
from mindspore_rl.algorithm.coma.coma_session import COMASession
from mindspore_rl.algorithm.coma.coma_trainer import COMATrainer

parser = argparse.ArgumentParser(description="MindSpore Reinforcement COMA")
parser.add_argument("--episode", type=int, default=1000, help="total episode numbers.")
parser.add_argument(
    "--device_target",
    type=str,
    default="Auto",
    choices=["Ascend", "CPU", "GPU", "Auto"],
    help="Choose a device to run the coma example(Default: Auto).",
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
    default="../env_yaml/2s3z.yaml",
    help="Choose an environment yaml to update the coma example(Default: 2s3z.yaml).",
)
parser.add_argument(
    "--algo_yaml",
    type=str,
    default=None,
    help="Choose an algo yaml to update the coma example(Default: None).",
)

options, _ = parser.parse_known_args()


def train(episode=options.episode):
    """COMA train entry."""
    if options.device_target != "Auto":
        context.set_context(device_target=options.device_target)
    if context.get_context("device_target") in ["CPU", "GPU"]:
        context.set_context(enable_graph_kernel=False)
    context.set_context(mode=context.PYNATIVE_MODE, save_graphs=False)
    compute_type = (
        mstype.float32 if options.precision_mode == "fp32" else mstype.float16
    )
    config.algorithm_config.get("policy_and_network").get("params")[
        "compute_type"
    ] = compute_type
    if compute_type == mstype.float16 and options.device_target != "Ascend":
        raise ValueError("Fp16 mode is supported by Ascend backend.")
    coma_session = COMASession(options.env_yaml, options.algo_yaml)
    coma_session.run(class_type=COMATrainer, episode=episode)


if __name__ == "__main__":
    train()
