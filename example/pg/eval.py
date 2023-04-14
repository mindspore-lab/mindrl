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
PG eval example.
"""
import argparse

from mindspore import context

from mindspore_rl.algorithm.pg import config
from mindspore_rl.algorithm.pg.pg_session import PGSession
from mindspore_rl.algorithm.pg.pg_trainer import PGTrainer

parser = argparse.ArgumentParser(description="MindSpore Reinforcement PG")
parser.add_argument(
    "--device_target",
    type=str,
    default="Auto",
    choices=["Ascend", "CPU", "GPU", "Auto"],
    help="Choose a device to run the pg example(Default: Auto).",
)
parser.add_argument(
    "--ckpt_path", type=str, default=None, help="The ckpt file in eval."
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
args = parser.parse_args()


def pg_eval():
    """Eval function"""
    if args.device_target != "Auto":
        context.set_context(device_target=args.device_target)
    context.set_context(mode=context.GRAPH_MODE)
    pg_session = PGSession(args.env_yaml, args.algo_yaml)
    config.trainer_params.update({"ckpt_path": args.ckpt_path})
    pg_session.run(class_type=PGTrainer, is_train=False)


if __name__ == "__main__":
    pg_eval()
