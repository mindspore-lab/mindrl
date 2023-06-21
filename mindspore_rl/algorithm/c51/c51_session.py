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
c51 session.
"""
from mindspore import context
from mindspore import dtype as mstype

from mindspore_rl.algorithm.c51 import config
from mindspore_rl.core import Session
from mindspore_rl.utils.callback import (
    CheckpointCallback,
    EvaluateCallback,
    LossCallback,
)
from mindspore_rl.utils.utils import update_config


class CategoricalSession(Session):
    """c51 session"""

    def __init__(self, env_yaml=None, algo_yaml=None):
        update_config(config, env_yaml, algo_yaml)
        td_step = 1
        compute_type = (
            mstype.float16
            if context.get_context("device_target") in ["Ascend"]
            else mstype.float32
        )
        config.algorithm_config["policy_and_network"]["params"][
            "compute_type"
        ] = compute_type
        env = config.algorithm_config.get("collect_environment").get("type")(
            config.collect_env_params[
                config.algorithm_config.get("collect_environment").get("type").__name__
            ]
        )
        config.algorithm_config["replay_buffer"]["data_shape"] = [
            env.observation_space.shape,
            (1,),
            (td_step,),
            env.observation_space.shape,
            (td_step,),
        ]
        config.algorithm_config["replay_buffer"]["data_type"] = [
            env.observation_space.ms_dtype,
            env.action_space.ms_dtype,
            env.reward_space.ms_dtype,
            env.observation_space.ms_dtype,
            env.done_space.ms_dtype,
        ]
        config.trainer_params["data_shape"] = [
            env.observation_space.shape,
            (1,),
            (1,),
            env.observation_space.shape,
            (1,),
        ]
        config.trainer_params["data_type"] = [
            env.observation_space.ms_dtype,
            env.action_space.ms_dtype,
            env.reward_space.ms_dtype,
            env.observation_space.ms_dtype,
            env.done_space.ms_dtype,
        ]
        params = config.trainer_params
        loss_cb = LossCallback()
        ckpt_cb = CheckpointCallback(
            config.trainer_params.get("save_per_episode"),
            config.trainer_params.get("ckpt_path"),
        )
        eval_cb = EvaluateCallback(config.trainer_params.get("eval_per_episode"))
        cbs = [loss_cb, ckpt_cb, eval_cb]
        super().__init__(config.algorithm_config, None, params=params, callbacks=cbs)
