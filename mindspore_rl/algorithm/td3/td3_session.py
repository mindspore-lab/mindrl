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
TD3 session.
"""
import os
import time

from mindspore_rl.algorithm.td3 import config
from mindspore_rl.core import Session
from mindspore_rl.utils.callback import (
    CheckpointCallback,
    EvaluateCallback,
    LossCallback,
    TimeCallback,
)
from mindspore_rl.utils.utils import update_config

from .td3_summary import EvalRecordCallback, SummaryCallback


class TD3Session(Session):
    """TD3 eval session."""

    def __init__(self, env_yaml=None, algo_yaml=None, cbs=None):
        update_config(config, env_yaml, algo_yaml)
        # Collect environment information and update replay buffer shape/dtype.
        # So the algorithm could change the environment type without aware of replay buffer schema.
        env_config = config.algorithm_config.get("collect_environment")

        env = env_config["type"](env_config.get("params"))
        obs_shape, obs_dtype = (
            env.observation_space.shape,
            env.observation_space.ms_dtype,
        )
        action_shape, action_dtype = env.action_space.shape, env.action_space.ms_dtype
        _, reward_dtype = env.reward_space.shape, env.reward_space.ms_dtype
        _, done_dtype = env.done_space.shape, env.done_space.ms_dtype
        config.learner_params["action_boundary"] = env.action_space.boundary

        replay_buffer_config = config.algorithm_config.get("replay_buffer")
        replay_buffer_config["data_shape"] = [
            obs_shape,
            action_shape,
            (1,),
            obs_shape,
            (1,),
        ]
        replay_buffer_config["data_type"] = [
            obs_dtype,
            action_dtype,
            reward_dtype,
            obs_dtype,
            done_dtype,
        ]

        loss_cb = LossCallback()
        ckpt_cb = CheckpointCallback(
            config.trainer_params.get("save_per_episode"),
            config.trainer_params.get("ckpt_path"),
        )
        time_cb = TimeCallback()
        cbs = [loss_cb, ckpt_cb, time_cb]

        if config.summary_config.get("mindinsight_on"):
            summary_base_dir = config.summary_config.get("base_dir")
            if not os.path.exists(summary_base_dir):
                os.mkdir(summary_base_dir)
            output_dir = os.path.join(
                summary_base_dir,
                time.strftime("%Y%m%d%H%I%S", time.localtime(time.time())),
            )
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            eval_record_cb = EvalRecordCallback(
                config.trainer_params.get("num_eval_episode")
            )
            res_cb = SummaryCallback(
                config.summary_config.get("collect_interval"),
                config.trainer_params.get("num_eval_episode"),
                output_dir,
            )
            cbs.append(eval_record_cb)
            cbs.append(res_cb)
        else:
            eval_cb = EvaluateCallback(config.trainer_params.get("num_eval_episode"))
            cbs.append(eval_cb)
        params = config.trainer_params
        super().__init__(config.algorithm_config, None, params=params, callbacks=cbs)
