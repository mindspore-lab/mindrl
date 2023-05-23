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
DDPG session.
"""
from mindspore_rl.algorithm.ddpg import config
from mindspore_rl.core import Session
from mindspore_rl.utils.callback import (
    CheckpointCallback,
    EvaluateCallback,
    LossCallback,
    TimeCallback,
)
from mindspore_rl.utils.utils import update_config


class DDPGSession(Session):
    """DDPG session"""

    def __init__(self, env_yaml=None, algo_yaml=None):
        update_config(config, env_yaml, algo_yaml)
        # Collect environment information and update replay buffer shape/dtype.
        # So the algorithm could change the environment type without aware of replay buffer schema.
        env_config = config.algorithm_config.get("collect_environment")
        env = env_config.get("type")(
            env_config.get("params")[env_config.get("type").__name__]
        )
        obs_shape, obs_dtype = (
            env.observation_space.shape,
            env.observation_space.ms_dtype,
        )
        action_shape, action_dtype = env.action_space.shape, env.action_space.ms_dtype
        reward_shape, reward_dtype = (1,), env.reward_space.ms_dtype
        done_shape, done_dtype = (1,), env.done_space.ms_dtype

        replay_buffer_config = config.algorithm_config.get("replay_buffer")
        replay_buffer_config["data_shape"] = [
            obs_shape,
            action_shape,
            reward_shape,
            obs_shape,
            done_shape,
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
        eval_cb = EvaluateCallback(config.trainer_params.get("num_eval_episode"))
        time_cb = TimeCallback()
        cbs = [loss_cb, ckpt_cb, eval_cb, time_cb]
        params = config.trainer_params
        super().__init__(config.algorithm_config, None, params=params, callbacks=cbs)
