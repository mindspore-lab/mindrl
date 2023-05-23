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
COMA session.
"""
from mindspore_rl.algorithm.coma import config
from mindspore_rl.algorithm.qmix.qmix_session import (
    StepInfoCollectCallback,
    StepInfoEvalCallback,
)
from mindspore_rl.core import Session
from mindspore_rl.utils.callback import CheckpointCallback, LossCallback
from mindspore_rl.utils.utils import update_config


class COMASession(Session):
    """COMA Session"""

    def __init__(self, env_yaml=None, algo_yaml=None):
        update_config(config, env_yaml, algo_yaml)
        env_config = config.algorithm_config.get("collect_environment")
        env = env_config.get("type")(
            env_config.get("params")[env_config.get("type").__name__]
        )
        num_agent = env.config.get("num_agent")
        global_obs_dim = env.config.get("global_observation_dim")
        action_dim = env.action_space.num_values
        num_env = 8
        params = config.trainer_params
        obs_shape, obs_type = (
            num_env,
            num_agent,
            env.observation_space.shape[-1],
        ), env.observation_space.ms_dtype
        state, state_type = (num_env, global_obs_dim), env.observation_space.ms_dtype
        action_shape, action_type = (num_env, num_agent), env.action_space.ms_dtype
        avail_action_shape, avail_action_type = (
            (
                num_env,
                num_agent,
                action_dim,
            ),
            env.action_space.ms_dtype,
        )
        reward_shape, reward_type = (
            (num_env, env.reward_space.shape[-1]),
            env.reward_space.ms_dtype,
        )
        done_shape, done_type = (
            (num_env, env.done_space.shape[-1]),
            env.done_space.ms_dtype,
        )
        onehot_shape, onehot_type = (
            (num_env, num_agent, action_dim),
            env.observation_space.ms_dtype,
        )
        filled_shape, filled_type = (
            (num_env, env.done_space.shape[-1]),
            env.done_space.ms_dtype,
        )
        replay_buffer_config = config.algorithm_config.get("replay_buffer")
        replay_buffer_config["data_shape"] = [
            obs_shape,
            state,
            action_shape,
            avail_action_shape,
            reward_shape,
            done_shape,
            onehot_shape,
            filled_shape,
        ]
        replay_buffer_config["data_type"] = [
            obs_type,
            state_type,
            action_type,
            avail_action_type,
            reward_type,
            done_type,
            onehot_type,
            filled_type,
        ]

        ckpt_cb = CheckpointCallback(
            config.trainer_params.get("save_per_episode"),
            config.trainer_params.get("ckpt_path"),
        )
        loss_cb = LossCallback()
        step_info_train_cb = StepInfoCollectCallback(100)
        step_info_eval_cb = StepInfoEvalCallback(200, 20)
        cbs = [ckpt_cb, step_info_train_cb, step_info_eval_cb, loss_cb]

        super().__init__(config.algorithm_config, None, params=params, callbacks=cbs)
