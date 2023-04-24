# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
PPO session.
"""
from mindspore_rl.algorithm.ppo import config
from mindspore_rl.core import Session
from mindspore_rl.utils.callback import (
    CheckpointCallback,
    EvaluateCallback,
    LossCallback,
    TimeCallback,
)
from mindspore_rl.utils.utils import update_config


class PPOSession(Session):
    """PPO session"""

    def __init__(self, env_yaml=None, algo_yaml=None, is_distribution=None):
        update_config(config, env_yaml, algo_yaml)
        env_config = config.algorithm_config.get("collect_environment")
        env = env_config.get("type")(env_config.get("params"))
        env_num = config.algorithm_config.get("collect_environment").get("number")
        obs_shape, obs_dtype = (
            env.observation_space.shape,
            env.observation_space.ms_dtype,
        )
        action_shape, action_dtype = env.action_space.shape, env.action_space.ms_dtype
        _, reward_dtype = env.reward_space.shape, env.reward_space.ms_dtype
        mu_shape, mu_dtype = action_shape, action_dtype
        sigma_shape, sigma_dtype = action_shape, action_dtype
        replay_buffer_config = config.algorithm_config.get("replay_buffer")
        replay_buffer_config["data_shape"] = [
            (env_num, obs_shape[-1]),
            (env_num, action_shape[-1]),
            (env_num, 1),
            (env_num, obs_shape[-1]),
            (env_num, mu_shape[-1]),
            (env_num, sigma_shape[-1]),
        ]
        replay_buffer_config["data_type"] = [
            obs_dtype,
            action_dtype,
            reward_dtype,
            obs_dtype,
            mu_dtype,
            sigma_dtype,
        ]
        loss_cb = LossCallback()
        ckpt_cb = CheckpointCallback(
            config.trainer_params.get("num_save_episode"),
            config.trainer_params.get("ckpt_path"),
        )
        eval_cb = EvaluateCallback(config.trainer_params.get("num_eval_episode"))
        time_cb = TimeCallback()
        cbs = [loss_cb, ckpt_cb, eval_cb, time_cb]
        params = config.trainer_params
        deploy_config = None
        if is_distribution:
            deploy_config = config.deploy_config
        super().__init__(
            config.algorithm_config, deploy_config, params=params, callbacks=cbs
        )
