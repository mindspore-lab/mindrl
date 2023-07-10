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
DQN session.
"""
from mindspore import context
from mindspore import dtype as mstype

from mindspore_rl.algorithm.dqn import config
from mindspore_rl.core import Session
from mindspore_rl.utils.callback import (
    CheckpointCallback,
    EvaluateCallback,
    LossCallback,
    TimeCallback,
    WandBCallback
)
from mindspore_rl.utils.utils import update_config


class DQNSession(Session):
    """DQN session"""

    def __init__(self, env_yaml=None, algo_yaml=None):
        update_config(config, env_yaml, algo_yaml)
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
            (1,),
            env.observation_space.shape,
        ]
        config.algorithm_config["replay_buffer"]["data_type"] = [
            env.observation_space.ms_dtype,
            env.action_space.ms_dtype,
            env.reward_space.ms_dtype,
            env.observation_space.ms_dtype,
        ]
        params = config.trainer_params
        loss_cb = LossCallback()
        ckpt_cb = CheckpointCallback(
            config.trainer_params.get("save_per_episode"),
            config.trainer_params.get("ckpt_path"),
        )
        eval_cb = EvaluateCallback(config.trainer_params.get("eval_per_episode"))
        time_cb = TimeCallback()

        # WandB project details
        authorization_key = "0c108d032f6db708d53d279c92e275b9833adee6"
        project_name = "MSRL_auto_test"
        name = "dqn"
        log_config = {
                    "learners_params": config.learner_params,
                    "trainer_params": config.trainer_params,
                    "collect_env_params": config.collect_env_params,
                    "eval_env_params": config.eval_env_params,
                    "policy_params": config.policy_params,
                    "algorithm_config": config.algorithm_config,  
        } # configuration of cofig to log into WandB

        wandb_cb = WandBCallback(authorization_key=authorization_key, project=project_name, 
                                 name=name, config=log_config)


        cbs = [loss_cb, ckpt_cb, eval_cb, time_cb, wandb_cb]
        super().__init__(config.algorithm_config, None, params=params, callbacks=cbs)
