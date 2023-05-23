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
Dreamer session.
"""

from mindspore_rl.algorithm.dreamer import config
from mindspore_rl.core import Session
from mindspore_rl.utils.callback import EvaluateCallback, LossCallback, TimeCallback
from mindspore_rl.utils.utils import update_config


class DreamerSession(Session):
    """Dreamer session"""

    def __init__(self, env_yaml=None, algo_yaml=None):
        update_config(config, env_yaml, algo_yaml)
        env_config = config.algorithm_config.get("collect_environment")
        env = env_config.get("type")(
            env_config.get("params")[env_config.get("type").__name__]
        )
        env.close()
        episode_limits = config.all_params.get("episode_limits")
        action_repeat = config.all_params.get("action_repeat")
        obs_shape, _ = (
            env.observation_space.shape,
            env.observation_space.ms_dtype,
        )
        action_shape, _ = env.action_space.shape, env.action_space.ms_dtype
        replay_buffer_config = config.algorithm_config.get("replay_buffer")
        replay_buffer_config["data_shape"] = [
            (int(episode_limits / action_repeat) + 1,) + obs_shape,
            (int(episode_limits / action_repeat) + 1,) + action_shape,
            (int(episode_limits / action_repeat + 1),) + (1,),
            (int(episode_limits / action_repeat) + 1,) + (1,),
        ]
        replay_buffer_config["data_type"] = [
            config.all_params["dtype"],
            config.all_params["dtype"],
            config.all_params["dtype"],
            config.all_params["dtype"],
        ]
        loss_cb = LossCallback()
        eval_cb = EvaluateCallback(50)
        time_cb = TimeCallback()
        cbs = [time_cb, loss_cb, eval_cb]
        params = config.trainer_params
        super().__init__(config.algorithm_config, None, params=params, callbacks=cbs)
