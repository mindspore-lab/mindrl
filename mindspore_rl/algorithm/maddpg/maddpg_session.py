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
MADDPG session.
"""
import mindspore as ms
from gym import spaces

from mindspore_rl.algorithm.maddpg import config
from mindspore_rl.core import Session
from mindspore_rl.utils.utils import update_config


class MADDPGSession(Session):
    """MADDPG session"""

    def __init__(self, env_yaml=None, algo_yaml=None):
        update_config(config, env_yaml, algo_yaml)
        env_config = config.algorithm_config.get("collect_environment")
        env = env_config.get("type")(
            env_config.get("params")[env_config.get("type").__name__]
        )
        if env_config.get("continous_actions"):
            assert isinstance(env.action_space[0], spaces.Box)
        env.close()
        agent_num = config.NUM_AGENT
        obs_shape, obs_dtype = (
            env.observation_space.shape,
            env.observation_space.ms_dtype,
        )
        action_shape, _ = env.action_space.shape, env.action_space.ms_dtype
        reward_shape, reward_dtype = env.reward_space.shape, env.reward_space.ms_dtype
        buffer_config = config.algorithm_config.get("replay_buffer")
        buffer_config["data_shape"] = [
            (*obs_shape,),
            (*action_shape,),
            (*reward_shape,),
            (*obs_shape,),
            (agent_num, 1),
        ]
        buffer_config["data_type"] = [
            obs_dtype,
            ms.float32,
            reward_dtype,
            obs_dtype,
            ms.bool_,
        ]
        params = config.trainer_params
        super().__init__(config.algorithm_config, None, params=params, callbacks=None)
