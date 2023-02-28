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
MAPPO session.
"""
from mindspore_rl.core import Session
from mindspore_rl.utils.utils import update_config
from mindspore_rl.utils.callback import LossCallback, TimeCallback
from mindspore_rl.algorithm.mappo import config


class MAPPOSession(Session):
    '''MAPPO session'''
    def __init__(self, env_yaml=None, algo_yaml=None):
        update_config(config, env_yaml, algo_yaml)
        env_config = config.algorithm_config.get('collect_environment')
        env = env_config.get('type')(env_config.get('params'))
        env_num = config.algorithm_config.get('collect_environment').get('number')
        env.close()
        agent_num = config.NUM_AGENT
        _, obs_dtype = env.observation_space.shape, env.observation_space.ms_dtype
        _, action_dtype = env.action_space.shape, env.action_space.ms_dtype
        local_buffer_config = config.algorithm_config.get('replay_buffer').get('local_replaybuffer')
        local_buffer_config['data_shape'] = [(env_num, agent_num * 6), (env_num, 1, 64),
                                             (env_num, 1, 64), (env_num, 1), (env_num, 1),
                                             (env_num, 1), (env_num, 1), (env_num, 1)]
        local_buffer_config['data_type'] = [obs_dtype, obs_dtype, obs_dtype, obs_dtype, action_dtype,
                                            obs_dtype, obs_dtype, obs_dtype]
        global_buffer_config = config.algorithm_config.get('replay_buffer').get('global_replaybuffer')
        global_buffer_config['data_shape'] = [(env_num, agent_num * agent_num * 6)]
        global_buffer_config['data_type'] = [obs_dtype]
        loss_cb = LossCallback()
        time_cb = TimeCallback()
        cbs = [time_cb, loss_cb]
        params = config.trainer_params
        super().__init__(config.algorithm_config, None, params=params, callbacks=cbs)
