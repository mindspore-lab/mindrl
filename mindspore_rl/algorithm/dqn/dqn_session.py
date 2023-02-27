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
from mindspore_rl.core import Session
from mindspore_rl.utils.utils import update_config
from mindspore_rl.utils.callback import CheckpointCallback, LossCallback, EvaluateCallback, TimeCallback
from mindspore_rl.algorithm.dqn import config
from mindspore import context
from mindspore import dtype as mstype


class DQNSession(Session):
    '''DQN session'''
    def __init__(self, env_yaml=None, algo_yaml=None):
        update_config(config, env_yaml, algo_yaml)
        compute_type = mstype.float16 if context.get_context('device_target') in ['Ascend'] else mstype.float32
        config.algorithm_config['policy_and_network']['params']['compute_type'] = compute_type
        env = config.algorithm_config.get('collect_environment').get('type')(config.collect_env_params)
        config.algorithm_config['replay_buffer']['data_shape'] = [env.observation_space.shape, (1,),
                                                                  env.reward_space.shape,
                                                                  env.observation_space.shape]
        config.algorithm_config['replay_buffer']['data_type'] = [env.observation_space.ms_dtype,
                                                                 env.action_space.ms_dtype,
                                                                 env.reward_space.ms_dtype,
                                                                 env.observation_space.ms_dtype]
        params = config.trainer_params
        loss_cb = LossCallback()
        ckpt_cb = CheckpointCallback(config.trainer_params.get('save_per_episode'),
                                     config.trainer_params.get('ckpt_path'))
        eval_cb = EvaluateCallback(config.trainer_params.get('eval_per_episode'))
        time_cb = TimeCallback()
        cbs = [loss_cb, ckpt_cb, eval_cb, time_cb]
        super().__init__(config.algorithm_config, None, params=params, callbacks=cbs)
