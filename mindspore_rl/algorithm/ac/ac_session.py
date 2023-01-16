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
AC session.
"""
from mindspore_rl.core import Session
from mindspore_rl.utils.utils import update_config
from mindspore_rl.utils.callback import LossCallback, EvaluateCallback
from mindspore_rl.algorithm.ac import config


class ACSession(Session):
    '''AC session'''
    def __init__(self, env_yaml=None, algo_yaml=None):
        update_config(config, env_yaml, algo_yaml)
        params = config.trainer_params
        loss_cb = LossCallback()
        eval_cb = EvaluateCallback(10)
        cbs = [loss_cb, eval_cb]
        super().__init__(config.algorithm_config, None, params=params, callbacks=cbs)
