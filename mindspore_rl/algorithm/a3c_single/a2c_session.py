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
A2C session.
"""
from mindspore_rl.algorithm.a3c_single import config
from mindspore_rl.core import Session
from mindspore_rl.utils.utils import update_config


class A2CSession(Session):
    """A2C session"""

    def __init__(self, env_yaml=None, algo_yaml=None, is_distribution=None):
        update_config(config, env_yaml, algo_yaml)
        deploy_config = None
        if is_distribution:
            deploy_config = config.deploy_config
        super().__init__(config.algorithm_config, deploy_config)
