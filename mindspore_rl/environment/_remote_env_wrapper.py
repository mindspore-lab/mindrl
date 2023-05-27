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
"""remote environment wrapper"""

from mindspore_rl.environment.wrapper import Wrapper


class _RemoteEnvWrapper(Wrapper):
    def __init__(self, env_creators, deploy_config):
        super().__init__(env_creators)
        self._worker_num = deploy_config.get("worker_num", 1)

    @property
    def num_environment(self) -> int:
        return self.environment.num_environment * (self._worker_num - 1)

    def _reset(self):
        return self.environment.reset()

    def _step(self, action):
        return self.environment.step(action)

    def _send(self, action, env_id):
        return self.environment.send(action, env_id)

    def _recv(self):
        return self.environment.recv()
