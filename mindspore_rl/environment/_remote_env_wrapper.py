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

from mindspore_rl.environment.pyfunc_wrapper import PyFuncWrapper
from mindspore_rl.environment.wrapper import Wrapper


class _RemoteEnvWrapper(Wrapper):
    """
    Inner wrapper for remote environment. Since remote environment will separate a set of environment
    to different workers, we need to get the information of the environment in each worker.
    """

    def __init__(self, env_creators, deploy_config):
        super().__init__(env_creators)
        self._worker_num = deploy_config.get("worker_num", 1)

    @property
    def num_environment(self) -> int:
        """number of environment"""
        return self.environment.num_environment * (self._worker_num - 1)

    @property
    def num_env_per_worker(self) -> int:
        """number of environment in each worker"""
        return self.environment.num_environment

    @property
    def reset_output_shape(self) -> list:
        """reset output shape"""
        if not isinstance(self.environment, PyFuncWrapper):
            raise ValueError(
                "Please make sure the 0th wrapper in the wrapper list must be PyFuncWrapper."
            )
        return getattr(self.environment, "_reset_output_full_shape")

    @property
    def reset_output_dtype(self) -> list:
        """reset output data type"""
        if not isinstance(self.environment, PyFuncWrapper):
            raise ValueError(
                "Please make sure the 0th wrapper in the wrapper list must be PyFuncWrapper."
            )
        return getattr(self.environment, "_reset_output_msdtype")

    @property
    def step_input_shape(self) -> list:
        """step input shape"""
        if not isinstance(self.environment, PyFuncWrapper):
            raise ValueError(
                "Please make sure the 0th wrapper in the wrapper list must be PyFuncWrapper."
            )
        return getattr(self.environment, "_step_input_full_shape")

    @property
    def step_input_dtype(self) -> list:
        """step input data type"""
        if not isinstance(self.environment, PyFuncWrapper):
            raise ValueError(
                "Please make sure the 0th wrapper in the wrapper list must be PyFuncWrapper."
            )
        return getattr(self.environment, "_step_input_msdtype")

    @property
    def step_output_shape(self) -> list:
        """step output shape"""
        if not isinstance(self.environment, PyFuncWrapper):
            raise ValueError(
                "Please make sure the 0th wrapper in the wrapper list must be PyFuncWrapper."
            )
        return getattr(self.environment, "_step_output_full_shape")

    @property
    def step_output_dtype(self) -> list:
        """step output data type"""
        if not isinstance(self.environment, PyFuncWrapper):
            raise ValueError(
                "Please make sure the 0th wrapper in the wrapper list must be PyFuncWrapper."
            )
        return getattr(self.environment, "_step_output_msdtype")

    def _reset(self):
        """Inner reset function"""
        return self.environment.reset()

    def _step(self, action):
        """Inner step function"""
        return self.environment.step(action)

    def _send(self, action, env_id):
        """Inner send function"""
        return self.environment.send(action, env_id)

    def _recv(self):
        """Inner recv function"""
        return self.environment.recv()
