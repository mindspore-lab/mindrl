# Copyright 2021 Huawei Technologies Co., Ltd
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
The environment registration class.
"""

class Registry:
    """
    Register an environment by environment name.

    Args:
        - **domain** (str): Environment domain.

    Examples:
        >>> from mindspore_rl.environment.tag_environment import TagEnvironment
        >>> register = Registry('MindSpore')
        >>> register.register('Tag', TagEnvironment)
        >>> print(register)
        Domain MindSpore, registered environment dict_keys(['Tag'])
    """

    def __init__(self, domain):
        self.domain = domain
        self.env_dict = {}

    def register(self, name, env_class):
        '''Register an environment'''
        if name in self.env_dict:
            raise TypeError("Environment {} is already registered.".format(name))

        self.env_dict[name] = env_class

    def create(self, name, **kwargs):
        '''Create an environment instance with given name'''
        if name not in self.env_dict:
            raise TypeError("Environment {} not registered.".format(name))
        return self.env_dict[name](**kwargs)

    def __repr__(self):
        return "Domain {}, registered environment {}".format(self.domain, self.env_dict.keys())

    def __str__(self):
        return self.__repr__()
