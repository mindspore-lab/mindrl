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
Generate fragment
"""

import sys
import os
import shutil
import importlib
from .annotation_parser import interface_parser
from .generate_code import generate_fragment


def fragment_generation(algorithm, algorithm_config, policy_name):
    '''Generate fragments'''
    path = os.path.dirname(os.path.abspath(policy_name))
    src_template = path+'/template.txt'
    template = path+'/template.py'
    shutil.copy(src_template, template)
    sys.path.insert(0, path)
    policy_module = importlib.import_module(policy_name)
    policy = getattr(policy_module, policy_name)()
    if policy.auto:
        parameter_list = interface_parser(policy)
        position = {'Trainer': 'train_one_episode'}
    else:
        algorithm, parameter_list = annotation_parser.anno_parser(algorithm)
    fragments = generate_fragment(algorithm, parameter_list, template, algorithm_config, position, policy)
    print(fragments)
    return fragments
