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
Parser for distribution policy
"""

import os
import stat
import re
import copy


def interface_parser(policy):
    '''Parsing the interface in distribution policy'''
    interface_parameters = []
    interfaces = copy.copy(policy.interface)
    if policy.fuse:
        for fused in policy.fuse:
            for key in fused:
                for f_type in fused[key]:
                    interfaces.pop(f_type)
                new_interface = {key: policy.interface[fused[key][-1]]}
                interface_parameters.append(new_interface)
        for f_type in interfaces:
            new_interface = {f_type: interfaces[f_type]}
            interface_parameters.append(new_interface)
    else:
        interface_parameters = interfaces

    return interface_parameters



def anno_parser(file_name):
    '''Parsing the annotation.'''
    annotation_list = []
    with open(file_name, 'r', encoding="utf-8") as fin:
        lines = fin.readlines()

        for line_no, _ in enumerate(lines):
            line = lines[line_no]
            if '#@msrl' in line:
                anno_dict = {}
                parameters = re.findall(r'\(.*?\)', line)
                parameters = parameters[0]
                parameters = parameters.strip('()').split(',')
                for i in parameters:
                    param = i.split('=')
                    anno_dict[param[0].strip(' ')] = param[1]
                annotation_list.append(anno_dict)
                space = 0
                for c in line:
                    if c == ' ':
                        space += 1
                    else:
                        break
                line = space * ' ' + str(anno_dict) + '\n'
                lines[line_no] = line

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    fout = file_name.split('.')[0]+'_convert'+'.py'
    with os.fdopen(os.open(fout, flags, modes), 'w') as fout:
        fout.writelines(lines)
