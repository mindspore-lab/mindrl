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
Distribution policy base class
"""


class DistributionPolicy:
    """Define distrbution policy"""

    def __init__(self):
        self.actor_number = 0
        self.learner_number = 0
        self.agent_number = 0
        self.environment_number = 0
        self.fragment_number = 0
        self.boundary = []
        self.interface = {}
        self.communication_data = {}
        self.fuse = []
        self.replicate_list = []
        self.topology = []

    def set_actor_number(self, actor_num):
        if actor_num >= 0:
            self.actor_number = actor_num
        else:
            raise Exception("Actor number cannot smaller than 0")

    def set_learner_number(self, learner_num):
        if learner_num >= 0:
            self.learner_number = learner_num
        else:
            raise Exception("Learner number cannot smaller than 0")

    def set_agent_number(self, agent_num):
        if agent_num >= 0:
            self.agent_number = agent_num
        else:
            raise Exception("Agent number cannot smaller than 0")

    def set_fragment_number(self, fragment_num):
        if fragment_num > 0:
            self.fragment_number = fragment_num
        else:
            raise Exception("fragment number cannot smaller than 1")

    def set_boundary(self, boundary):
        self.boundary = boundary

    def add_interface(self, fragment_type, parameters):
        self.interface[fragment_type] = parameters

    def add_communication_data(self, name, data):
        self.communication_data[name] = data

    def fuse_fragment(self, fused_type, fragments_list):
        fused = {fused_type: fragments_list}
        self.fuse.append(fused)

    def set_replicate_list(self, fragment_type, num):
        self.replicate_list.append({fragment_type: num})

    def set_topology(self, topology):
        self.topology = topology
