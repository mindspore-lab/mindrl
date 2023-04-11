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
Multi Actor single learner policy
"""
from mindspore_rl.distribution.distribution_policies.distribution_policy import (
    DistributionPolicy,
)


class MultiActorEnvSingleLearnerDP(DistributionPolicy):
    """define multi actor single learner policy"""

    def __init__(self, msrl, algorithm_config=None):
        # pylint: disable=R1725
        super(MultiActorEnvSingleLearnerDP, self).__init__()
        if algorithm_config is not None:
            self.set_actor_number(algorithm_config["actor"]["number"])
            self.set_learner_number(algorithm_config["learner"]["number"])
            self.set_fragment_number(self.actor_number + self.learner_number)
        self.set_boundary("algorithmic")
        self.add_interface("Actor", {"operations": {"AllGather"}})
        self.add_interface("Learner", {"operations": {"AllGather"}})
        self.add_communication_data(
            "Data", list(map(lambda b: b.name, msrl.buffers.buffer))
        )
        self.add_communication_data(
            "Weight", list(map(lambda b: b.name, msrl.shared_network.get_parameters()))
        )
        self.set_replicate_list("Actor", self.actor_number)
        self.set_topology({"Learner": "Actor"})
        self.name = "MultiActorEnvSingleLearnerDP"
        self.auto = True
