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
Pre-defined Distribution policies
"""

from mindspore_rl.distribution.distribution_policies.async_multi_actor_single_learner_dp import (
    AsyncMultiActorSingleLearnerDP,
)
from mindspore_rl.distribution.distribution_policies.multi_actor_single_learner_dp import (
    MultiActorSingleLearnerDP,
)
from mindspore_rl.distribution.distribution_policies.single_actor_learner_with_multi_env_dp import (
    SingleActorLearnerMultiEnvDP,
)

from .distribution_policy import DistributionPolicy

__all__ = [
    "DistributionPolicy",
    "MultiActorSingleLearnerDP",
    "AsyncMultiActorSingleLearnerDP",
    "SingleActorLearnerMultiEnvDP",
]
