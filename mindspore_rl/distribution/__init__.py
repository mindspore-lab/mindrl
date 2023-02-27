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
Components for Automatic distributin.
"""

from mindspore_rl.distribution.code_generation import fragment_generation
from mindspore_rl.distribution.distribution_policies import DistributionPolicy
from mindspore_rl.distribution.distribution_policies import MultiActorEnvSingleLearnerDP
from mindspore_rl.distribution.distribution_policies import AsyncMultiActorSingleLearnerDP

__all__ = ["fragment_generation", "DistributionPolicy",
           "MultiActorEnvSingleLearnerDP", "AsyncMultiActorSingleLearnerDP"]
