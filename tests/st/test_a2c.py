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
Test case for A2C training.
"""

# pylint: disable=C0413
# pylint: disable=C0411
# pylint: disable=W0611
import pytest
from mindspore import context

from mindspore_rl.algorithm.a2c import A2CSession, A2CTrainer


@pytest.mark.skip(reason="Need mindspore update")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_train_a2c():
    """
    Feature: Test A2C train.
    Description: A2C net.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    ac_session = A2CSession()
    ac_session.run(class_type=A2CTrainer, episode=5)
    assert True
