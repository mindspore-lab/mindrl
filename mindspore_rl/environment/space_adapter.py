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
Space adapter utils
"""
import numpy as np
from gym import spaces

from mindspore_rl.environment.space import Space


def gym2ms_adapter(gym_space):
    """gym space to ms space adapter"""
    batch_shape = None
    if isinstance(gym_space, list):
        batch_shape = (len(gym_space),)
        gym_space = gym_space[0]
    shape = gym_space.shape
    gym_type = gym_space.dtype.type
    # The dtype get from gym.space is np.int64, but step() accept np.int32 actually.
    if gym_type == np.int64:
        dtype = np.int32
    # The float64 is not supported, cast to float32
    elif gym_type == np.float64:
        dtype = np.float32
    else:
        dtype = gym_type

    if isinstance(gym_space, spaces.Discrete):
        return Space(shape, dtype, low=0, high=gym_space.n, batch_shape=batch_shape)

    return Space(
        shape, dtype, low=gym_space.low, high=gym_space.high, batch_shape=batch_shape
    )


def dmc2ms_adapter(dmc_space):
    """dmc space to ms space adapter"""
    batch_shape = None
    if isinstance(dmc_space, list):
        batch_shape = (len(dmc_space),)
        dmc_space = dmc_space[0]

    shape = dmc_space.shape
    dmc_type = dmc_space.dtype.type
    if dmc_type == np.int64:
        dtype = np.int32
    elif dmc_type == np.float64:
        dtype = np.float32
    else:
        dtype = dmc_type

    return Space(
        shape,
        dtype,
        low=dmc_space.minimum,
        high=dmc_space.maximum,
        batch_shape=batch_shape,
    )
