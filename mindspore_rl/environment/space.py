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
The class define action space and observation class.
"""

import numpy as np
from mindspore.common import dtype as mstype


np_types = (np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64, np.float16,
            np.float32, np.float64, np.bool_)


class Space:
    """
    The class for environment action/observation space.

    Args:
        feature_shape (Union[list(int), tuple(int), int]): The action/observation shape before batching.
        dtype (np.dtype): The action/observation space dtype.
        low (Union[int, float], optional): The action/observation space lower boundary.
        high (Union[int, float], optional): The action/observation space upper boundary.
        batch_shape (Union[list(int), tuple(int), int], optional): The batch shape for vectorization.
          It usually be used in multi-environment and multi-agent cases.

    Examples:
        >>> action_space = Space(feature_shape=(6,), dtype=np.int32)
        >>> print(action_space.ms_dtype)
        Int32
    """

    def __init__(self, feature_shape, dtype, low=None, high=None, batch_shape=None):
        if not issubclass(dtype, np_types):
            raise ValueError("Dtype {} not supported!".format(dtype))

        self._feature_shape = tuple(feature_shape)
        self._dtype = dtype
        self._batch_shape = tuple(
            batch_shape) if batch_shape is not None else tuple()
        self._low, self._high = self._range(low, high)

    def sample(self):
        '''
        Sample a valid action from the space

        Returns:
            Tensor, a valid action.
        '''

        if self.is_discrete:
            return np.random.randint(low=self._low, high=self._high, size=self.shape).astype(self._dtype)

        return np.random.uniform(low=self._low, high=self._high, size=self.shape).astype(self._dtype)

    @property
    def shape(self):
        '''
        Space shape after batching.

        Returns:
            The shape of current space.
        '''
        return self._batch_shape + self._feature_shape

    @property
    def np_dtype(self):
        '''
        Numpy data type of current Space.

        Returns:
            The numpy dtype of current space.
        '''
        return self._dtype

    @property
    def ms_dtype(self):
        '''
        MindSpore data type of current Space.

        Returns:
            The mindspore data type of current space.
        '''
        return mstype.pytype_to_dtype(self._dtype)

    @property
    def is_discrete(self):
        '''
        Is discrete space.

        Returns:
            Whether the current space is discrete or continuous.
        '''
        return issubclass(self._dtype, np.integer) or self._dtype == np.bool_

    @property
    def num_values(self):
        '''
        available action number of current Space.

        Returns:
            The available action of current space.
        '''
        if not self.is_discrete:
            return self.shape[-1]

        enums_range = self._high - self._low
        if enums_range.shape == ():
            return enums_range.item(0)

        num = 1
        for i in enums_range:
            num *= i.item(0)
        return num

    @property
    def boundary(self):
        '''
        The space boundary of current Space.

        Returns:
            Uppoer and lower boundary of current space.
        '''
        return self._low, self._high

    def _range(self, low, high):
        '''Return the space range.'''

        if self.is_discrete:
            if self._dtype == np.bool_:
                dtype_low, dtype_high = 0, 2
            else:
                dtype_low, dtype_high = np.iinfo(
                    self._dtype).min, np.iinfo(self._dtype).max
        else:
            dtype_low, dtype_high = np.finfo(
                self._dtype).min, np.finfo(self._dtype).max

        low = dtype_low if low is None else low
        high = dtype_high if high is None else high
        return np.broadcast_to(low, self._feature_shape), np.broadcast_to(high, self._feature_shape)

    def __repr__(self):
        return "shape {}, dtype {}, range ({}, {})".format(self.shape, self.np_dtype, self._low, self._high)

    def __str__(self):
        return self.__repr__()
