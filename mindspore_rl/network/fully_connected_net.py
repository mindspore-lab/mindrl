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
FullyConnectedNet.
"""

import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore.ops import operations as P


class FullyConnectedNet(nn.Cell):
    """
    A basic fully connected neural network.

    Args:
        input_size(int): numbers of input size.
        hidden_size(int): numbers of hidden layers.
        output_size(int): numbers of output size.
        compute_type(mindspore.dtype): data type used for fully connected layer.
            Default: ``mindspore.dtype.float32`` .

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore_rl.network.fully_connected_net import FullyConnectedNet
        >>> input = Tensor(np.ones([2, 4]).astype(np.float32))
        >>> net = FullyConnectedNet(4, 10, 2)
        >>> output = net(input)
        >>> print(output.shape)
        (2, 2)
    """

    def __init__(self, input_size, hidden_size, output_size, compute_type=mstype.float32):
        super(FullyConnectedNet, self).__init__()
        self.linear1 = nn.Dense(
            input_size,
            hidden_size,
            weight_init="XavierUniform").to_float(compute_type)
        self.linear2 = nn.Dense(
            hidden_size,
            output_size,
            weight_init="XavierUniform").to_float(compute_type)
        self.relu = nn.ReLU()
        self.cast = P.Cast()

    def construct(self, x):
        """
        Returns output of Dense layer.

        Args:
            x (Tensor): Tensor as the input of network.

        Returns:
            The output of the Dense layer.
        """
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.cast(x, mstype.float32)
        return x


class FullyConnectedLayers(nn.Cell):
    r"""
    This is a fully connected layers module. User can input abitrary number of `fc_layer_params`, then
    this module can create corresponding number of fully connect layers.

    Args:
        fc_layer_params (list[int]): A list of int states for the input and output size of fully
            connected layer. For example, if the input list is [10, 20, 3], then the module will
            create two fully connected layers whose input and output size are (10, 20) and (20, 3)
            respectively. The length of `fc_layer_params` should be greater than or equal to 3.
        dropout_layer_params (list[float]): A list of float states for the dropout rate. If the input
            list if [0.5, 0.3], then two dropout layers will be created after each fully connected
            layer. The length of `dropout_layer_params` should be one less than `fc_layer_params`.
            `dropout_layer_params` is a optional value. Default: ``None`` .
        activation_fn (Union[str, Cell, Primitive]): An instance of activation function.
            Default: ``nn.ReLu()`` .
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable `weight_init` parameter.
            The dtype is same as `x`. The values of str refer to the function `initializer`,
            e.g.  ``normal`` , ``uniform`` . Default: ``'normal'`` .
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable `bias_init` parameter. The
            dtype is same as `x`. The values of str refer to the function `initializer`,
            e.g.  ``normal`` , ``uniform`` . Default: ``'zeros'`` .

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, fc\_layers\_params[0])`.

    Outputs:
        Tensor of shape :math:`(*, fc\_layers\_params[-1])`.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore_rl.network.fully_connected_net import FullyConnectedLayers
        >>> input = Tensor(np.ones([2, 4]).astype(np.float32))
        >>> net = FullyConnectedLayers(fc_layer_params=[4, 10, 2])
        >>> output = net(input)
        >>> print(output.shape)
        (2, 2)
    """
    def __init__(self,
                 fc_layer_params,
                 dropout_layer_params=None,
                 activation_fn=nn.ReLU(),
                 weight_init='normal',
                 bias_init='zeros'):
        super().__init__()
        layers = []
        if len(fc_layer_params) < 3:
            raise ValueError("The length of fc_layer_params must be greater than or equal to 3, \
                             but the length of fc_layer_params is %d." % len(fc_layer_params))
        if dropout_layer_params:
            if len(dropout_layer_params) != (len(fc_layer_params) - 1):
                raise ValueError("The length of dropout_layer_params must be one less than fc_layer_params, \
                                 but got %d and %d." % (len(fc_layer_params), len(dropout_layer_params)))
            for i in range(len(fc_layer_params) - 1):
                layers.append(nn.Dense(fc_layer_params[i],
                                       fc_layer_params[i+1],
                                       weight_init=weight_init,
                                       bias_init=bias_init,
                                       activation=activation_fn))
                layers.append(nn.Dropout(keep_prob=dropout_layer_params[i]))
        else:
            for i in range(len(fc_layer_params) - 1):
                layers.append(nn.Dense(fc_layer_params[i],
                                       fc_layer_params[i+1],
                                       weight_init=weight_init,
                                       bias_init=bias_init,
                                       activation=activation_fn))
        self.fc_layers = nn.SequentialCell(layers)

    def construct(self, x):
        r"""
        Args:
            x (Tensor): Tensor of shape :math:`(*, fc\_layers\_params[0])`.

        Returns:
            Tensor of shape :math:`(*, fc\_layers\_params[-1])`.
        """
        return self.fc_layers(x)
