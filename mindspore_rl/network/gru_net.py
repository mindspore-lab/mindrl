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
Cudnn Gru network.
"""
from mindspore_rl.network._rnns import GRU

import mindspore.nn as nn
from mindspore import context
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore import _checkparam as validator
from mindspore.ops.operations import _rl_inner_ops as rl_ops


class GruNet(nn.Cell):
    """
    Stacked GRU (Gated Recurrent Unit) layers.

    Apply GRU layer to the input.

    For detailed information, please refer to
    `mindspore.nn.GRU <https://www.mindspore.cn/docs/en/r2.0/api_python/nn/mindspore.nn.GRU.html>`_.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        weight_init (str or Initializer): Initialize method. Default: 'normal'.
        num_layers (int): Number of layers of stacked GRU. Default: 1.
        has_bias (bool): Whether the cell has bias. Default: True.
        batch_first (bool): Specifies whether the first dimension of input `x` is batch_size. Default: False.
        dropout (float): If not 0.0, append `Dropout` layer on the outputs of each
            GRU layer except the last layer. Default 0.0. The range of dropout is [0.0, 1.0).
        bidirectional (bool): Specifies whether it is a bidirectional GRU,
            num_directions=2 if bidirectional=True otherwise 1. Default: False.
        enable_fusion (bool): Whether need to use GRU fusion ops. Default: True.

    Inputs:
        - **x_in** (Tensor) - Tensor of data type mindspore.float32 and
          shape (seq_len, batch_size, `input_size`) or (batch_size, seq_len, `input_size`).
        - **h_in** (Tensor) - Tensor of data type mindspore.float32 and
          shape (num_directions * `num_layers`, batch_size, `hidden_size`). The data type of `h_in` must be the same as
          `x_in`.

    Outputs:
        Tuple, a tuple contains (`x_out`, `h_out`).

        - **x_out** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`) or
          (batch_size, seq_len, num_directions * `hidden_size`).
        - **h_out** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).

    Examples:
        >>> net = GruNet(10, 16, 2, has_bias=True, bidirectional=False)
        >>> x_in = Tensor(np.ones([3, 5, 10]).astype(np.float32))
        >>> h_in = Tensor(np.ones([1, 5, 16]).astype(np.float32))
        >>> x_out, h_out = net(x_in, h_in)
        >>> print(x_out.shape)
        (3, 5, 16)
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 weight_init='normal',
                 num_layers=1,
                 has_bias=True,
                 batch_first=False,
                 dropout=0.0,
                 bidirectional=False,
                 enable_fusion=True):

        super().__init__()
        validator.check_positive_int(hidden_size, "hidden_size", self.cls_name)
        validator.check_positive_int(input_size, "input_size", self.cls_name)
        validator.check_positive_int(num_layers, "num_layers", self.cls_name)
        validator.check_is_float(dropout, "dropout", self.cls_name)
        validator.check_value_type("has_bias", has_bias, [bool], self.cls_name)
        validator.check_value_type(
            "batch_first", batch_first, [bool], self.cls_name)
        validator.check_value_type(
            "bidirectional", bidirectional, [bool], self.cls_name)

        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.enable_cudnn = context.get_context('device_target') in ['GPU']
        self.enable_fusion = enable_fusion
        if self.enable_cudnn and self.enable_fusion:
            weight_size = 0
            gate_size = 3 * hidden_size
            num_directions = 2 if bidirectional else 1
            for layer in range(num_layers):
                input_layer_size = input_size if layer == 0 else hidden_size * num_directions
                increment_size = gate_size * input_layer_size
                increment_size += gate_size * hidden_size
                if has_bias:
                    increment_size += 2 * gate_size
                weight_size += increment_size * num_directions
            self.weight = Parameter(initializer(
                weight_init, [weight_size, 1, 1]), name="cudnn_weight")
            self.gru = rl_ops.CudnnGRU(input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_layers=num_layers,
                                       has_bias=has_bias,
                                       bidirectional=bidirectional,
                                       dropout=float(dropout))
        else:
            self.gru = GRU(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           has_bias=has_bias,
                           batch_first=batch_first,
                           bidirectional=bidirectional,
                           dropout=float(dropout),
                           enable_fusion=enable_fusion)

    def construct(self, x_in, h_in):
        """
        The forward calculation of gru net

        Args:
            x_in (Tensor): Tensor of data type mindspore.float32 and shape (seq_len, batch_size, `input_size`)
                            or (batch_size, seq_len, `input_size`).
            h_in (Tensor): Tensor of data type mindspore.float32 and shape (num_directions * `num_layers`,
                            batch_size, `hidden_size`). The data type of `h_in` must be the same as `x_in`.

        Returns:
            - **x_out** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`) or
              (batch_size, seq_len, num_directions * `hidden_size`).
            - **h_out** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        """
        x_out = None
        h_out = None
        if self.enable_cudnn and self.enable_fusion:
            x_out, h_out, _, _ = self.gru(x_in, h_in, self.weight)
        else:
            x_out, h_out = self.gru(x_in, h_in)
        return x_out, h_out
