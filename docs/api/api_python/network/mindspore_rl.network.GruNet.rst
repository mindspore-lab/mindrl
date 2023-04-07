
.. py:class:: mindspore_rl.network.GruNet(input_size, hidden_size, weight_init='normal', num_layers=1, has_bias=True, batch_first=False, dropout=0.0, bidirectional=False, enable_fusion=True)

    GRU (门控递归单元)层。
    将GRU层应用于输入。
    有关详细信息，请参见： `mindspore.nn.GRU <https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.GRU.html>`_ 。

    参数：
        - **input_size** (int) - 输入的特征数。
        - **hidden_size** (int) - 隐藏层的特征数量。
        - **weight_init** (str or Initializer) - 初始化方法，如normal、uniform。默认值： 'normal'。
        - **num_layers** (int) - GRU层的数量。默认值： 1。
        - **has_bias** (bool) - cell中是否有偏置。默认值： True。
        - **batch_first** (bool) - 指定输入 `x` 的第一个维度是否为批处理大小。默认值： False。
        - **dropout** (float) - 如果不是0.0, 则在除最后一层外的每个GRU层的输出上附加 `Dropout` 层。默认值： 0.0。取值范围 [0.0, 1.0)。
        - **bidirectional** (bool) - 指定它是否为双向GRU，如果bidirectional=True则为双向，否则为单向。默认值： False。
        - **enable_fusion** (bool) - 是否需要使用GRU的融合算子。默认值：True。

    输入：
        - **x_in** (Tensor) - 数据类型为mindspore.float32和shape为 :math:`(seq\_len, batch\_size, input\_size)` 或 :math:`(batch\_size, seq\_len, input\_size)` 的Tensor。
        - **h_in** (Tensor) - 数据类型为mindspore.float32和shape为 :math:`(num\_directions * num\_layers, batch\_size, hidden\_size)` 的Tensor。`h_in` 的数据类型必须和 `x_in` 一致。

    输出：
        元组，包含(`x_out`, `h_out`)。

        - **x_out** (Tensor) - shape为 :math:`(seq\_len, batch\_size, num\_directions * hidden\_size)` 的Tensor。
        - **h_out** (Tensor) - shape为 :math:`(num\_directions * num\_layers, batch\_size, hidden\_size)` 的Tensor。

    .. py:method:: construct(x_in, h_in)

        gru网络的正向输出。

        参数：
            - **x_in** (Tensor) - 数据类型为mindspore.float32和shape为 :math:`(seq\_len, batch\_size, input\_size)` 或 :math:`(batch\_size, seq\_len, input\_size)` 的Tensor。
            - **h_in** (Tensor) - 数据类型为mindspore.float32和shape为 :math:`(num\_directions * num\_layers, batch\_size, hidden\_size)` 的Tensor。`h_in` 的数据类型必须和 `x_in` 一致。

        返回：
            - **x_out** (Tensor) - shape为 :math:`(seq\_len, batch\_size, num\_directions * hidden\_size)` 的Tensor。
            - **h_out** (Tensor) - shape为 :math:`(num\_directions * num\_layers, batch\_size, hidden\_size)` 的Tensor。
