
.. py:class:: mindspore_rl.utils.OUNoise(stddev, damping, action_shape)

    在action上加入Ornstein-Uhlenbeck (OU)噪声。

    设均值为0的正态分布为 :math:`N(0, stddev)`，
    则下一个时序值是 :math:`x\_next = (1 - damping) * x - N(0, stddev)`，
    加入OU噪声的action是 :math:`action += x\_next`。

    参数：
        - **stddev** (float) - Ornstein-Uhlenbeck (OU) 噪声标准差。
        - **damping** (float) - Ornstein-Uhlenbeck (OU) 噪声阻尼。
        - **action_shape** (tuple) - 动作的维度。

    输入：
        - **actions** (Tensor) - 添加OU噪声之前的动作。

    输出：
        - **actions** (Tensor) - 添加OU噪声之后的动作。

