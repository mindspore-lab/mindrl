
.. py:class:: mindspore_rl.policy.RandomPolicy(action_space_dim, shape=(1,))

    在[0, `action_space_dim`)之间产生随机动作。

    参数：
        - **action_space_dim** (int) - 动作空间的维度。
        - **shape** (tuple, 可选) - random policy输出的动作shape。默认值为(1,)。

    .. py:method:: construct()

        返回[0, `action_space_dim`)之间的随机数。

        返回：
            [0, `action_space_dim`)之间的随机数。
