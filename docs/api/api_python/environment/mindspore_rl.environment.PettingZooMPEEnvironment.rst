.. py:class:: mindspore_rl.environment.PettingZooMPEEnvironment(params, env_id=0)

    PettingZooMPEEnvironment `PettingZoo <https://pettingzoo.farama.org/environments/mpe/>`_ 封装成一个类来提供在MindSpore图模式下也能和PettingZoo环境交互的能力。

    参数：
        - **params** (dict) - 字典包含PettingZooMPEEnvironment类中所需要的所有参数。

        +------------------------------+-------------------------------+
        |             配置参数         |            备注               |
        +==============================+===============================+
        |             名字             |           游戏名              |
        +------------------------------+-------------------------------+
        |             个数             |          环境的个数           |
        +------------------------------+-------------------------------+
        |           是否连续动作       |         动作空间的类型        |
        +------------------------------+-------------------------------+

        - **env_id** (int，可选) - 环境id，用于设置环境内种子，默认为第0个环境。默认： ``0`` 。

    .. py:method:: action_space
        :property:

        获取环境的动作空间。

        返回：
            Space，环境的动作空间。

    .. py:method:: close

        关闭环境以释放环境资源

        返回：
            - **Success** (np.bool\_) - 是否成功释放资源。

    .. py:method:: config
        :property:

        获取环境的配置信息。

        返回：
            dict，一个包含环境信息的字典。

    .. py:method:: done_space
        :property:

        获取环境的终止空间。

        返回：
            Space，环境的终止空间。

    .. py:method:: observation_space
        :property:

        获取环境的状态空间。

        返回：
            Space，环境的状态空间。

    .. py:method:: render()

        渲染环境，仅支持PyNative模式。

    .. py:method:: reset()

        将环境重置为初始状态。reset方法一般在每一局游戏开始时使用，并返回环境的初始状态值。

        返回：
            Tensor，表示环境初始状态。

    .. py:method:: reward_space
        :property:

        获取环境的状态空间。

        返回：
            Space，环境的奖励空间。

    .. py:method:: step(action)

        执行环境Step函数来和环境交互一回合。

        参数：
            - **action** (Tensor) - 包含动作信息的Tensor。

        返回：
            - **state** (Tensor) - 输入动作后的环境返回的新状态。
            - **reward** (Tensor) - 输入动作后环境返回的奖励。
            - **done** (Tensor) - 输入动作后环境是否终止。
