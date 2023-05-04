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

    .. py:method:: close

        关闭环境以释放环境资源

        返回：
            - **Success** (np.bool\_) - 是否成功释放资源。
