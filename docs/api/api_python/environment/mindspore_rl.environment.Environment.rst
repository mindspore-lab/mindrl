.. py:class:: mindspore_rl.environment.Environment(env_name=None, env=None, config=None)

    环境的虚基类。每一个子类环境都需要继承这个基类，并且需要在子类中实现\_reset，\_get\_action，\_step，\_get\_min\_max\_action和\_get\_min\_max\_observation。基类提供了自动将python实现的reset和step方法用mindspore算子（PyFunc）抱起来的能力，并且也提供了自动生成环境Space的能力。

    参数：
        - **env_name** (str) - 子类环境的名字。Default：None
        - **env** (Environment) - 子类环境的实例。Default：None
        - **config** (dict) - 环境的配置信息，可以通过调用环境的config属性来获得。Default: None

    .. py:method:: action_space
        :property:

        获取环境的动作空间。

        返回：
            - **action_space** (Space) - 返回环境的动作空间。

    .. py:method:: close

        关闭环境以释放环境资源

        返回：
            - **Success** (bool) - 是否成功释放资源。

    .. py:method:: config
        :property:

        获取环境的配置信息。

        返回：
             - **config** (dict) - 一个包含环境信息的字典。

    .. py:method:: done_space
        :property:

        获取环境的终止空间。

        返回：
            - **done_space** (Space) - 返回环境的终止空间。

    .. py:method:: observation_space
        :property:

        获取环境的状态空间。

        返回：
            - **observation_space** (Space) - 返回环境的状态空间。

    .. py:method:: reset()

        将环境重置为初始状态。reset方法一般在每一局游戏开始时使用，并返回环境的初始状态值以及其reset方法初始信息。

        返回：
            - **state** (Tensor) - 一个表示环境初始状态的Tensor。
            - **other** (Tensor) - \_reset方法中除了state以外的其他输出。

    .. py:method:: reward_space
        :property:

        获取环境的状态空间。

        返回：
            - **reward_space** (Space) - 返回环境的奖励空间。

    .. py:method:: step(action)

        执行环境Step函数来和环境交互一回合。

        参数：
            - **action** (Tensor) - 包含动作信息的Tensor。

        返回：
            - **state** (Tensor) - 输入动作后的环境返回的新状态。
            - **reward** (Tensor) - 输入动作后环境返回的奖励。
            - **done** (Tensor) - 输入动作后环境是否终止。
            - **other** (Tensor) - \_step方法中剩下的返回值。
