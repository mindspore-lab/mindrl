.. py:class:: mindspore_rl.environment.Environment()

    环境的虚基类。所有环境或者Wrapper都需要继承这个基类，并且子类需要重写相应的函数和属性。

    .. py:method:: action_space
        :property:

        获取环境的动作空间。

        返回：
            - **action_space** (Space) - 返回环境的动作空间。

    .. py:method:: batched
        :property:

        环境是否batched

        返回：
            - **batched** (bool) - 是否环境是batched。默认为False。

    .. py:method:: close

        关闭环境以释放环境资源

        返回：
            - **Success** (np.bool\_) - 是否成功释放资源。

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

    .. py:method:: num_agent
        :property:

        环境中的智能体个数

        返回：
            - **num_agent** (Space) - 环境中的智能体个数。如果环境为单智能体，会返回1。其他情况，子类需要重写这个这个属性去返回对应的智能体个数。默认为1。

    .. py:method:: num_environment
        :property:

        环境个数。

        返回：
            - **num_env** (Space) - 环境的个数。

    .. py:method:: observation_space
        :property:

        获取环境的状态空间。

        返回：
            - **observation_space** (Space) - 返回环境的状态空间。

    .. py:method:: recv()

        接受和环境交互的结果。

        参数：
            - **action** (Union[Tensor, np.ndarray]) - 包含动作信息的Tensor。

        返回：
            - **state** (Union[np.ndarray, Tensor]) - 输入动作后的环境返回的新状态。
            - **reward** (Union[np.ndarray, Tensor]) - 输入动作后环境返回的奖励。
            - **done** (Union[np.ndarray, Tensor]) - 输入动作后环境是否终止。
            - **env_id** (Union[np.ndarray, Tensor]) - 哪些环境被交互到了。
            - **arg** (Union[np.ndarray, Tensor]) - 支持任意输出，但是用户需要保证它的shape和dtype。

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

    .. py:method:: send(action: Union[Tensor, np.ndarray], env_id: Union[Tensor, np.ndarray])

        执行环境Step函数来和环境交互一回合。

        参数：
            - **action** (Union[Tensor, np.ndarray]) - 一个包含动作信息的Tensor或者array。
            - **env_id** (Union[Tensor, np.ndarray]) - 与哪些环境交互。

        返回：
            - **Success** (bool) - 是否传输的动作成功和环境交互。

    .. py:method:: set_seed(seed_value: Union[int, Sequence[int]])

        设置种子去控制环境的随机性。

        参数：
            - **seed_value** (Union[int, Sequence[int]]) - 用于设置的种子值。

        返回：
            - **Success** (bool) - 是否成功设置种子。

    .. py:method:: step(action: Union[Tensor, np.ndarray])

        执行环境Step函数来和环境交互一回合。

        参数：
            - **action** (Union[Tensor, np.ndarray]) - 包含动作信息的Tensor。

        返回：
            - **state** (Tensor) - 输入动作后的环境返回的新状态。
            - **reward** (Tensor) - 输入动作后环境返回的奖励。
            - **done** (Tensor) - 输入动作后环境是否终止。
            - **other** (Tensor) - \_step方法中剩下的返回值。
