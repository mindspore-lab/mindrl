
.. py:class:: mindspore_rl.core.MSRL(alg_config, deploy_config=None)

    MSRL提供了用于强化学习算法开发的方法和API。
    它向用户公开以下方法。这些方法的输入和输出与用户定义的方法相同。

    .. code-block::

        agent_act
        agent_get_action
        sample_buffer
        agent_learn
        replay_buffer_sample
        replay_buffer_insert
        replay_buffer_reset

    参数：
        - **alg_config** (dict) - 提供算法配置。
        - **deploy_config** (dict) - 提供分布式配置。

          - **顶层** - 定义算法组件。

        - 关键字: `actor`， 值： actor的配置 (dict)。
        - 关键字: `learner`， 值： learner的配置 (dict)。
        - 关键字: `policy_and_network`， 值： actor和learner使用的策略和网络 (dict)。
        - 关键字: `collect_environment`， 值： 收集环境的配置 (dict)。
        - 关键字: `eval_environment`， 值： 评估环境的配置 (dict)。
        - 关键字: `replay_buffer`， 值： 重放缓存的配置 (dict)。

          - **第二层** - 每个算法组件的配置。

        - 关键字: `number`， 值： actor/learner的数量 (int)。
        - 关键字: `type`， 值： actor/learner/policy_and_network/environment (class)。
        - 关键字: `params`， 值： actor/learner/policy_and_network/environment的参数 (dict)。
        - 关键字: `policies`， 值： actor/learner使用的策略列表 (list)。
        - 关键字: `networks`， 值： actor/learner使用的网络列表 (list)。
        - 关键字: `pass_environment`， 值： 如果为 ``True``， 用户需要传递环境实例给actor， 为 ``False`` 则不需要 (bool)。

    .. py:method:: create_environments(config, env_type, deploy_config=None, need_batch=False)

        通过配置文件创建环境，并且返回环境实例和环境个数。

        参数：
            - **config** (dict) - 算法的配置文件。
            - **env_type** (str) - 环境的类型，可以是 ``collect_environment`` 或 ``eval_environment``。
            - **deploy_config** (dict，可选) - 提供分布式配置。默认：``None``。
            - **need_batched** (bool，可选) - 是否需要批量环境。默认：``False``。

    .. py:method:: get_replay_buffer

        返回重放缓存的实例。

        返回：
            - **buffers** (object) - 重放缓存的实例。如果缓存为 ``None``， 返回也为 ``None``。

    .. py:method:: get_replay_buffer_elements(transpose=False, shape=None)

        返回重放缓存中的所有元素。

        参数：
            - **transpose** (bool) - 输出元素是否需要转置，如果为 ``True``，则shape也需指定。默认值：``False``。
            - **shape** (tuple[int]) - 转置的shape。默认值：``None``。

        返回：
            - **elements** (List[Tensor]) - 一组包含所有重放缓存中数据的张量。

    .. py:method:: init(config)

        MSRL 对象的初始化。该方法创建算法所需的所有数据/对象。它会初始化所有的方法。

        参数：
            - **config** (dict) - 算法的配置文件。
