.. py:class:: mindspore_rl.environment.StarCraft2Environment(params, env_id=0)

    StarCraft2Environment是一个SMAC的包装器。SMAC是WhiRL的一个基于暴雪星际争霸2开发的用于多智能体合作场景的强化学习环境。

    SMAC通过调用暴雪星际争霸2的机器学习API和DeepMind的PySC2提供的API，方便算法中的智能体与星际争霸2交互来获得环境的状态和合法的动作。更多的信息请查阅官方的SMAC官方的GitHub：
    <https://github.com/oxwhirl/smac>。

    参数：
        - **params** (dict) - 字典包含StarCraft2Environment类中所需要的所有参数。

          +------------------------------+---------------------------------------------------+
          |  配置参数                    |  备注                                             |
          +==============================+===================================================+
          |  sc2_args                    |  一个用于创建SMAC实例的字典包含一些SMAC需要的key值|
          |                              |  如map_name. 详细配置信息请查看官方GitHub。       |
          +------------------------------+---------------------------------------------------+

        - **env_id** (int，可选) - 环境id，用于设置环境内种子，默认为第0个环境。默认值：0

    .. py:method:: close

        关闭环境以释放环境资源

        返回：
            - **Success** (np.bool\_) - 是否成功释放资源。
