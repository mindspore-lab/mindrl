
.. py:class:: mindspore_rl.utils.CallbackParam()
    
    包含回调函数执行时需要的参数。


.. py:class:: mindspore_rl.utils.CallbackManager(callbacks)

    依次执行回调函数。

    参数：
        - **callbacks** (list[Callback]) - 一个包含回调函数的list。

    .. py:method:: begin(params)

        在训练执行开始调用，仅执行一次。

        参数：
            - **params** (CallbackParam) - begin执行用的参数。

    .. py:method:: end(params)

        在训练执行结束调用，仅执行一次。

        参数：
            - **params** (CallbackParam) - end执行用的参数。

    .. py:method:: episode_begin(params)

        在每个episode执行前调用。

        参数：
            - **params** (CallbackParam) - episode_begin执行用的参数。

    .. py:method:: episode_end(params)

        在每个episode执行后调用。

        参数：
            - **params** (CallbackParam) - episode_end执行用的参数。


.. py:class:: mindspore_rl.utils.LossCallback(print_rate=1)

    在每个episode结束时打印loss值。

    参数：
        - **print_rate** (int, 可选) - 打印loss的频率。默认值： ``1`` 。

    .. py:method:: episode_end(params)

        在每个episode执行后调用，打印loss值。

        参数：
            - **params** (CallbackParam) - 训练参数，用于获取结果。


.. py:class:: mindspore_rl.utils.TimeCallback(print_rate=1, fixed_steps_in_episode=None)

    在每个episode结束时打印耗时。

    参数：
        - **print_rate** (int, 可选) - 打印耗时的频率，默认值： ``1`` 。
        - **fixed_steps_in_episode** (int, 可选) - 如果每个episode的steps是固定的，则提供一个固定steps值。如果是 ``None`` ，params中需要提供实际steps。默认值： ``None`` 。

    .. py:method:: episode_begin(params)

        在每个episode执行前调用，打印耗时。

        参数：
            - **params** (CallbackParam) - 训练参数，用于获取结果。

    .. py:method:: episode_end(params)

        在每个episode执行后记录时间。

        参数：
            - **params** (CallbackParam) - 训练参数，用于获取结果。

.. py:class:: mindspore_rl.utils.CheckpointCallback(save_per_episode=0, directory=None, max_ckpt_nums=5)

    保存模型的checkpoint文件，保留最新的 `max_ckpt_nums` 个。

    参数：
        - **save_per_episode** (int, 可选) - 保存ckpt文件的频率。默认值： ``0`` （不保存）。
        - **directory** (str, 可选) - 保存ckpt文件的路径。默认： ``None`` ，保存至 ``'./'`` 路径。
        - **max_ckpt_nums** (int, 可选) - 最大保留ckpt的个数。默认值： ``5`` 。

    .. py:method:: episode_end(params)

        在每个episode执行后调用，保存ckpt文件。

        参数：
            - **params** (CallbackParam) - 训练参数，用于获取结果。

.. py:class:: mindspore_rl.utils.EvaluateCallback(eval_rate=0)

    推理回调。

    参数：
        - **eval_rate** (int, 可选) - 推理的频率。默认值： ``0`` （不推理）。

    .. py:method:: begin(params)

        在训练开始前保存推理频率。

        参数：
            - **params** (CallbackParam) - episode开始时用的参数。

    .. py:method:: episode_end(params)

        在每个episode执行后调用，推理并打印结果。

        参数：
            - **params** (CallbackParam) - episode结束后用的参数。
