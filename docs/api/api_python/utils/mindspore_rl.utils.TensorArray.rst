
.. py:class:: mindspore_rl.utils.TensorArray(dtype, element_shape, dynamic_size=True, size=0, name="TA")

    用来存Tensor的TensorArray。

    .. warning::
        - 这是一个实验特性，未来有可能被修改或删除。

    参数：
        - **dtype** (mindspore.dtype) - TensorArray的数据类型。
        - **element_shape** (tuple(int)) - TensorArray中每个Tensor的shape。
        - **dynamic_size** (bool,可选) - 如果是True，则该数组可以动态增长，否则为固定大小。默认：True。
        - **size** (int，可选) - 如果 `dynamic_size=False` , 则 `size` 表示该数组的最大容量。
        - **name** (str，可选) - TensorArray的名字，任意str。默认："TA"。

    .. py:method:: clear()

        清理创建的TensorArray。仅重置该数组，清理数据和重置大小，保留数组实例。

        返回：
            True。

    .. py:method:: close()

        关闭TensorArray。

        .. warning::
            - 一旦关闭了TensorArray，每个属于该TensorArray的方法都将失效。所有该数组中的资源也将被清除。如果该数组还将在别的地方使用，如下一个循环，请用 `clear` 代替。

        返回：
            True。

    .. py:method:: read(index)

        从TensorArray的指定位置读Tensor。

        参数：
            - **index** ([int, mindspore.int64]) - 读取的位置。

        返回：
            Tensor, 指定位置的值。

    .. py:method:: size()

        TensorArray的逻辑大小。

        返回：
            Tensor, TensorArray大小。

    .. py:method:: stack()

        堆叠TensorArray中的Tensor为一个整体。

        返回：
            Tensor, TensorArray中的所有Tensor将堆叠成一个整体。

    .. py:method:: write(index, value)

        向TensorArray的指定位置写入值（Tensor）。

        参数：
            - **index** ([int, mindspore.int64]) - 写入的位置。
            - **value** (Tensor) - 写入的Tensor。

        返回：
            True。