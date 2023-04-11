
.. py:class:: mindspore_rl.utils.TensorsQueue(dtype, shapes, size=0, name="TQ")

    用来存TensorsQueue的队列。

    .. warning::
        - 这是一个实验特性，未来有可能被修改或删除。

    参数：
        - **dtype** (mindspore.dtype) - TensorsQueue的数据类型。每个Tensor需要相同的类型。
        - **shapes** (tuple[int64]) - TensorsQueue中每个Tensor的shape。
        - **size** (int，可选) - TensorsQueue的大小。默认：0。
        - **name** (str，可选) - TensorsQueue的名字。默认："TQ"。

    异常：
        - **TypeError** - `dtype` 不是 MindSpore 数字类型.
        - **ValueError** - `size` 小于0.
        - **ValueError** - `shapes` 的长度小于1.

    .. py:method:: clear()

        清理创建的TensorsQueue。仅重置该队列，清理数据和重置大小，保留队列实例。

        返回：
            True。

    .. py:method:: close()

        关闭TensorsQueue。

        .. warning::
            - 一旦关闭了TensorsQueue，每个属于该TensorsQueue的方法都将失效。所有该队列中的资源也将被清除。如果该队列还将在别的地方使用，如下一个循环，请用 `clear` 代替。

        返回：
            True。

    .. py:method:: get()

        从TensorsQueue的头部取出一个元素。

        返回：
            tuple(Tensor)，一个元素。

    .. py:method:: pop()

        从TensorsQueue的头部取出一个元素并删除。

        返回：
            tuple(Tensor)，一个元素。

    .. py:method:: put(element)

        向TensorsQueue的底部放入元素（tuple(Tensors)）。

        参数：
            - **element** (tuple(Tensor) 或 list[tensor]) - 写入的元素。

        返回：
            True。

    .. py:method:: size()

        TensorsQueue的已使用大小。

        返回：
            Tensor(mindspore.int64)，TensorsQueue的已使用大小。
